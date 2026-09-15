"""
The training loop.

There used to be three near-identical copies of this file, differing only
in how they picked the target columns, and the `output_type` argument
meant something different in each. There is now one loop, and the target
selection is a single explicit argument.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from gnn_comsol.data.normalization import (
    PRESSURE_COLUMNS,
    STATE_COLUMNS,
    VELOCITY_COLUMNS,
)
from gnn_comsol.physics import continuity_loss, momentum_loss

def _prepare_batch(batch, net, device, target_columns):
    """
    Prepare either a standard PyG batch or a BSMS tensor batch.

    Standard PyG models:
        batch -> Data
        prediction = net(batch)

    BSMS models:
        batch -> (X, Y)
        prediction = net(X)

    Returns (batch, preds, target), where `batch` is the one ON THE
    DEVICE: Data.to() is out of place, so a caller that needs more of
    the batch than the prediction - the physics loss needs its node
    features and its graph boundaries - must be handed the moved copy
    rather than the original.
    """

    if isinstance(batch, (list, tuple)):

        X, Y = batch

        X = X.to(device)
        Y = Y.to(device)

        preds = net(X)
        target = Y[..., target_columns]

        return (X, Y), preds, target

    batch = batch.to(device)

    preds = net(batch)
    target = batch.y[:, target_columns]

    return batch, preds, target
@dataclass
class PhysicsLoss:
    """
    The PDE residuals, ready to be added to the training loss.

    It is an object rather than a pile of keyword arguments because the
    residuals need several things that must agree with each other, and
    a mismatch between them is silent: the network's predictions are in
    NORMALIZED units, the residuals are physical quantities, and how one
    is turned into the other depends on whether the network predicts the
    absolute next state or the increment.

    Strategy: this is the DECOUPLED form of the physics loss. A velocity
    network is scored on the continuity residual of its own prediction
    and on the momentum residual of that prediction against the TRUE
    pressure of the dataset, which acts as a teacher. The two networks
    are still trained one after the other; coupling them - both
    optimized together on a residual that contains both predictions - is
    a separate training loop and is not this.

    Attributes
    ----------
    geometries : dict
        simulation_id -> physics.PhysicsGeometry.

    normalizer : StateNormalizer
        Scaling of the absolute state.

    continuity_weight, momentum_weight : float
        Multiply their residual in the total loss. Zero switches a term
        off entirely, including its cost.

    delta_normalizer : StateNormalizer or None
        Scaling of the increment. Required when predict_delta is True,
        and unused otherwise: an increment is not the same quantity as
        an absolute state and does not share its mean and std.

    predict_delta : bool
        Whether the network output, and the target it is compared
        against, hold the normalized increment rather than the
        normalized next state.

    enforce_boundary_values : bool
        Replace the predicted velocity at boundary nodes with the true
        one before computing the residuals. This is the hard imposition
        of boundary conditions of Sec. 2.4 of the paper: the WLSQ
        stencil of a boundary node is one-sided and its gradient is
        only first-order accurate, so the residual there measures the
        discretization more than it measures the prediction. The
        boundary values are known - they are the boundary conditions of
        the simulation - so there is nothing to learn at those nodes.
    """

    geometries: dict
    normalizer: object
    continuity_weight: float = 0.0
    momentum_weight: float = 0.0
    delta_normalizer: object = None
    predict_delta: bool = False
    enforce_boundary_values: bool = True

    def __post_init__(self):

        if self.continuity_weight <= 0.0 and self.momentum_weight <= 0.0:
            raise ValueError(
                "A PhysicsLoss with no positive weight does nothing; "
                "pass physics=None instead."
            )

        if not self.geometries:
            raise ValueError(
                "The PDE residuals need the WLSQ geometry of every "
                "simulation, and none was given. The .mat files must "
                "carry neighbors_python, G_wlsq and cell_index."
            )

        if self.normalizer is None:
            raise ValueError(
                "The residuals are computed in physical units, so they "
                "need the state normalizer."
            )

        if self.predict_delta and self.delta_normalizer is None:
            raise ValueError(
                "predict_delta is True, so the predictions are "
                "normalized INCREMENTS and the delta normalizer is "
                "needed to turn them back into a state. Scaling them "
                "with the state normalizer would silently produce a "
                "meaningless field."
            )

    @property
    def terms(self):
        """Names of the active residuals, in reporting order."""

        active = []

        if self.continuity_weight > 0.0:
            active.append("continuity")

        if self.momentum_weight > 0.0:
            active.append("momentum")

        return active

    def to_physical(self, normalized, columns, current=None):
        """
        Normalized network output or target -> physical units.

        `current` is the physical state the increment applies to, and
        is required exactly when the quantity is an increment.
        """

        if not self.predict_delta:

            return self.normalizer.inverse_transform(
                normalized,
                columns=columns,
            )

        if current is None:
            raise ValueError(
                "An increment cannot be turned into a state without "
                "the state it is an increment of."
            )

        return current + self.delta_normalizer.inverse_transform(
            normalized,
            columns=columns,
        )

    def __call__(self, batch, preds):
        """
        (weighted total, {name: unweighted residual}) for one batch.

        Each graph may come from a different simulation and therefore
        from a different mesh, so the residuals are computed per graph.
        """

        totals = {name: [] for name in self.terms}

        for graph_i in range(batch.num_graphs):

            # Nodes of this graph inside the batch
            start = int(batch.ptr[graph_i].item())
            end = int(batch.ptr[graph_i + 1].item())

            geometry = self._geometry_of(batch, graph_i)

            for name, value in self._residuals_of_graph(
                batch,
                preds,
                graph_i,
                start,
                end,
                geometry,
            ).items():

                totals[name].append(value)

        components = {
            name: torch.stack(values).mean()
            for name, values in totals.items()
        }

        weights = {
            "continuity": self.continuity_weight,
            "momentum": self.momentum_weight,
        }

        total = sum(
            weights[name] * value
            for name, value in components.items()
        )

        return total, components

    def _geometry_of(self, batch, graph_i):

        simulation_id = int(batch.simulation_id[graph_i].item())

        geometry = self.geometries.get(simulation_id)

        if geometry is None:
            raise KeyError(
                f"Simulation {simulation_id} has no WLSQ geometry, so "
                "the PDE residuals cannot be computed for it. "
                "Regenerate its .mat with first_database.m, or set the "
                "physics weights to 0."
            )

        return geometry

    def _residuals_of_graph(
        self,
        batch,
        preds,
        graph_i,
        start,
        end,
        geometry,
    ):
        """The unweighted residuals of one graph."""

        device = preds.device
        dtype = preds.dtype

        operators = geometry.operators(device=device, dtype=dtype)
        cells = geometry.cells(device=device, dtype=dtype)

        # The current state, in physical units. Both feature encodings
        # put the state first, see data.features.
        state_now = self.normalizer.inverse_transform(
            batch.x[start:end, STATE_COLUMNS],
            columns=STATE_COLUMNS,
        )

        velocity_now = state_now[:, VELOCITY_COLUMNS]

        velocity_next = self.to_physical(
            preds[start:end],
            VELOCITY_COLUMNS,
            current=velocity_now,
        )

        if self.enforce_boundary_values:

            true_velocity_next = self.to_physical(
                batch.y[start:end, VELOCITY_COLUMNS],
                VELOCITY_COLUMNS,
                current=velocity_now,
            )

            velocity_next = torch.where(
                cells.boundary_node.unsqueeze(1),
                true_velocity_next,
                velocity_next,
            )

        residuals = {}

        if self.continuity_weight > 0.0:

            residuals["continuity"] = continuity_loss(
                velocity_next[:, 0],
                velocity_next[:, 1],
                operators,
                cells,
            )

        if self.momentum_weight > 0.0:

            residuals["momentum"] = momentum_loss(
                velocity_now,
                velocity_next,
                self._true_pressure_next(
                    batch,
                    start,
                    end,
                    state_now,
                ),
                self._delta_t(batch, graph_i),
                geometry.require_fluid(),
                operators,
                cells,
            )

        return residuals

    def _true_pressure_next(self, batch, start, end, state_now):
        """
        The pressure at t + dt, from the dataset.

        Strategy A: the velocity network is scored against the pressure
        COMSOL computed, not against the one the pressure network
        predicts. The residual then has a single unknown in it, and a
        velocity network cannot lower it by exploiting a bad pressure.
        """

        return self.to_physical(
            batch.y[start:end, PRESSURE_COLUMNS],
            PRESSURE_COLUMNS,
            current=state_now[:, PRESSURE_COLUMNS],
        ).squeeze(1)

    @staticmethod
    def _delta_t(batch, graph_i):
        """The physical duration of this transition, in seconds."""

        delta_t = getattr(batch, "dt_physical", None)

        if delta_t is None:
            raise ValueError(
                "The momentum residual has a time derivative in it and "
                "needs the physical timestep, which this batch does "
                "not carry. create_graph_dataset must be given "
                "delta_t."
            )

        return delta_t[graph_i]


class _TermAccumulator:
    """
    Running mean of every named term of the loss, for the epoch line.

    A physics-informed run has at least three numbers worth watching -
    the data term, each residual, and their total - and reporting only
    the total hides the case the weight is there to prevent: a residual
    that has quietly taken over the fit.
    """

    def __init__(self):
        self.sums = {}
        self.count = 0

    def add(self, terms):

        for name, value in terms.items():
            self.sums[name] = self.sums.get(name, 0.0) + value.item()

        self.count += 1

    def report(self):

        if not self.count:
            return "no batches"

        return " | ".join(
            f"{name} {total / self.count:.6e}"
            for name, total in self.sums.items()
        )


def train_network(
    net,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    target_columns=slice(None),
    physics=None,
    verbose=True,
):
    """
    Train a network and keep the weights with the lowest validation loss.

    Parameters
    ----------
    target_columns : slice
        Which columns of batch.y this network is responsible for.
        VELOCITY_COLUMNS, PRESSURE_COLUMNS or STATE_COLUMNS from
        gnn_comsol.data.normalization. The datasets always carry the full
        normalized state as the target, so one dataset can serve networks
        predicting different variables.

    physics : PhysicsLoss or None
        Physics-informed residuals added to the data loss. None trains
        on the data alone. It is the CALLER's job to build it only for
        a network whose output is a velocity: the residuals read
        columns 0 and 1 of the prediction as u and v.

    Returns
    -------
    best_model_state : dict
        Weights at the epoch with the lowest validation loss.

    train_loss_history, val_loss_history : list of float
        The TOTAL loss, data plus weighted physics. Every term is also
        reported separately at each epoch: a total alone cannot say
        whether a residual is helping or quietly dominating the fit.

    Note: this keeps the best checkpoint but does not stop early; the
    loop always runs for num_epochs.
    """

    best_val_loss = np.inf
    best_model_state = None

    train_loss_history = []
    val_loss_history = []

    def losses_for(batch):
        """(total, {term: value}) for one batch."""

        batch, preds, target = _prepare_batch(
            batch,
            net,
            device,
            target_columns,
        )

        loss_data = criterion(preds, target)

        if physics is None:
            return loss_data, {"data": loss_data}

        weighted, components = physics(batch, preds)

        total = loss_data + weighted

        return total, {"data": loss_data, **components}

    for epoch in range(num_epochs):

        # ---------------------------------------------------
        # Training
        # ---------------------------------------------------

        net.train()

        train_totals = []
        train_terms = _TermAccumulator()

        for batch in train_loader:

            optimizer.zero_grad()

            total, terms = losses_for(batch)

            total.backward()
            optimizer.step()

            train_totals.append(total.item())
            train_terms.add(terms)

        mean_train_loss = float(np.mean(train_totals))

        # ---------------------------------------------------
        # Validation
        #
        # Scored on the SAME objective that is being optimized, so
        # that the two curves are comparable and the checkpoint kept
        # is the best one for the objective rather than for one of its
        # two terms.
        # ---------------------------------------------------

        net.eval()

        val_totals = []
        val_terms = _TermAccumulator()

        with torch.no_grad():

            for batch in val_loader:

                total, terms = losses_for(batch)

                val_totals.append(total.item())
                val_terms.add(terms)

        mean_val_loss = float(np.mean(val_totals))

        train_loss_history.append(mean_train_loss)
        val_loss_history.append(mean_val_loss)

        if verbose:

            message = (
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {mean_train_loss:.6e} | "
                f"Val Loss: {mean_val_loss:.6e}"
            )

            if physics is not None:

                message += (
                    f"\n    train: {train_terms.report()}"
                    f"\n    val:   {val_terms.report()}"
                )

            print(message)

        if mean_val_loss < best_val_loss:

            best_val_loss = mean_val_loss
            best_model_state = copy.deepcopy(net.state_dict())

    return best_model_state, train_loss_history, val_loss_history


def train_bsms_multi_simulation(
    net,
    train_loaders,
    val_loaders,
    hierarchies,
    criterion,
    optimizer,
    num_epochs,
    device,
    target_columns=slice(None),
    verbose=True,
):
    """
    Train one BSMS network on multiple simulations with different meshes.

    Each simulation has:
        - its own DataLoader;
        - its own BSMS hierarchy.

    The neural-network parameters are shared across all simulations.

    Parameters
    ----------
    train_loaders : dict
        simulation_id -> TensorDataLoader

    val_loaders : dict
        simulation_id -> TensorDataLoader

    hierarchies : dict
        simulation_id -> {
            "edge_indices": [...],
            "pool_indices": [...],
            "pos": Tensor
        }

    target_columns : slice
        Columns of Y predicted by this network.

    Returns
    -------
    best_model_state : dict

    train_loss_history : list[float]

    val_loss_history : list[float]
    """

    best_val_loss = np.inf
    best_model_state = None

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):

        # =====================================================
        # TRAINING
        # =====================================================

        net.train()

        train_loss_sum = 0.0
        train_sample_count = 0

        # Randomize simulation order at every epoch.
        simulation_ids = list(
            train_loaders.keys()
        )

        np.random.shuffle(simulation_ids)

        for simulation_id in simulation_ids:

            loader = train_loaders[
                simulation_id
            ]

            hierarchy = hierarchies[
                simulation_id
            ]

            # Move this mesh hierarchy to the device once
            # for this simulation.
            edge_indices = [
                edge_index.to(device)
                for edge_index
                in hierarchy["edge_indices"]
            ]

            pool_indices = [
                indices.to(device)
                for indices
                in hierarchy["pool_indices"]
            ]

            pos = hierarchy["pos"].to(device)

            for X, Y in loader:

                X = X.to(device)
                Y = Y.to(device)

                optimizer.zero_grad()

                preds = net(
                    X,
                    pool_indices,
                    edge_indices,
                    pos,
                )

                target = Y[
                    ...,
                    target_columns
                ]

                loss = criterion(
                    preds,
                    target,
                )

                loss.backward()

                optimizer.step()

                # Weight the epoch mean by number of samples,
                # not number of batches.
                batch_size = X.shape[0]

                train_loss_sum += (
                    loss.item() * batch_size
                )

                train_sample_count += batch_size

        mean_train_loss = (
            train_loss_sum
            / train_sample_count
        )

        # =====================================================
        # VALIDATION
        # =====================================================

        net.eval()

        val_loss_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():

            for simulation_id, loader in (
                val_loaders.items()
            ):

                hierarchy = hierarchies[
                    simulation_id
                ]

                edge_indices = [
                    edge_index.to(device)
                    for edge_index
                    in hierarchy["edge_indices"]
                ]

                pool_indices = [
                    indices.to(device)
                    for indices
                    in hierarchy["pool_indices"]
                ]

                pos = hierarchy["pos"].to(
                    device
                )

                for X, Y in loader:

                    X = X.to(device)
                    Y = Y.to(device)

                    preds = net(
                        X,
                        pool_indices,
                        edge_indices,
                        pos,
                    )

                    target = Y[
                        ...,
                        target_columns
                    ]

                    loss = criterion(
                        preds,
                        target,
                    )

                    batch_size = X.shape[0]

                    val_loss_sum += (
                        loss.item() * batch_size
                    )

                    val_sample_count += (
                        batch_size
                    )

        mean_val_loss = (
            val_loss_sum
            / val_sample_count
        )

        # =====================================================
        # HISTORY
        # =====================================================

        train_loss_history.append(
            mean_train_loss
        )

        val_loss_history.append(
            mean_val_loss
        )

        if verbose:

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: "
                f"{mean_train_loss:.6e} | "
                f"Val Loss: "
                f"{mean_val_loss:.6e}"
            )

        # =====================================================
        # BEST CHECKPOINT
        # =====================================================

        if mean_val_loss < best_val_loss:

            best_val_loss = mean_val_loss

            best_model_state = (
                copy.deepcopy(
                    net.state_dict()
                )
            )

    return (
        best_model_state,
        train_loss_history,
        val_loss_history,
    )
