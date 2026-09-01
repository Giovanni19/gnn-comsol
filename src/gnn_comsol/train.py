"""
The training loop.

There used to be three near-identical copies of this file, differing only
in how they picked the target columns, and the `output_type` argument
meant something different in each. There is now one loop, and the target
selection is a single explicit argument.
"""

import copy

import numpy as np
import torch

def _prepare_batch(batch, net, device, target_columns):
    """
    Prepare either a standard PyG batch or a BSMS tensor batch.

    Standard PyG models:
        batch -> Data
        prediction = net(batch)

    BSMS models:
        batch -> (X, Y)
        prediction = net(X)
    """

    if isinstance(batch, (list, tuple)):

        X, Y = batch

        X = X.to(device)
        Y = Y.to(device)

        preds = net(X)
        target = Y[..., target_columns]

    else:

        batch = batch.to(device)

        preds = net(batch)
        target = batch.y[:, target_columns]

    return preds, target

def train_network(
    net,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    target_columns=slice(None),
    verbose=True
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

    Returns
    -------
    best_model_state : dict
        Weights at the epoch with the lowest validation loss.

    train_loss_history, val_loss_history : list of float

    Note: this keeps the best checkpoint but does not stop early; the
    loop always runs for num_epochs.
    """

    best_val_loss = np.inf
    best_model_state = None

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):

        # ---------------------------------------------------
        # Training
        # ---------------------------------------------------

        net.train()

        train_losses = []

        for batch in train_loader:

            optimizer.zero_grad()

            preds, target = _prepare_batch(
                batch,
                net,
                device,
                target_columns,
            )

            loss = criterion(preds, target)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))

        # ---------------------------------------------------
        # Validation
        # ---------------------------------------------------

        net.eval()

        val_losses = []

        with torch.no_grad():

            for batch in val_loader:

                preds, target = _prepare_batch(
                    batch,
                    net,
                    device,
                    target_columns,
                )

                val_losses.append(
                    criterion(preds, target).item()
                )
        mean_val_loss = float(np.mean(val_losses))

        train_loss_history.append(mean_train_loss)
        val_loss_history.append(mean_val_loss)

        if verbose:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {mean_train_loss:.6e} | "
                f"Val Loss: {mean_val_loss:.6e}"
            )

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
