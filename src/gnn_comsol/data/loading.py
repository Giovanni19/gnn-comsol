"""
Reading the COMSOL dataset.

The dataset is produced outside this repository by a COMSOL + MATLAB
LiveLink script and saved as a MATLAB v7.3 file, which is HDF5 and is
therefore read with h5py.

Expected contents
-----------------
X            (3, N, T)   state per node and timestep, as MATLAB stores it
edge_index   (2, E)      mesh connectivity, zero-based node indices
edge_weight  (E,)        weight of each edge
t            (T,)        simulation time of each snapshot
h            scalar/array  local mesh size

The state carries three variables per node, in this order:
    0 -> u   horizontal velocity
    1 -> v   vertical velocity
    2 -> p   pressure
"""

from dataclasses import dataclass

import h5py
import numpy as np
from .normalization import NUM_PHYSICS_FEATURES

@dataclass
class RawDataset:
    """
    One simulation, already arranged as (input, target) pairs.

    With `skip` initial snapshots dropped, sample i is the transition

        X[skip + i]  ->  X[skip + i + 1]

    Attributes
    ----------
    X_input : (S, N, 3)
        State fed to the network.

    Y_target : (S, N, 3)
        State to predict, one snapshot later.

    edge_index : (2, E)
    edge_weight : (E,)

    delta_t : (S,)
        Duration of the transition each sample has to advance:
        delta_t[i] = t[skip + i + 1] - t[skip + i].
        Same length as X_input, index by index.

    h : array
        Mesh size information, currently carried around but unused.
    """

    X_input: np.ndarray
    Y_target: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    delta_t: np.ndarray
    h: np.ndarray
    pos: np.ndarray
    physics_features: np.ndarray | None = None
    simulation_id: int | None = None
    file_path: str | None = None

    @property
    def num_samples(self):
        return len(self.X_input)

    @property
    def num_nodes(self):
        return self.X_input.shape[1]

    @property
    def num_edges(self):
        return self.edge_index.shape[1]


def load_data(file_path, skip_initial=0, simulation_id=None):
    """
    Load one simulation and build the one-step-ahead pairs.

    Parameters
    ----------
    skip_initial : int
        How many snapshots to drop from the start of the simulation.

        The very first snapshot is the initial condition, not a state of
        the flow: it is usually artificial (uniform or zero velocity,
        pressure zero or from a preliminary solve) and does not satisfy
        the governing equations the way a converged step does. The first
        step of an adaptive solver is also atypically small. Training on
        that transition teaches the solver settling in, not the dynamics.

        The cost is one sample per snapshot dropped, so this is cheap.
        Whether 1 is enough depends on how long the solver takes to relax
        the initial condition: plot_pressure_statistics shows it. With a
        temporal split any startup transient sits entirely in the
        training block, so it is worth looking at.

    Returns
    -------
    RawDataset

    Note on delta_t
    ---------------
    delta_t[i] is the duration of the transition sample i has to advance,
    t[skip+i+1] - t[skip+i], and lines up index by index with X_input.

    The original code passed t[skip+i] - t[skip+i-1] instead, that is the
    step that led INTO the input state rather than the one being
    predicted. It only mattered with a variable time step, which adaptive
    solvers normally use.
    """

    with h5py.File(file_path, "r") as f:

        X = np.array(f["X"])
        edge_index = np.array(f["edge_index"])
        edge_weight = np.array(f["edge_weight"])
        t = np.array(f["t"])
        h = np.array(f["h"])
        P = np.array(f["P"])

        if "physics_features" in f:
            physics_features = np.array(
                f["physics_features"]
            )
        else:
            physics_features = None

    # MATLAB stores arrays in Fortran order: (3, N, T) -> (T, N, 3)
    X = np.transpose(X, (2, 1, 0))
    if physics_features is not None:

        # MATLAB/HDF5:
        #     (5, N, T)
        #
        # Python:
        #     (T, N, 5)

        physics_features = np.transpose(
            physics_features,
            (2, 1, 0)
        )
    edge_index = edge_index.astype(np.int64)
    edge_weight = edge_weight.squeeze()
    t = t.squeeze()
    h = h.squeeze()

    num_snapshots = X.shape[0]

    if skip_initial < 0:
        raise ValueError(
            f"skip_initial must be >= 0, got {skip_initial}."
        )

    if skip_initial >= num_snapshots - 1:
        raise ValueError(
            f"skip_initial={skip_initial} leaves no (input, target) "
            f"pairs: the simulation has {num_snapshots} snapshots."
        )

    # step[k] = t[k+1] - t[k], the duration of the transition X[k] -> X[k+1]
    step = t[1:] - t[:-1]

    X_input = X[skip_initial:-1]
    Y_target = X[skip_initial + 1:]
    delta_t = step[skip_initial:]
    if physics_features is not None:

        physics_input = physics_features[
            skip_initial:-1
        ]

    else:

        physics_input = None
    # ------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------
    if physics_input is not None:

        if physics_input.shape[0] != X_input.shape[0]:
            raise ValueError(
                f"{file_path}: physics features have "
                f"{physics_input.shape[0]} samples, "
                f"but X_input has {X_input.shape[0]}."
            )

        if physics_input.shape[1] != X_input.shape[1]:
            raise ValueError(
                f"{file_path}: physics features have "
                f"{physics_input.shape[1]} nodes, "
                f"but X_input has {X_input.shape[1]}."
            )

        if physics_input.shape[2] != NUM_PHYSICS_FEATURES:
            raise ValueError(
                f"{file_path}: expected {NUM_PHYSICS_FEATURES} physics features, "
                f"got {physics_input.shape[2]}."
            )

        if not np.all(np.isfinite(physics_input)):
            raise ValueError(
                f"{file_path}: physics features contain "
                f"NaN or Inf values."
            )
    num_nodes = X_input.shape[1]

    # P may be stored as (N, 2) or (2, N)
    if P.shape[0] == num_nodes:
        pos = P
    elif P.shape[1] == num_nodes:
        pos = P.T
    else:
        raise ValueError(
            f"{file_path}: position array P has shape {P.shape}, "
            f"but the state has {num_nodes} nodes."
        )

    if edge_index.min() < 0:
        raise ValueError(
            f"{file_path}: edge_index contains negative indices."
        )

    if edge_index.max() >= num_nodes:
        raise ValueError(
            f"{file_path}: edge_index references node "
            f"{edge_index.max()}, but the mesh has only "
            f"{num_nodes} nodes."
        )
    return RawDataset(
        X_input=X_input,
        Y_target=Y_target,
        edge_index=edge_index,
        edge_weight=edge_weight,
        delta_t=delta_t,
        h=h,
        pos=pos,
        physics_features=physics_input,
        simulation_id=simulation_id,
        file_path=str(file_path)
    )


def load_simulations(file_paths, skip_initial=0):

    simulations = []

    for simulation_id, file_path in enumerate(file_paths):

        simulation = load_data(
            file_path,
            skip_initial=skip_initial,
            simulation_id=simulation_id,
        )

        simulations.append(simulation)

    return simulations

