"""
Reading the COMSOL dataset.

The dataset is produced outside this repository by a COMSOL + MATLAB
LiveLink script and saved as a MATLAB v7.3 file, which is HDF5 and is
therefore read with h5py.

Expected contents
-----------------
X                  (3, N, T)   state per node and timestep, as MATLAB
                               stores it
edge_index         (2, E)      mesh connectivity, zero-based node
                               indices
edge_weight        (E,)        weight of each edge
t                  (T,)        simulation time of each snapshot
h                  scalar/array  local mesh size
physics_features   (5, N, T)   optional, per node and timestep
geometry_features  (N, 6) or (6, N)   optional, static per node -
                               NOT indexed by time, unlike
                               physics_features

The state carries three variables per node, in this order:
    0 -> u   horizontal velocity
    1 -> v   vertical velocity
    2 -> p   pressure
"""

from dataclasses import dataclass

import h5py
import numpy as np
from .normalization import NUM_GEOMETRY_FEATURES, NUM_PHYSICS_FEATURES

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

    geometry_features : (N, NUM_GEOMETRY_FEATURES) or None
        Static per-node boundary distance/direction features (wall,
        inlet, outlet). One row per mesh node - NOT indexed by time,
        unlike physics_features. None for datasets generated before
        this feature existed.
    """

    X_input: np.ndarray
    Y_target: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    delta_t: np.ndarray
    h: np.ndarray
    pos: np.ndarray

    # WLSQ / physics-loss geometry
    neighbors: list[np.ndarray] | None = None
    G_wlsq: list[np.ndarray] | None = None
    cell_index: np.ndarray | None = None

    # Fluid properties, needed by the momentum residual only. None when
    # the .mat does not carry them, in which case the experiment file
    # has to supply them - see physics.fluid.
    rho: float | None = None
    mu: float | None = None

    physics_features: np.ndarray | None = None
    geometry_features: np.ndarray | None = None
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

def _load_matlab_cell_array(f, key):
    """
    Load a MATLAB cell array stored in a v7.3 HDF5 .mat file.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file.

    key : str
        Name of the MATLAB cell array.

    Returns
    -------
    list[np.ndarray] | None
        Contents of the MATLAB cell array.
    """

    if key not in f:
        return None

    refs = np.array(f[key]).reshape(-1)

    values = []

    for ref in refs:
        values.append(np.array(f[ref]))

    return values


def _load_optional_scalar(f, key):
    """
    One number from the .mat, or None if the file does not have it.

    MATLAB writes a scalar as a 1x1 array, and a value that is present
    but not a single number is a mistake worth reporting rather than
    silently reducing.
    """

    if key not in f:
        return None

    value = np.array(f[key]).reshape(-1)

    if value.size != 1:
        raise ValueError(
            f"{key} must be a single number, "
            f"but has {value.size} elements."
        )

    return float(value[0])


def _process_wlsq(
    neighbors,
    G_wlsq,
    cell_index,
    num_nodes,
    file_path,
):
    """
    Bring the WLSQ export into the layout the physics loss expects.

    MATLAB writes arrays to a v7.3 .mat in Fortran order, so h5py hands
    every one of them back with its dimensions REVERSED - the same rule
    that turns the (T, N, 3) state into (3, N, T). The orientation is
    therefore not guessed from the shape: a G_i of a node with exactly
    two neighbours is (2, 2) either way, and guessing left it silently
    transposed, giving wrong gradients at the corners of the domain.

    Returns the triple (neighbors, G_wlsq, cell_index), any of which may
    be None: the older .mat files predate the WLSQ export.
    """

    # --------------------------------------------------------------
    # Stencils: MATLAB (1, k_i) -> h5py (k_i, 1) -> (k_i,)
    # --------------------------------------------------------------

    if neighbors is not None:

        if len(neighbors) != num_nodes:
            raise ValueError(
                f"{file_path}: neighbors_python contains "
                f"{len(neighbors)} entries, "
                f"but the mesh has {num_nodes} nodes."
            )

        # reshape(-1), not squeeze(): squeeze() on the stencil of a node
        # with a single neighbour would return a 0-d array.
        neighbors = [
            np.asarray(neigh).reshape(-1).astype(np.int64)
            for neigh in neighbors
        ]

        for node_i, neigh in enumerate(neighbors):

            if neigh.size < 2:
                raise ValueError(
                    f"{file_path}: node {node_i} has {neigh.size} WLSQ "
                    "neighbor(s); at least 2 are needed to reconstruct "
                    "a 2D gradient."
                )

            if np.any(neigh < 0) or np.any(neigh >= num_nodes):
                raise ValueError(
                    f"{file_path}: invalid WLSQ neighbor "
                    f"indices for node {node_i}."
                )

    # --------------------------------------------------------------
    # Operators: MATLAB (2, k_i) -> h5py (k_i, 2) -> (2, k_i)
    # --------------------------------------------------------------

    if G_wlsq is not None:

        if neighbors is None:
            raise ValueError(
                f"{file_path}: G_wlsq exists but "
                f"neighbors_python is missing."
            )

        if len(G_wlsq) != num_nodes:
            raise ValueError(
                f"{file_path}: G_wlsq contains "
                f"{len(G_wlsq)} operators, "
                f"but the mesh has {num_nodes} nodes."
            )

        processed_G = []

        for node_i, G_i in enumerate(G_wlsq):

            G_i = np.asarray(G_i, dtype=np.float64)

            k_i = neighbors[node_i].size

            if G_i.shape != (k_i, 2):
                raise ValueError(
                    f"{file_path}: G_wlsq[{node_i}] has shape "
                    f"{G_i.shape}; MATLAB writes it as (2, {k_i}), "
                    f"so h5py must read it back as ({k_i}, 2)."
                )

            processed_G.append(G_i.T)

        G_wlsq = processed_G

    # --------------------------------------------------------------
    # Cells: MATLAB (Nc, 3) -> h5py (3, Nc) -> (Nc, 3)
    # --------------------------------------------------------------

    if cell_index is not None:

        if cell_index.shape[0] != 3:
            raise ValueError(
                f"{file_path}: cell_index has shape "
                f"{cell_index.shape}; MATLAB writes it as (Nc, 3), so "
                "h5py must read it back as (3, Nc)."
            )

        cell_index = cell_index.T.astype(np.int64)

        if np.any(cell_index < 0):
            raise ValueError(
                f"{file_path}: cell_index contains "
                f"negative indices. It must be zero-based: "
                f"first_database.m exports T.' - 1."
            )

        if np.any(cell_index >= num_nodes):
            raise ValueError(
                f"{file_path}: cell_index references "
                f"a node outside the mesh."
            )

    return neighbors, G_wlsq, cell_index


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
        cell_index = (
            np.array(f["cell_index"])
            if "cell_index" in f
            else None
        )
        neighbors = _load_matlab_cell_array(
            f,
            "neighbors_python",
        )

        G_wlsq = _load_matlab_cell_array(
            f,
            "G_wlsq",
        )

        # Written by first_database.m when it manages to read them out
        # of the COMSOL model, absent otherwise: the variable names
        # they are read from belong to the model, not to this code.
        rho = _load_optional_scalar(f, "rho")
        mu = _load_optional_scalar(f, "mu")
        if "physics_features" in f:
            physics_features = np.array(
                f["physics_features"]
            )
        else:
            physics_features = None

        if "geometry_features" in f:
            geometry_features = np.array(
                f["geometry_features"]
            )
        else:
            geometry_features = None

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

    # ------------------------------------------------------------
    # WLSQ geometry for the physics loss
    # ------------------------------------------------------------

    neighbors, G_wlsq, cell_index = _process_wlsq(
        neighbors,
        G_wlsq,
        cell_index,
        num_nodes,
        file_path,
    )

    # ------------------------------------------------------------
    # Position array
    # ------------------------------------------------------------

    if P.shape[0] == num_nodes:
        pos = P

    elif P.shape[1] == num_nodes:
        pos = P.T

    else:
        raise ValueError(
            f"{file_path}: position array P has shape {P.shape}, "
            f"but the state has {num_nodes} nodes."
        )

    # ------------------------------------------------------------
    # Geometry features: static per node, not per timestep.
    # ------------------------------------------------------------
    if geometry_features is not None:

        # Saved by MATLAB as (N, NUM_GEOMETRY_FEATURES); HDF5/h5py may
        # hand it back transposed, exactly like P above.
        if geometry_features.shape[0] == num_nodes:
            pass
        elif geometry_features.shape[1] == num_nodes:
            geometry_features = geometry_features.T
        else:
            raise ValueError(
                f"{file_path}: geometry_features has shape "
                f"{geometry_features.shape}, but the state has "
                f"{num_nodes} nodes."
            )

        if geometry_features.shape[1] != NUM_GEOMETRY_FEATURES:
            raise ValueError(
                f"{file_path}: expected {NUM_GEOMETRY_FEATURES} "
                f"geometry features, got {geometry_features.shape[1]}."
            )

        if not np.all(np.isfinite(geometry_features)):
            raise ValueError(
                f"{file_path}: geometry_features contain "
                f"NaN or Inf values."
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

        # WLSQ / physics-loss geometry
        neighbors=neighbors,
        G_wlsq=G_wlsq,
        cell_index=cell_index,
        rho=rho,
        mu=mu,

        physics_features=physics_input,
        geometry_features=geometry_features,
        simulation_id=simulation_id,
        file_path=str(file_path),
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

