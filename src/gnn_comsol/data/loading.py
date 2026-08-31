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

    @property
    def num_samples(self):
        return len(self.X_input)

    @property
    def num_nodes(self):
        return self.X_input.shape[1]

    @property
    def num_edges(self):
        return self.edge_index.shape[1]


def load_data(file_path, skip_initial=1):
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

    # MATLAB stores arrays in Fortran order: (3, N, T) -> (T, N, 3)
    X = np.transpose(X, (2, 1, 0))

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

    return RawDataset(
        X_input=X_input,
        Y_target=Y_target,
        edge_index=edge_index,
        edge_weight=edge_weight,
        delta_t=delta_t,
        h=h,
        pos=P
    )
