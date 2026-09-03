"""
Train / validation / test splitting.

The default is a temporal split. Consecutive snapshots of the same
simulation are almost identical, so a shuffled split puts near-duplicates
of the training samples into validation and test and reports an
over-optimistic error: on this dataset the nearest training neighbour of
a test sample was one single snapshot away.
"""

from dataclasses import dataclass, replace

import numpy as np

from .normalization import VARIABLE_NAMES


@dataclass
class Split:
    """One block of the dataset, plus the indices it came from."""

    X: np.ndarray
    Y: np.ndarray
    dt: np.ndarray
    indices: np.ndarray
    physics_features: np.ndarray | None = None

    def __len__(self):
        return len(self.X)


@dataclass
class SplitDataset:
    """The three blocks."""

    train: Split
    val: Split
    test: Split


def compute_split_indices(
    n_samples,
    mode="temporal",
    train_fraction=0.70,
    val_fraction=0.15,
    gap=1,
    groups=None,
    seed=68
):
    """
    Compute train / validation / test indices.

    Parameters
    ----------
    mode : {"temporal", "group", "random"}

        "temporal"
            Contiguous blocks in time: the first snapshots go to
            training, the last ones to test. Measures extrapolation in
            time, which is what a surrogate is actually asked to do.

        "group"
            Whole simulations are kept together, so no simulation appears
            in more than one block. The correct protocol once several
            simulations are available. Requires `groups`.

        "random"
            Legacy shuffled split. Kept only to reproduce the old,
            over-optimistic numbers for comparison.

    gap : int
        "temporal" only. Samples dropped between two consecutive blocks.
        Sample i is the pair X[i+1] -> X[i+2], so without a gap the
        snapshot X[n_train+1] is at the same time the target of the last
        training sample and the input of the first validation sample.
        gap=1 removes exactly that overlap.

    groups : array-like or None
        "group" only. One simulation id per sample.

    seed : int
        Used by "random" (shuffles samples) and "group" (shuffles whole
        simulations). "temporal" is deterministic and ignores it.

    Returns
    -------
    train_indices, val_indices, test_indices : numpy.ndarray
    """

    if mode == "temporal":

        n_train = int(train_fraction * n_samples)
        n_val = int(val_fraction * n_samples)

        train_end = n_train
        val_start = train_end + gap
        val_end = val_start + n_val
        test_start = val_end + gap

        indices = np.arange(n_samples)

        train_indices = indices[:train_end]
        val_indices = indices[val_start:val_end]
        test_indices = indices[test_start:]

        if (
            len(train_indices) == 0
            or len(val_indices) == 0
            or len(test_indices) == 0
        ):
            raise ValueError(
                f"Temporal split leaves an empty block: "
                f"n_samples={n_samples}, "
                f"train_fraction={train_fraction}, "
                f"val_fraction={val_fraction}, gap={gap}."
            )

        return train_indices, val_indices, test_indices

    if mode == "group":

        if groups is None:
            raise ValueError(
                "mode='group' requires `groups`: "
                "one simulation id per sample."
            )

        groups = np.asarray(groups)

        if len(groups) != n_samples:
            raise ValueError(
                f"`groups` has {len(groups)} entries "
                f"but there are {n_samples} samples."
            )

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if n_groups < 3:
            raise ValueError(
                f"mode='group' needs at least 3 simulations, "
                f"found {n_groups}."
            )

        # Shuffle whole simulations, never individual snapshots
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_groups)

        n_train_groups = max(1, int(train_fraction * n_groups))
        n_train_groups = min(n_train_groups, n_groups - 2)

        n_val_groups = max(1, int(val_fraction * n_groups))
        n_val_groups = min(n_val_groups, n_groups - n_train_groups - 1)

        train_groups = unique_groups[:n_train_groups]

        val_groups = unique_groups[
            n_train_groups:n_train_groups + n_val_groups
        ]

        test_groups = unique_groups[n_train_groups + n_val_groups:]

        return (
            np.flatnonzero(np.isin(groups, train_groups)),
            np.flatnonzero(np.isin(groups, val_groups)),
            np.flatnonzero(np.isin(groups, test_groups))
        )

    if mode == "random":

        rng = np.random.default_rng(seed)

        indices = np.arange(n_samples)
        rng.shuffle(indices)

        n_train = int(train_fraction * n_samples)
        n_val = int(val_fraction * n_samples)

        return (
            indices[:n_train],
            indices[n_train:n_train + n_val],
            indices[n_train + n_val:]
        )

    raise ValueError(
        f"Unknown mode: {mode!r}. "
        "Expected 'temporal', 'group' or 'random'."
    )


def split_dataset(
    raw,
    mode="temporal",
    train_fraction=0.70,
    val_fraction=0.15,
    gap=1,
    groups=None,
    seed=68
):
    """
    Split a RawDataset into train / validation / test.

    See compute_split_indices for the meaning of the arguments.
    """

    n_samples = raw.num_samples

    delta_t = raw.delta_t
    if raw.physics_features is not None:

        if len(raw.physics_features) != n_samples:
            raise ValueError(
                f"physics_features has "
                f"{len(raw.physics_features)} samples "
                f"but there are {n_samples} X samples: "
                "they must line up index by index."
            )

        if raw.physics_features.shape[1] != raw.num_nodes:
            raise ValueError(
                f"physics_features has "
                f"{raw.physics_features.shape[1]} nodes "
                f"but X has {raw.num_nodes} nodes."
            )

    # load_data lines delta_t up with X_input index by index. A mismatch
    # here would silently shift the time step of every sample, so fail
    # instead of quietly truncating.
    if len(delta_t) != n_samples:
        raise ValueError(
            f"delta_t has {len(delta_t)} entries but there are "
            f"{n_samples} samples: they must line up index by index."
        )

    train_indices, val_indices, test_indices = compute_split_indices(
        n_samples,
        mode=mode,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        gap=gap,
        groups=groups,
        seed=seed
    )

    def block(indices):

        physics_features = None

        if raw.physics_features is not None:
            physics_features = raw.physics_features[indices]

        return Split(
            X=raw.X_input[indices],
            Y=raw.Y_target[indices],
            dt=delta_t[indices],
            indices=indices,
            physics_features=physics_features,
        )

    return SplitDataset(
        train=block(train_indices),
        val=block(val_indices),
        test=block(test_indices)
    )


def format_split_statistics(splits):
    """
    Per-variable statistics of the three blocks, as a string.

    With a temporal split the blocks cover different phases of the
    simulation and are therefore NOT identically distributed. This is the
    only way to tell a genuine generalisation error from a plain
    distribution shift: if the test block spans a range of pressure the
    training block never contains, no model can be expected to fit it.
    """

    lines = [
        "",
        "=" * 44,
        "SPLIT STATISTICS",
        "=" * 44
    ]

    blocks = [
        ("train", splits.train),
        ("val  ", splits.val),
        ("test ", splits.test)
    ]

    for column, name in enumerate(VARIABLE_NAMES):

        lines.append(f"\n{name}:")

        for label, block in blocks:

            values = block.X[:, :, column]

            lines.append(
                f"  {label} | "
                f"n={len(block):4d} | "
                f"mean={values.mean():+.4e} | "
                f"std={values.std():.4e} | "
                f"min={values.min():+.4e} | "
                f"max={values.max():+.4e}"
            )

    return "\n".join(lines)

@dataclass
class SimulationSplits:
    train: list
    val: list
    test: list


def split_simulations(
    simulations,
    train_fraction=0.70,
    seed=68,
):
    """
    Split complete simulations into train, validation and test.

    Rules
    -----
    1. Train receives train_fraction of the simulations,
       rounded to the nearest integer.

    2. The remaining simulations are split equally between
       validation and test.

    3. If the number of remaining simulations is odd,
       test receives one more simulation than validation.

    The split is performed at simulation level, so all timesteps
    of a simulation always belong to the same split.
    """

    n_simulations = len(simulations)

    if n_simulations < 3:
        raise ValueError(
            "At least 3 simulations are required."
        )

    # ---------------------------------------------------------
    # Shuffle simulations reproducibly
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed)

    indices = np.arange(n_simulations)

    rng.shuffle(indices)

    # ---------------------------------------------------------
    # Number of training simulations
    # ---------------------------------------------------------

    n_train = int(
        np.round(n_simulations * train_fraction)
    )

    # Make sure validation and test can contain at least one
    # simulation each.
    n_train = min(
        n_train,
        n_simulations - 2
    )

    # ---------------------------------------------------------
    # Split remaining simulations between validation and test
    # ---------------------------------------------------------

    n_remaining = n_simulations - n_train

    # If odd, test automatically receives the extra simulation.
    n_val = n_remaining // 2

    n_test = n_remaining - n_val

    # ---------------------------------------------------------
    # Split indices
    # ---------------------------------------------------------

    train_indices = indices[:n_train]

    val_indices = indices[
        n_train:n_train + n_val
    ]

    test_indices = indices[
        n_train + n_val:
    ]

    # ---------------------------------------------------------
    # Build simulation lists
    # ---------------------------------------------------------

    train = [
        simulations[i]
        for i in train_indices
    ]

    val = [
        simulations[i]
        for i in val_indices
    ]

    test = [
        simulations[i]
        for i in test_indices
    ]
    print(
        f"Simulation split: "
        f"{n_train} train | "
        f"{n_val} val | "
        f"{n_test} test"
    )

    print(
        "Train simulations:",
        [sim.simulation_id for sim in train]
    )

    print(
        "Validation simulations:",
        [sim.simulation_id for sim in val]
    )

    print(
        "Test simulations:",
        [sim.simulation_id for sim in test]
    )

    return SimulationSplits(
        train=train,
        val=val,
        test=test,
    )


def subset_simulation(simulation, indices):
    """
    One simulation restricted to a subset of its samples.

    The mesh is shared, not copied: edge_index, edge_weight and pos are
    properties of the geometry and do not depend on which samples are
    kept.
    """

    indices = np.asarray(indices, dtype=np.int64)

    physics_features = None

    if simulation.physics_features is not None:
        physics_features = simulation.physics_features[indices]

    return replace(
        simulation,
        X_input=simulation.X_input[indices],
        Y_target=simulation.Y_target[indices],
        delta_t=simulation.delta_t[indices],
        physics_features=physics_features,
    )


def split_simulations_by_sample(
    simulations,
    mode="temporal",
    train_fraction=0.70,
    val_fraction=0.15,
    gap=1,
    seed=68,
):
    """
    Split every simulation along its own samples, then collect the blocks.

    This is the "temporal" and "random" behaviour, generalised to more
    than one simulation: each simulation is cut independently and
    contributes its own slice to each of the three blocks. With a single
    simulation it is exactly the original behaviour.

    Note that this is NOT the same question as split_simulations: here
    every geometry is seen during training, and what is measured is
    extrapolation in time. Use mode="simulation" to measure
    generalisation to an unseen geometry instead.

    Returns
    -------
    SimulationSplits
        Each block is a list of RawDataset, one per input simulation,
        sharing the mesh of the simulation it came from.
    """

    if mode not in ("temporal", "random"):
        raise ValueError(
            f"split_simulations_by_sample does not handle mode={mode!r}. "
            "Expected 'temporal' or 'random'; whole-simulation modes are "
            "split_simulations and split_simulations_by_group."
        )

    if not simulations:
        raise ValueError("At least one simulation is required.")

    train, val, test = [], [], []

    for simulation in simulations:

        n_samples = simulation.num_samples

        # Same guard as split_dataset: a mismatch here would silently
        # shift the time step of every sample.
        if len(simulation.delta_t) != n_samples:
            raise ValueError(
                f"Simulation {simulation.simulation_id}: delta_t has "
                f"{len(simulation.delta_t)} entries but there are "
                f"{n_samples} samples: they must line up index by index."
            )

        train_indices, val_indices, test_indices = compute_split_indices(
            n_samples,
            mode=mode,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            gap=gap,
            seed=seed,
        )

        train.append(subset_simulation(simulation, train_indices))
        val.append(subset_simulation(simulation, val_indices))
        test.append(subset_simulation(simulation, test_indices))

    return SimulationSplits(train=train, val=val, test=test)


def split_simulations_by_group(
    simulations,
    train_fraction=0.70,
    val_fraction=0.15,
    seed=68,
):
    """
    Whole simulations, allocated by fraction.

    Same guarantee as split_simulations - no simulation appears in more
    than one block - but the sizes follow train_fraction AND
    val_fraction, instead of giving validation and test half of the
    remainder each. It reuses compute_split_indices(mode="group"), which
    is the tested implementation of that allocation.

    Needs at least three simulations.
    """

    n_simulations = len(simulations)

    groups = np.arange(n_simulations)

    train_indices, val_indices, test_indices = compute_split_indices(
        n_simulations,
        mode="group",
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        groups=groups,
        seed=seed,
    )

    def block(indices):
        return [simulations[index] for index in indices]

    return SimulationSplits(
        train=block(train_indices),
        val=block(val_indices),
        test=block(test_indices),
    )


