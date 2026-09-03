"""
End-to-end smoke test: run a whole experiment on tiny synthetic
simulations, in seconds.

This is the only test that actually executes torch, so it is the one
that covers the model forward passes, batching, the training loop,
checkpoint writing and reloading, and the plots.

The simulations deliberately have DIFFERENT mesh sizes: the runner is
multi-geometry now, and a test on identical meshes would not exercise
that.

If this passes, the pipeline runs. It says nothing about whether the
model is any good: two epochs on a handful of samples cannot.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from gnn_comsol.data.normalization import (
    NUM_PHYSICS_FEATURES,
    PHYSICS_FEATURE_NAMES
)

REPO = Path(__file__).resolve().parents[1]

# Meshes of four different sizes, all big enough to survive the BSMS
# pooling depth used below.
MESHES = [(5, 6), (6, 6), (5, 7), (6, 7)]


@pytest.fixture
def simulations(make_dataset):
    """Four synthetic simulations, each on its own mesh."""

    return [
        make_dataset(rows=rows, cols=cols, seed=index, num_snapshots=14)
        for index, (rows, cols) in enumerate(MESHES)
    ]


@pytest.fixture
def simulations_without_physics(make_dataset):
    """
    The same four, in the layout that predates physics features.

    This is what the six multi-geometry .mat files look like, and what
    the MATLAB generator in this repository still produces.
    """

    return [
        make_dataset(
            rows=rows,
            cols=cols,
            seed=index,
            num_snapshots=14,
            with_physics=False
        )
        for index, (rows, cols) in enumerate(MESHES)
    ]


VELOCITY_NET = {
    "predicts": "velocity",
    "features": "time",
    "architecture": "gcn",
    "num_neurons": 8,
    "num_layers": 2,
    "dropout": 0.0,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5
}

PRESSURE_BSMS_NET = {
    "predicts": "pressure",
    "features": "time",
    "architecture": "bsms",
    "num_neurons": 8,
    "unet_depth": 2,
    "hidden_layers": 1,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-3,
    "num_layers": 1,
    "dropout": 0.0
}

PRESSURE_BSMS_PHYSICS_NET = {
    **PRESSURE_BSMS_NET,
    "use_physics_features": True
}

# "time" is 4 features: u, v, p, dt
BASE_FEATURES = 4


def write_config(
    tmp_path,
    simulations,
    networks,
    name="smoke",
    allow_partial_state=False,
    dataset=None,
    split=None
):

    config = {
        "name": name,
        "seed": 68,
        "dataset": dataset or {
            "paths": [str(info["path"]) for info in simulations],
            "skip_initial": 1
        },
        "split": split or {"mode": "simulation", "train_fraction": 0.70},
        "training": {"num_epochs": 2, "batch_size": 2},
        "networks": networks
    }

    if allow_partial_state:
        config["allow_partial_state"] = True

    path = tmp_path / f"{name}.yaml"

    with open(path, "w") as handle:
        yaml.safe_dump(config, handle)

    return path


def run(config_path, output_root):
    """Invoke the real entry point."""

    spec = importlib.util.spec_from_file_location(
        "run_experiment", REPO / "scripts" / "run_experiment.py"
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = sys.argv

    sys.argv = [
        "run_experiment.py",
        str(config_path),
        "--output-root", str(output_root),
        "--quiet"
    ]

    try:
        module.main()
    finally:
        sys.argv = argv

    runs = sorted(Path(output_root).iterdir())

    assert len(runs) == 1, f"expected one run directory, got {runs}"

    return runs[0]


def test_velocity_only_experiment_runs(simulations, tmp_path):
    """The simplest working configuration."""

    config = write_config(
        tmp_path,
        simulations,
        networks={"velocity": VELOCITY_NET},
        allow_partial_state=True
    )

    run_dir = run(config, tmp_path / "outputs")

    for expected in [
        "config.json",
        "metrics.json",
        "velocity.pth",
        "velocity_loss.png"
    ]:
        assert (run_dir / expected).exists(), f"missing {expected}"

    metrics = json.loads((run_dir / "metrics.json").read_text())

    loss = metrics["test_loss_normalized"]["velocity"]

    assert np.isfinite(loss) and loss >= 0


def test_split_reported_in_metrics_covers_every_simulation(
    simulations, tmp_path
):
    """No simulation may sit in two blocks, and none may be dropped."""

    config = write_config(
        tmp_path,
        simulations,
        networks={"velocity": VELOCITY_NET},
        allow_partial_state=True,
        name="smoke_split"
    )

    run_dir = run(config, tmp_path / "outputs")

    metrics = json.loads((run_dir / "metrics.json").read_text())

    train = set(metrics["train_simulations"])
    val = set(metrics["val_simulations"])
    test = set(metrics["test_simulations"])

    assert not train & val
    assert not val & test
    assert not train & test

    assert train | val | test == set(range(len(MESHES)))
    assert val and test


TEMPORAL_SPLIT = {
    "mode": "temporal",
    "train_fraction": 0.60,
    "val_fraction": 0.20,
    "gap": 1
}


def test_single_simulation_config_runs(simulations, tmp_path):
    """
    A config with dataset.path (singular), the way the first three
    experiments are written.

    The multi-geometry work made the runner read dataset.paths
    unconditionally and always split whole simulations, so every
    single-simulation config died with KeyError: 'paths' - including the
    command in the README. Nothing executed that path from the entry
    point, which is exactly why it went unnoticed.
    """

    config = write_config(
        tmp_path,
        simulations,
        networks={"velocity": VELOCITY_NET},
        allow_partial_state=True,
        name="smoke_single",
        dataset={
            "path": str(simulations[0]["path"]),
            "skip_initial": 1
        },
        split=TEMPORAL_SPLIT
    )

    run_dir = run(config, tmp_path / "outputs")

    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert metrics["split_mode"] == "temporal"

    # One simulation, cut in three along its own samples
    assert metrics["train_simulations"] == [0]
    assert metrics["val_simulations"] == [0]
    assert metrics["test_simulations"] == [0]

    for block in ("train", "val", "test"):
        assert metrics["samples"][block] > 0, f"{block} block is empty"

    loss = metrics["test_loss_normalized"]["velocity"]

    assert np.isfinite(loss) and loss >= 0


def test_split_mode_is_respected(simulations, tmp_path):
    """
    split.mode used to be validated and then ignored.

    The two modes ask different questions and must therefore produce
    different splits: "simulation" holds whole geometries out, so the
    blocks are disjoint; "temporal" cuts inside every simulation, so
    every geometry appears in all three blocks.
    """

    all_ids = set(range(len(MESHES)))

    temporal = json.loads(
        (
            run(
                write_config(
                    tmp_path,
                    simulations,
                    networks={"velocity": VELOCITY_NET},
                    allow_partial_state=True,
                    name="smoke_temporal",
                    split=TEMPORAL_SPLIT
                ),
                tmp_path / "outputs_temporal"
            ) / "metrics.json"
        ).read_text()
    )

    by_simulation = json.loads(
        (
            run(
                write_config(
                    tmp_path,
                    simulations,
                    networks={"velocity": VELOCITY_NET},
                    allow_partial_state=True,
                    name="smoke_by_simulation",
                    split={"mode": "simulation", "train_fraction": 0.70}
                ),
                tmp_path / "outputs_by_simulation"
            ) / "metrics.json"
        ).read_text()
    )

    assert temporal["split_mode"] == "temporal"

    for block in ("train_simulations", "val_simulations",
                  "test_simulations"):
        assert set(temporal[block]) == all_ids, (
            f"a temporal split must cut inside every simulation, "
            f"but {block} is {temporal[block]}"
        )

    assert by_simulation["split_mode"] == "simulation"

    train = set(by_simulation["train_simulations"])
    val = set(by_simulation["val_simulations"])
    test = set(by_simulation["test_simulations"])

    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == all_ids


def test_velocity_and_bsms_pressure_run(simulations, tmp_path):
    """The full configuration, across meshes of different sizes."""

    config = write_config(
        tmp_path,
        simulations,
        networks={
            "velocity": VELOCITY_NET,
            "pressure": PRESSURE_BSMS_NET
        },
        name="smoke_bsms"
    )

    run_dir = run(config, tmp_path / "outputs")

    for expected in [
        "velocity.pth",
        "pressure.pth",
        "velocity_loss.png",
        "pressure_loss.png"
    ]:
        assert (run_dir / expected).exists(), f"missing {expected}"

    metrics = json.loads((run_dir / "metrics.json").read_text())

    for name in ("velocity", "pressure"):
        loss = metrics["test_loss_normalized"][name]
        assert np.isfinite(loss), f"{name} test loss is {loss}"


# ---------------------------------------------------------------------
# Physics-derived input features
#
# These were briefly mandatory: every run rebuilt their normalizer and
# refused any dataset without them, which broke the six multi-geometry
# .mat files and every test in this file. They are opt-in now, and the
# three cases below are with, without, and asked-for-but-missing.
# ---------------------------------------------------------------------

def test_pressure_uses_physics_features_when_asked(
    simulations, tmp_path
):
    """The flag widens the model and the checkpoint records it."""

    config = write_config(
        tmp_path,
        simulations,
        networks={
            "velocity": VELOCITY_NET,
            "pressure": PRESSURE_BSMS_PHYSICS_NET
        },
        name="smoke_physics"
    )

    run_dir = run(config, tmp_path / "outputs")

    from gnn_comsol.checkpoints import (
        load_physics_normalizer,
        read_checkpoint
    )

    bundle = read_checkpoint(run_dir / "pressure.pth")

    metadata = bundle.metadata

    assert metadata["use_physics_features"] is True
    assert metadata["physics_feature_names"] == PHYSICS_FEATURE_NAMES
    assert metadata["num_in"] == BASE_FEATURES + NUM_PHYSICS_FEATURES

    # The scaling of the physics features must travel with the weights
    # for the same reason the state scaling does: whoever rebuilds these
    # features at inference cannot be left to guess it.
    physics_normalizer = load_physics_normalizer(
        run_dir / "pressure.pth",
        required=True
    )

    assert physics_normalizer.mean.shape == (NUM_PHYSICS_FEATURES,)
    assert np.all(physics_normalizer.std > 0)

    assert np.allclose(
        bundle.physics_normalizer.mean, physics_normalizer.mean
    )

    # The velocity network was not fed them, so it must not claim to
    # carry their scaling.
    velocity_bundle = read_checkpoint(run_dir / "velocity.pth")

    assert velocity_bundle.metadata["use_physics_features"] is False
    assert velocity_bundle.physics_normalizer is None
    assert load_physics_normalizer(run_dir / "velocity.pth") is None

    # The time-step scaling is part of how EVERY network was fed, so it
    # has to travel with the weights. It used to live only in the stdout
    # of a run, and the evaluation script carried a stale copy of it.
    for spec in (bundle, velocity_bundle):
        assert spec.dt_mean is not None
        assert spec.dt_std is not None and spec.dt_std > 0

    assert bundle.dt_mean == velocity_bundle.dt_mean
    assert bundle.dt_std == velocity_bundle.dt_std

    metrics = json.loads((run_dir / "metrics.json").read_text())

    for name in ("velocity", "pressure"):
        assert np.isfinite(metrics["test_loss_normalized"][name])


def test_physics_features_are_off_unless_requested(
    simulations, tmp_path
):
    """
    A dataset that HAS physics features must not get them silently.

    The width of the model is part of the experiment, so it has to
    follow the configuration and not what happens to be in the .mat.
    """

    config = write_config(
        tmp_path,
        simulations,
        networks={
            "velocity": VELOCITY_NET,
            "pressure": PRESSURE_BSMS_NET
        },
        name="smoke_no_physics_flag"
    )

    run_dir = run(config, tmp_path / "outputs")

    from gnn_comsol.checkpoints import (
        load_checkpoint,
        load_physics_normalizer
    )

    _, _, metadata = load_checkpoint(run_dir / "pressure.pth")

    assert metadata["use_physics_features"] is False
    assert metadata["physics_feature_names"] == []
    assert metadata["num_in"] == BASE_FEATURES

    assert load_physics_normalizer(run_dir / "pressure.pth") is None


def test_dataset_without_physics_features_still_runs(
    simulations_without_physics, tmp_path
):
    """
    The regression that mattered: the six multi-geometry datasets have
    no physics features, and an experiment that does not ask for them
    must not care.
    """

    config = write_config(
        tmp_path,
        simulations_without_physics,
        networks={
            "velocity": VELOCITY_NET,
            "pressure": PRESSURE_BSMS_NET
        },
        name="smoke_legacy_dataset"
    )

    run_dir = run(config, tmp_path / "outputs")

    metrics = json.loads((run_dir / "metrics.json").read_text())

    for name in ("velocity", "pressure"):
        assert np.isfinite(metrics["test_loss_normalized"][name])


def test_missing_physics_features_are_refused_clearly(
    simulations_without_physics, tmp_path
):
    """
    Asking for features the dataset does not carry must say so, and say
    which simulation is missing them.
    """

    config = write_config(
        tmp_path,
        simulations_without_physics,
        networks={
            "velocity": VELOCITY_NET,
            "pressure": PRESSURE_BSMS_PHYSICS_NET
        },
        name="smoke_physics_missing"
    )

    with pytest.raises(ValueError, match="physics_features"):
        run(config, tmp_path / "outputs")


def test_sweep_runs_and_writes_a_table(simulations, tmp_path):

    config = write_config(
        tmp_path,
        simulations,
        networks={
            "velocity": {**VELOCITY_NET, "num_neurons": [4, 8]}
        },
        allow_partial_state=True,
        name="smoke_sweep"
    )

    run_dir = run(config, tmp_path / "outputs")

    rows = (run_dir / "sweep.csv").read_text().strip().split("\n")

    assert len(rows) == 3, "header plus two configurations"


def test_checkpoint_round_trips(simulations, tmp_path):
    """
    The point of storing the normalizer and the architecture: a
    checkpoint must be usable without knowing how it was trained.
    """

    sys.path.insert(0, str(REPO / "src"))

    from gnn_comsol.checkpoints import load_checkpoint
    from gnn_comsol.models import build_model

    config = write_config(
        tmp_path,
        simulations,
        networks={"velocity": VELOCITY_NET},
        allow_partial_state=True,
        name="smoke_ckpt"
    )

    run_dir = run(config, tmp_path / "outputs")

    _, _, metadata = load_checkpoint(run_dir / "velocity.pth")

    assert metadata["predicts"] == "velocity"
    assert metadata["architecture"] == "gcn"

    # Everything needed to rebuild the model is in the metadata
    model = build_model(
        architecture=metadata["architecture"],
        num_in=metadata["num_in"],
        num_out=metadata["num_out"],
        num_neurons=metadata["num_neurons"],
        num_layers=metadata["num_layers"],
        dropout=metadata["dropout"]
    )

    model, normalizer, _ = load_checkpoint(
        run_dir / "velocity.pth", model=model
    )

    values = np.array([[1.0, 2.0, 3000.0]])

    assert np.allclose(
        normalizer.inverse_transform(normalizer.transform(values)),
        values
    )


def test_bsms_checkpoint_records_its_own_hyperparameters(
    simulations, tmp_path
):
    """A BSMS model cannot be rebuilt without unet_depth."""

    sys.path.insert(0, str(REPO / "src"))

    from gnn_comsol.checkpoints import load_checkpoint

    config = write_config(
        tmp_path,
        simulations,
        networks={
            "velocity": VELOCITY_NET,
            "pressure": PRESSURE_BSMS_NET
        },
        name="smoke_bsms_meta"
    )

    run_dir = run(config, tmp_path / "outputs")

    _, _, metadata = load_checkpoint(run_dir / "pressure.pth")

    assert metadata["architecture"] == "bsms"
    assert metadata["unet_depth"] == PRESSURE_BSMS_NET["unet_depth"]
    assert metadata["hidden_layers"] == PRESSURE_BSMS_NET["hidden_layers"]


def test_legacy_checkpoint_is_refused(tmp_path):
    """A raw state_dict has no recoverable target scaling."""

    import torch

    sys.path.insert(0, str(REPO / "src"))

    from gnn_comsol.checkpoints import load_checkpoint

    path = tmp_path / "legacy.pth"

    torch.save({"weight": torch.zeros(2)}, path)

    with pytest.raises(ValueError, match="no normalizer"):
        load_checkpoint(path)
