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


def write_config(
    tmp_path,
    simulations,
    networks,
    name="smoke",
    allow_partial_state=False
):

    config = {
        "name": name,
        "seed": 68,
        "dataset": {
            "paths": [str(info["path"]) for info in simulations],
            "skip_initial": 1
        },
        "split": {"mode": "simulation", "train_fraction": 0.70},
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
