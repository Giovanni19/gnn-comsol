"""
End-to-end smoke test: run a whole experiment on a tiny synthetic
dataset, in seconds.

This is the test that covers everything the other test files cannot,
because it is the only one that actually executes torch: the model
forward passes, batching, the training loop, checkpoint writing and
reloading, inference and the plots.

If this passes, the pipeline runs. It says nothing about whether the
model is any good.
"""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

NUM_SNAPSHOTS = 16
GRID = (3, 4)                       # 12 mesh nodes
NUM_NODES = GRID[0] * GRID[1]


def grid_edges(rows, cols):
    """4-neighbour grid, a stand-in for a mesh."""

    edges = []

    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append((r * cols + c, (r + 1) * cols + c))
            if c + 1 < cols:
                edges.append((r * cols + c, r * cols + c + 1))

    edges += [(j, i) for i, j in edges]

    return np.array(edges).T


@pytest.fixture
def dataset_path(tmp_path):

    rng = np.random.default_rng(0)

    edge_index = grid_edges(*GRID)

    # A slowly drifting field, with pressure on a much larger scale than
    # velocity: that ratio is what used to break the target scaling.
    base = rng.normal(size=(NUM_NODES, 3))
    base[:, 2] *= 500.0

    X = np.stack([
        base * (1 + 0.02 * k) + 0.01 * rng.normal(size=(NUM_NODES, 3))
        for k in range(NUM_SNAPSHOTS)
    ])

    # Non-uniform time steps, like an adaptive solver
    steps = 0.01 + 0.02 * rng.uniform(size=NUM_SNAPSHOTS - 1)
    t = np.concatenate([[0.0], np.cumsum(steps)])

    path = tmp_path / "tiny.mat"

    with h5py.File(path, "w") as f:
        # MATLAB order: (3, N, T)
        f["X"] = np.transpose(X, (2, 1, 0))
        f["edge_index"] = edge_index.astype(np.int64)
        f["edge_weight"] = np.ones(edge_index.shape[1])
        f["t"] = t.reshape(1, -1)
        f["h"] = np.array([[0.1]])

    return path


def write_config(tmp_path, dataset_path, networks, name="smoke"):

    config = {
        "name": name,
        "seed": 68,
        "dataset": {"path": str(dataset_path), "skip_initial": 1},
        "split": {
            "mode": "temporal",
            "train_fraction": 0.60,
            "val_fraction": 0.20,
            "gap": 1
        },
        "training": {"num_epochs": 2, "batch_size": 2},
        "networks": networks
    }

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


SMALL_GCN = {
    "architecture": "gcn",
    "num_neurons": 8,
    "num_layers": 2,
    "dropout": 0.0,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5
}


def test_two_network_experiment_runs(dataset_path, tmp_path):
    """The virtual-node setup: two networks, two feature encodings."""

    config = write_config(
        tmp_path,
        dataset_path,
        networks={
            "velocity": {
                "predicts": "velocity", "features": "time", **SMALL_GCN
            },
            "pressure": {
                "predicts": "pressure",
                "features": "time_fourier",
                **{**SMALL_GCN, "architecture": "gcn_virtual_node"}
            }
        }
    )

    run_dir = run(config, tmp_path / "outputs")

    for expected in [
        "config.json",
        "metrics.json",
        "velocity.pth",
        "pressure.pth",
        "velocity_loss.png",
        "pressure_loss.png"
    ]:
        assert (run_dir / expected).exists(), f"missing {expected}"

    metrics = json.loads((run_dir / "metrics.json").read_text())

    for name in ("u", "v", "p"):
        rmse = metrics["inference_metrics_physical"][name]["rmse"]
        assert np.isfinite(rmse), f"{name} RMSE is {rmse}"

    for name in ("velocity", "pressure"):
        loss = metrics["test_loss_normalized"][name]
        assert np.isfinite(loss) and loss >= 0


def test_monolithic_experiment_runs(dataset_path, tmp_path):
    """A single network predicting all three variables."""

    config = write_config(
        tmp_path,
        dataset_path,
        networks={
            "state": {
                "predicts": "state", "features": "time", **SMALL_GCN
            }
        },
        name="smoke_mono"
    )

    run_dir = run(config, tmp_path / "outputs")

    assert (run_dir / "state.pth").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert np.isfinite(metrics["inference_metrics_physical"]["p"]["rmse"])


def test_sweep_runs_and_writes_a_table(dataset_path, tmp_path):

    config = write_config(
        tmp_path,
        dataset_path,
        networks={
            "state": {
                "predicts": "state",
                "features": "time",
                **{**SMALL_GCN, "num_neurons": [4, 8]}
            }
        },
        name="smoke_sweep"
    )

    run_dir = run(config, tmp_path / "outputs")

    sweep = (run_dir / "sweep.csv").read_text().strip().split("\n")

    assert len(sweep) == 3, "header plus two configurations"


def test_checkpoint_round_trips(dataset_path, tmp_path):
    """
    The point of storing the normalizer: a checkpoint must be usable
    without knowing how it was trained.
    """

    sys.path.insert(0, str(REPO / "src"))

    from gnn_comsol.checkpoints import load_checkpoint
    from gnn_comsol.models import build_model

    config = write_config(
        tmp_path,
        dataset_path,
        networks={
            "state": {
                "predicts": "state", "features": "time", **SMALL_GCN
            }
        },
        name="smoke_ckpt"
    )

    run_dir = run(config, tmp_path / "outputs")

    model = build_model(
        architecture="gcn",
        num_in=4,
        num_out=3,
        num_neurons=8,
        num_layers=2,
        dropout=0.0
    )

    model, normalizer, metadata = load_checkpoint(
        run_dir / "state.pth", model=model
    )

    assert metadata["predicts"] == "state"
    assert metadata["features"] == "time"

    # The scaling survived the round trip
    values = np.array([[1.0, 2.0, 3000.0]])

    assert np.allclose(
        normalizer.inverse_transform(normalizer.transform(values)),
        values
    )


def test_legacy_checkpoint_is_refused(tmp_path):
    """A raw state_dict has no recoverable target scaling."""

    import torch

    sys.path.insert(0, str(REPO / "src"))

    from gnn_comsol.checkpoints import load_checkpoint

    path = tmp_path / "legacy.pth"

    torch.save({"weight": torch.zeros(2)}, path)

    with pytest.raises(ValueError, match="no normalizer"):
        load_checkpoint(path)
