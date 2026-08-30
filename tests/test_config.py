"""Experiment files: defaults, validation and hyperparameter sweeps."""

import pytest
import yaml

from gnn_comsol.config import expand_grid, load_config


BASE = {
    "name": "unit_test",
    "dataset": {"path": "data/whatever.mat"},
    "networks": {
        "velocity": {
            "predicts": "velocity",
            "features": "time",
            "architecture": "gcn"
        },
        "pressure": {
            "predicts": "pressure",
            "features": "time_fourier",
            "architecture": "gcn_virtual_node"
        }
    }
}


def write(tmp_path, config, name="experiment.yaml"):

    path = tmp_path / name

    with open(path, "w") as handle:
        yaml.safe_dump(config, handle)

    return path


def test_defaults_are_filled_in(tmp_path):

    config = load_config(write(tmp_path, BASE))

    assert config["seed"] == 68
    assert config["split"]["mode"] == "temporal"
    assert config["split"]["gap"] == 1
    assert config["training"]["num_epochs"] == 200

    # Network defaults too
    assert config["networks"]["velocity"]["num_neurons"] == 64


def test_explicit_values_win_over_defaults(tmp_path):

    custom = {**BASE, "split": {"mode": "random", "gap": 0}}

    config = load_config(write(tmp_path, custom))

    assert config["split"]["mode"] == "random"
    assert config["split"]["gap"] == 0
    # untouched keys still get their default
    assert config["split"]["train_fraction"] == 0.70


def test_shipped_configs_are_valid():
    """The configs in configs/ must load."""

    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs"

    files = sorted(configs.glob("*.yaml"))

    assert files, "no config files found"

    for path in files:
        load_config(path)


def test_partial_coverage_is_rejected(tmp_path):
    """A config predicting only velocity would silently leave p unset."""

    partial = {
        **BASE,
        "networks": {"velocity": {"predicts": "velocity"}}
    }

    with pytest.raises(ValueError, match="exactly once"):
        load_config(write(tmp_path, partial))


def test_overlapping_coverage_is_rejected(tmp_path):

    overlapping = {
        **BASE,
        "networks": {
            "a": {"predicts": "state"},
            "b": {"predicts": "pressure"}
        }
    }

    with pytest.raises(ValueError, match="exactly once"):
        load_config(write(tmp_path, overlapping))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("predicts", "temperature", "predicts"),
        ("features", "wavelet", "features"),
        ("architecture", "transformer", "architecture"),
    ],
)
def test_unknown_network_options_are_rejected(
    tmp_path, field, value, message
):

    broken = {
        **BASE,
        "networks": {
            "velocity": {"predicts": "velocity"},
            "pressure": {"predicts": "pressure", field: value}
        }
    }

    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, broken))


def test_missing_dataset_path_is_rejected(tmp_path):

    broken = {k: v for k, v in BASE.items() if k != "dataset"}

    with pytest.raises(ValueError, match="dataset.path"):
        load_config(write(tmp_path, broken))


def test_unknown_split_mode_is_rejected(tmp_path):

    broken = {**BASE, "split": {"mode": "kfold"}}

    with pytest.raises(ValueError, match="split mode"):
        load_config(write(tmp_path, broken))


def test_scalars_give_exactly_one_combination():

    network = {
        "num_neurons": 64,
        "num_layers": 3,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5
    }

    assert len(expand_grid(network)) == 1


def test_lists_expand_to_the_full_product():

    network = {
        "num_neurons": [64, 128],
        "num_layers": [4, 8, 12],
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5
    }

    combinations = expand_grid(network)

    assert len(combinations) == 6

    pairs = {
        (c["num_neurons"], c["num_layers"]) for c in combinations
    }

    assert len(pairs) == 6


def test_expand_grid_preserves_other_fields():

    network = {
        "predicts": "pressure",
        "architecture": "gcn_virtual_node",
        "num_neurons": [64, 128],
        "num_layers": 3,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5
    }

    for concrete in expand_grid(network):
        assert concrete["predicts"] == "pressure"
        assert concrete["architecture"] == "gcn_virtual_node"
