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

    with pytest.raises(ValueError, match="more than one network"):
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


# ---------------------------------------------------------------------
# One simulation or many
# ---------------------------------------------------------------------

MULTI = {
    "name": "multi",
    "dataset": {"paths": ["a.mat", "b.mat", "c.mat"]},
    "split": {"mode": "simulation", "train_fraction": 0.70},
    "networks": BASE["networks"]
}


def test_paths_is_accepted_for_multiple_simulations(tmp_path):

    config = load_config(write(tmp_path, MULTI))

    assert len(config["dataset"]["paths"]) == 3


def test_path_and_paths_together_are_rejected(tmp_path):

    both = {
        **MULTI,
        "dataset": {"path": "a.mat", "paths": ["a.mat", "b.mat", "c.mat"]}
    }

    with pytest.raises(ValueError, match="not both"):
        load_config(write(tmp_path, both))


def test_simulation_split_requires_paths(tmp_path):

    single = {
        **BASE,
        "split": {"mode": "simulation", "train_fraction": 0.70}
    }

    with pytest.raises(ValueError, match="requires"):
        load_config(write(tmp_path, single))


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5, -0.2])
def test_simulation_split_rejects_impossible_fractions(
    tmp_path, fraction
):

    bad = {
        **MULTI,
        "split": {"mode": "simulation", "train_fraction": fraction}
    }

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_config(write(tmp_path, bad))


def test_allow_partial_state_permits_a_velocity_only_experiment(tmp_path):
    """
    A velocity-only run is legitimate while the pressure head is being
    worked on, but only if the config says so explicitly.
    """

    velocity_only = {
        **MULTI,
        "allow_partial_state": True,
        "networks": {"velocity": {"predicts": "velocity"}}
    }

    config = load_config(write(tmp_path, velocity_only))

    assert set(config["networks"]) == {"velocity"}

    # Without the flag the same config must still be refused
    del velocity_only["allow_partial_state"]

    with pytest.raises(ValueError, match="exactly once"):
        load_config(write(tmp_path, velocity_only, name="no_flag.yaml"))


# ---------------------------------------------------------------------
# Extra input features
# ---------------------------------------------------------------------

BSMS_PRESSURE = {
    "predicts": "pressure",
    "features": "time",
    "architecture": "bsms",
    "unet_depth": 2,
    "hidden_layers": 1
}


def test_extra_feature_flags_default_to_off(tmp_path):
    """
    The width of a model must follow the configuration, so both flags
    have to exist with a known value rather than being read with .get()
    in whichever module happens to need them.
    """

    config = load_config(write(tmp_path, BASE))

    for network in config["networks"].values():
        assert network["use_physics_features"] is False
        assert network["use_predicted_velocity"] is False


@pytest.mark.parametrize(
    "flag",
    ["use_physics_features", "use_predicted_velocity"]
)
def test_extra_features_are_rejected_outside_bsms(tmp_path, flag):
    """
    Only the BSMS loader assembles them. On any other architecture the
    model would be built wider than the loader that feeds it, and the
    mismatch would only surface inside the first forward pass.
    """

    broken = {
        **BASE,
        "networks": {
            "velocity": {"predicts": "velocity"},
            "pressure": {
                "predicts": "pressure",
                "architecture": "gcn_virtual_node",
                flag: True
            }
        }
    }

    with pytest.raises(ValueError, match="only wired into"):
        load_config(write(tmp_path, broken))


@pytest.mark.parametrize(
    "flag",
    ["use_physics_features", "use_predicted_velocity"]
)
def test_extra_feature_flags_must_be_boolean(tmp_path, flag):

    broken = {
        **BASE,
        "networks": {
            "velocity": {"predicts": "velocity"},
            "pressure": {**BSMS_PRESSURE, flag: "yes"}
        }
    }

    with pytest.raises(ValueError, match="true or false"):
        load_config(write(tmp_path, broken))


def test_extra_features_are_accepted_on_bsms(tmp_path):

    accepted = {
        **BASE,
        "networks": {
            "velocity": {"predicts": "velocity"},
            "pressure": {
                **BSMS_PRESSURE,
                "use_physics_features": True,
                "use_predicted_velocity": True
            }
        }
    }

    config = load_config(write(tmp_path, accepted))

    pressure = config["networks"]["pressure"]

    assert pressure["use_physics_features"] is True
    assert pressure["use_predicted_velocity"] is True


@pytest.mark.parametrize("skip", [-1, 1.5, "two"])
def test_invalid_skip_initial_is_rejected(tmp_path, skip):

    bad = {**BASE, "dataset": {"path": "a.mat", "skip_initial": skip}}

    with pytest.raises(ValueError, match="skip_initial"):
        load_config(write(tmp_path, bad))
