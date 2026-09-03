"""
Experiment configuration.

An experiment is a YAML file, not a Python script. The four scripts that
used to live at the top level of this repository differed only in which
networks they built, which time encoding each network got and which
hyperparameters they used: all of that is now data.

Schema
------
name: str
seed: int
dataset:
  path: str              # one simulation
  paths: [str, ...]      # or several; exactly one of the two
  skip_initial: int      # snapshots dropped from the start
split:
  mode: temporal | group | random | simulation
  train_fraction: float
  val_fraction: float    # ignored by "simulation"
  gap: int               # "temporal" only
allow_partial_state: bool   # let the networks cover only part of u,v,p
training:
  num_epochs: int
  batch_size: int
networks:
  <network name>:
    predicts: velocity | pressure | state
    features: time | time_fourier
    architecture: gcn | gcn_virtual_node | bsms
    num_neurons: int or [int, ...]
    num_layers: int or [int, ...]
    dropout: float or [float, ...]
    learning_rate: float or [float, ...]
    weight_decay: float or [float, ...]
    unet_depth: int          # BSMS only
    hidden_layers: int       # BSMS only
    use_physics_features: bool      # BSMS only, default False
    use_predicted_velocity: bool    # BSMS only, default False

Any of the last five may be a list, in which case every combination is
trained and the one with the lowest validation loss is kept.
"""

import itertools
from pathlib import Path

import numpy as np
import yaml

from .data.features import FEATURE_SIZES
from .data.normalization import TARGET_COLUMNS
from .models import ARCHITECTURES

# Fields that may be swept
SWEEPABLE = [
    "num_neurons",
    "num_layers",
    "dropout",
    "learning_rate",
    "weight_decay"
]

DEFAULTS = {
    "seed": 68,
    "dataset": {
        # The first snapshot is the initial condition, not a state of the
        # flow. See load_data for why, and for how to tell whether 1 is
        # enough.
        "skip_initial": 1
    },
    "split": {
        "mode": "temporal",
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "gap": 1
    },
    "training": {
        "num_epochs": 200,
        "batch_size": 8
    }
}

NETWORK_DEFAULTS = {
    "features": "time",
    "architecture": "gcn",
    "num_neurons": 64,
    "num_layers": 3,
    "dropout": 0.0,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,

    # Extra input features, off unless a config asks for them. They must
    # be declared here and nowhere else: the number of input channels of
    # the model is derived from these flags, and so is the feature vector
    # the loaders build. Deriving either of them from the NAME of the
    # network instead would make renaming a network in the YAML silently
    # change the architecture.
    "use_physics_features": False,
    "use_predicted_velocity": False
}

BSMS_DEFAULTS = {
    "unet_depth": 2,
    "hidden_layers": 2,
}

def load_config(path):
    """Read, fill in defaults and validate an experiment file."""

    path = Path(path)

    with open(path) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a YAML mapping.")

    config.setdefault("name", path.stem)
    config.setdefault("seed", DEFAULTS["seed"])

    for section in ("dataset", "split", "training"):
        merged = dict(DEFAULTS[section])
        merged.update(config.get(section) or {})
        config[section] = merged

    _validate(config, path)

    return config


def _validate(config, path):

    dataset = config.get("dataset", {})

    has_path = (
        "path" in dataset
        and dataset["path"] is not None
    )

    has_paths = (
        "paths" in dataset
        and isinstance(dataset["paths"], list)
        and len(dataset["paths"]) > 0
    )

    if not has_path and not has_paths:
        raise ValueError(
            f"{path}: dataset must contain either "
            "'dataset.path' for one simulation or "
            "'dataset.paths' for multiple simulations."
        )

    if has_path and has_paths:
        raise ValueError(
            f"{path}: use either 'dataset.path' or "
            "'dataset.paths', not both."
        )

    skip_initial = config["dataset"]["skip_initial"]

    if not isinstance(skip_initial, int) or skip_initial < 0:
        raise ValueError(
            f"{path}: dataset.skip_initial must be a non-negative "
            f"integer, got {skip_initial!r}."
        )

    networks = config.get("networks")

    if not networks:
        raise ValueError(f"{path}: at least one network is required.")

    mode = config["split"]["mode"]

    if mode not in ("temporal", "group", "random", "simulation"):
        raise ValueError(
            f"{path}: unknown split mode {mode!r}. "
            "Expected 'temporal', 'group', 'random' or 'simulation'."
        )
    # Both modes keep whole simulations together, so a block can only be
    # non-empty if there are at least three of them. Checking it here
    # fails before the .mat files are read rather than after.
    if mode in ("simulation", "group"):

        if not has_paths:
            raise ValueError(
                f"{path}: split mode {mode!r} requires "
                "'dataset.paths' with multiple simulations."
            )

        if len(dataset["paths"]) < 3:
            raise ValueError(
                f"{path}: split mode {mode!r} keeps whole simulations "
                "together, so it needs at least 3 of them; "
                f"'dataset.paths' lists {len(dataset['paths'])}."
            )

    if mode == "simulation":

        train_fraction = config["split"].get("train_fraction")

        if train_fraction is None:
            raise ValueError(
                f"{path}: split mode 'simulation' requires "
                "'split.train_fraction'."
            )

        if not 0 < train_fraction < 1:
            raise ValueError(
                f"{path}: split.train_fraction must be between 0 and 1, "
                f"got {train_fraction!r}."
            )
    # Every state column must be predicted by exactly one network
    coverage = np.zeros(3, dtype=int)

    for name, network in networks.items():

        merged = dict(NETWORK_DEFAULTS)
        merged.update(network or {})
        if merged["architecture"] == "bsms":
            for key, value in BSMS_DEFAULTS.items():
                merged.setdefault(key, value)
        networks[name] = merged

        predicts = merged.get("predicts")

        if predicts not in TARGET_COLUMNS:
            raise ValueError(
                f"{path}: network {name!r} has predicts={predicts!r}. "
                f"Expected one of {sorted(TARGET_COLUMNS)}."
            )

        if merged["features"] not in FEATURE_SIZES:
            raise ValueError(
                f"{path}: network {name!r} has "
                f"features={merged['features']!r}. "
                f"Expected one of {sorted(FEATURE_SIZES)}."
            )

        if merged["architecture"] not in ARCHITECTURES:
            raise ValueError(
                f"{path}: network {name!r} has "
                f"architecture={merged['architecture']!r}. "
                f"Expected one of {sorted(ARCHITECTURES)}."
            )
        if merged["architecture"] == "bsms":

            if merged["predicts"] != "pressure":
                raise ValueError(
                    f"{path}: BSMS network {name!r} must currently use "
                    "predicts='pressure'."
                )

            if not isinstance(merged["unet_depth"], int):
                raise ValueError(
                    f"{path}: BSMS network {name!r} requires "
                    "unet_depth to be an integer."
                )

            if merged["unet_depth"] < 1:
                raise ValueError(
                    f"{path}: BSMS network {name!r} requires "
                    "unet_depth >= 1."
                )

            if not isinstance(merged["hidden_layers"], int):
                raise ValueError(
                    f"{path}: BSMS network {name!r} requires "
                    "hidden_layers to be an integer."
                )

            if merged["hidden_layers"] < 1:
                raise ValueError(
                    f"{path}: BSMS network {name!r} requires "
                    "hidden_layers >= 1."
                )

        # Extra input features. Both are wired into the loader that
        # feeds the BSMS network only, so allowing them elsewhere would
        # build a model with more input channels than the loader
        # supplies and fail at the first forward pass.
        for flag in ("use_physics_features", "use_predicted_velocity"):

            if not isinstance(merged[flag], bool):
                raise ValueError(
                    f"{path}: network {name!r} has "
                    f"{flag}={merged[flag]!r}; it must be true or false."
                )

            if merged[flag] and merged["architecture"] != "bsms":
                raise ValueError(
                    f"{path}: network {name!r} sets {flag}=true, but "
                    f"has architecture={merged['architecture']!r}. "
                    "Extra input features are currently only wired into "
                    "the BSMS pressure path."
                )

        coverage[TARGET_COLUMNS[predicts]] += 1

    allow_partial_state = config.get(
        "allow_partial_state",
        False
    )

    if np.any(coverage > 1):
        raise ValueError(
            f"{path}: some state variables are predicted by more than "
            f"one network; coverage per column is {coverage.tolist()}."
        )

    if not allow_partial_state and not np.all(coverage == 1):
        raise ValueError(
            f"{path}: the networks must predict u, v and p exactly once "
            f"between them; coverage per column is {coverage.tolist()}. "
            "Use one network with predicts='state', or one with "
            "'velocity' and one with 'pressure'."
        )



def expand_grid(network):
    """
    Every combination of the sweepable fields, as concrete configs.

    A field given as a scalar contributes a single value, so a config
    with no lists yields exactly one combination.
    """

    axes = []

    for field in SWEEPABLE:

        value = network[field]

        axes.append(value if isinstance(value, list) else [value])

    combinations = []

    for values in itertools.product(*axes):

        concrete = dict(network)
        concrete.update(dict(zip(SWEEPABLE, values)))

        combinations.append(concrete)

    return combinations
