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
  path: str
  skip_initial: int      # snapshots dropped from the start
split:
  mode: temporal | group | random
  train_fraction: float
  val_fraction: float
  gap: int
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
    "weight_decay": 1.0e-5
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

    if "dataset" not in config or "path" not in (config["dataset"] or {}):
        raise ValueError(
            f"{path}: a 'dataset.path' pointing at the .mat file "
            "is required."
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

    if mode not in ("temporal", "group", "random"):
        raise ValueError(
            f"{path}: unknown split mode {mode!r}. "
            "Expected 'temporal', 'group' or 'random'."
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
        coverage[TARGET_COLUMNS[predicts]] += 1

    if not np.all(coverage == 1):
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
