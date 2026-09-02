#!/usr/bin/env python3
"""
Run one experiment, described by a YAML file in configs/.

    python scripts/run_experiment.py configs/multi_geometry_bsms_test.yaml

Everything a run produces goes into outputs/<name>_<timestamp>/:
the resolved configuration, one checkpoint per network, the loss curves,
the sweep table if there was one, and the metrics as JSON.

main() at the bottom is the outline of the experiment; every step above
it is one function.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TensorDataLoader

# Make the package importable without installing it first
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_comsol import data as gdata                      # noqa: E402
from gnn_comsol import plots                              # noqa: E402
from gnn_comsol.checkpoints import save_checkpoint        # noqa: E402
from gnn_comsol.config import expand_grid, load_config    # noqa: E402
from gnn_comsol.evaluate import (                         # noqa: E402
    evaluate_bsms_multi_simulation,
    evaluate_dataset
)
from gnn_comsol.graph.bsms import BistrideMultiLayerGraph  # noqa: E402
from gnn_comsol.models import build_model                 # noqa: E402
from gnn_comsol.train import (                            # noqa: E402
    train_bsms_multi_simulation,
    train_network
)


def set_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Nice to have, not required for correctness. Several PyTorch
    # Geometric ops use scatter kernels with no deterministic CUDA
    # implementation, and enabling this makes them raise. Losing exact
    # reproducibility is a much smaller problem than not running at all.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as error:
        print(
            f"Note: deterministic algorithms unavailable ({error}). "
            "Runs stay seeded but are not bit-for-bit reproducible."
        )


def parse_args():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("config", help="path to a YAML experiment file")

    parser.add_argument(
        "--output-root",
        default="outputs",
        help="where to create the run directory (default: outputs)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print a line per epoch"
    )

    return parser.parse_args()


def create_run_dir(output_root, config):
    """One directory per run, holding the resolved configuration."""

    run_dir = Path(output_root) / (
        f"{config['name']}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    return run_dir


# =====================================================================
# Data
# =====================================================================

def dataset_paths(config):
    """
    The simulations to load, from either dataset.path or dataset.paths.

    load_config guarantees exactly one of the two is present, so a single
    simulation is just a list of length one and the rest of the pipeline
    does not need to know the difference.
    """

    dataset = config["dataset"]

    if dataset.get("paths"):
        return list(dataset["paths"])

    return [dataset["path"]]


def load_and_report_simulations(config):

    simulations = gdata.load_simulations(
        dataset_paths(config),
        skip_initial=config["dataset"]["skip_initial"]
    )

    print(
        f"\nLoaded {len(simulations)} simulations "
        f"({config['dataset']['skip_initial']} initial snapshot(s) "
        f"dropped per simulation)"
    )

    for simulation in simulations:

        print(
            f"  Simulation {simulation.simulation_id}: "
            f"{simulation.num_samples} samples | "
            f"{simulation.num_nodes} nodes | "
            f"{simulation.num_edges} edges | "
            f"{simulation.file_path}"
        )

    # If the solver used a constant step this is ~0 and the time feature
    # carries no information; if it is large the step varies a lot.
    for simulation in simulations:

        dt = simulation.delta_t

        print(
            f"  Simulation {simulation.simulation_id} dt: "
            f"mean={dt.mean():.4e} | "
            f"min={dt.min():.4e} | "
            f"max={dt.max():.4e}"
        )

    return simulations


def split_and_report(simulations, config):
    """
    Apply the split the configuration asked for.

    split.mode used to be validated and then ignored: whatever it said,
    whole simulations were split. The four modes now do different things,
    and the difference is the question being asked.

        simulation / group
            Whole simulations, so a block never shares a geometry with
            another. Measures generalisation to an UNSEEN GEOMETRY.
            "simulation" gives train_fraction to training and halves the
            rest; "group" follows train_fraction and val_fraction.

        temporal / random
            Every simulation is cut along its own samples and contributes
            a slice to each block. Measures EXTRAPOLATION IN TIME, with
            every geometry seen during training. "temporal" uses
            contiguous blocks plus a gap, "random" is the legacy shuffled
            split kept only for comparison.
    """

    split = config["split"]

    mode = split["mode"]

    if mode == "simulation":

        splits = gdata.split_simulations(
            simulations,
            train_fraction=split["train_fraction"],
            seed=config["seed"]
        )

    elif mode == "group":

        splits = gdata.split_simulations_by_group(
            simulations,
            train_fraction=split["train_fraction"],
            val_fraction=split["val_fraction"],
            seed=config["seed"]
        )

    else:

        splits = gdata.split_simulations_by_sample(
            simulations,
            mode=mode,
            train_fraction=split["train_fraction"],
            val_fraction=split["val_fraction"],
            gap=split["gap"],
            seed=config["seed"]
        )

    print(f"\nSplit ({mode}):")

    for name in ("train", "val", "test"):

        block = getattr(splits, name)

        samples = sum(
            simulation.num_samples for simulation in block
        )

        print(
            f"\n{name.upper()} "
            f"({len(block)} simulation(s), {samples} samples)"
        )

        for simulation in block:

            print(
                f"  Simulation {simulation.simulation_id}: "
                f"{simulation.num_nodes} nodes | "
                f"{simulation.num_samples} samples | "
                f"{simulation.file_path}"
            )

    return splits


# =====================================================================
# Scaling, computed on the training block only
# =====================================================================

def build_normalizer(train_simulations):

    x_mean, x_std, dt_mean, dt_std = (
        gdata.compute_multi_simulation_normalization_parameters(
            train_simulations
        )
    )

    normalizer = gdata.StateNormalizer(x_mean, x_std)

    print(f"\n{normalizer}")

    print(
        f"delta_t normalization: "
        f"mean={dt_mean:.6e}, std={dt_std:.6e}"
    )

    return normalizer, dt_mean, dt_std


def normalize_all_splits(splits, normalizer, dt_mean, dt_std):
    """split name -> list of normalized simulations, mesh preserved."""

    blocks = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }

    return {
        split_name: [
            gdata.normalize_simulation(
                simulation,
                normalizer,
                dt_mean,
                dt_std,
            )
            for simulation in split_simulations
        ]
        for split_name, split_simulations in blocks.items()
    }


# =====================================================================
# BSMS hierarchies
# =====================================================================

def uses_bsms(config):

    return any(
        network["architecture"] == "bsms"
        for network in config["networks"].values()
    )


def build_bsms_hierarchies(simulations, config):
    """
    simulation_id -> {edge_indices, pool_indices, pos}

    Empty when no network in the experiment is a BSMS one. Every mesh
    gets its own hierarchy: the model parameters are shared across
    geometries, the geometry itself is not.
    """

    bsms_hierarchies = {}

    if not uses_bsms(config):
        return bsms_hierarchies

    bsms_networks = [
        network
        for network in config["networks"].values()
        if network["architecture"] == "bsms"
    ]

    unet_depths = {
        network["unet_depth"]
        for network in bsms_networks
    }

    if len(unet_depths) != 1:
        raise ValueError(
            "All BSMS networks in one experiment must currently "
            "use the same unet_depth."
        )

    unet_depth = next(iter(unet_depths))

    print("\nBuilding BSMS hierarchies:")

    for simulation in simulations:

        multi_layer_graph = BistrideMultiLayerGraph(
            simulation.edge_index,
            unet_depth,
            simulation.num_nodes,
            simulation.pos,
        )

        _, m_flat_es, m_ids = (
            multi_layer_graph.get_multi_layer_graphs()
        )

        if len(m_flat_es) != unet_depth + 1:
            raise ValueError(
                f"Simulation {simulation.simulation_id}: "
                f"expected {unet_depth + 1} BSMS edge levels, "
                f"got {len(m_flat_es)}."
            )

        if len(m_ids) != unet_depth:
            raise ValueError(
                f"Simulation {simulation.simulation_id}: "
                f"expected {unet_depth} pooling levels, "
                f"got {len(m_ids)}."
            )

        bsms_hierarchies[simulation.simulation_id] = {
            "edge_indices": [
                torch.tensor(edges, dtype=torch.long)
                for edges in m_flat_es
            ],

            "pool_indices": [
                torch.tensor(indices, dtype=torch.long)
                for indices in m_ids
            ],

            "pos": torch.tensor(
                simulation.pos,
                dtype=torch.float32,
            ),
        }

        print(
            f"  Simulation {simulation.simulation_id}: "
            f"{simulation.num_nodes} fine nodes -> "
            f"{len(m_flat_es)} BSMS levels"
        )

    return bsms_hierarchies

def predict_velocity_for_simulation(
    velocity_model,
    simulation,
    velocity_encoding,
    device,
):
    """
    Predict normalized u(t+1), v(t+1) for every sample
    of one normalized simulation.

    Returns
    -------
    predictions : np.ndarray
        Shape (S, N, 2), still in normalized units.
    """

    features = gdata.build_features(
        velocity_encoding,
        simulation["X"],
        simulation["dt"],
    )

    edge_index = torch.as_tensor(
        simulation["edge_index"],
        dtype=torch.long,
        device=device,
    )

    edge_weight = torch.as_tensor(
        simulation["edge_weight"],
        dtype=torch.float32,
        device=device,
    )

    predictions = []

    velocity_model.eval()

    with torch.no_grad():

        for timestep in range(features.shape[0]):

            x = torch.as_tensor(
                features[timestep],
                dtype=torch.float32,
                device=device,
            )

            from torch_geometric.data import Data

            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

            prediction = velocity_model(graph)

            predictions.append(
                prediction.cpu().numpy()
            )

    return np.stack(
        predictions,
        axis=0,
    )

# =====================================================================
# Loaders
#
# Standard GNNs use PyG batches, which carry the mesh with every sample.
# BSMS uses dense tensor batches [B, N, F] per simulation, because all
# snapshots of one simulation share exactly the same mesh hierarchy.
# =====================================================================

def build_velocity_loaders(normalized, config):

    velocity_network = config["networks"]["velocity"]

    batch_size = config["training"]["batch_size"]

    velocity_loaders = {}

    for split_name in ("train", "val", "test"):

        velocity_dataset = (
            gdata.create_multi_simulation_graph_dataset(
                normalized[split_name],
                velocity_network["features"],
            )
        )

        velocity_loaders[split_name] = PyGDataLoader(
            velocity_dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
        )

        print(
            f"Velocity {split_name}: "
            f"{len(velocity_dataset)} graph samples"
        )

    return velocity_loaders


def build_pressure_loaders(
    normalized,
    config,
    velocity_model=None,
    device=None,
):
    """
    One BSMS loader per simulation.

    If use_predicted_velocity=True, normalized predictions
    u(t+1), v(t+1) from the trained velocity network are
    concatenated to the base pressure features.
    """

    pressure_loaders = {
        "train": {},
        "val": {},
        "test": {},
    }

    if not uses_bsms(config):
        return pressure_loaders

    pressure_network = config["networks"]["pressure"]

    pressure_encoding = pressure_network[
        "features"
    ]

    use_predicted_velocity = pressure_network.get(
        "use_predicted_velocity",
        False,
    )

    batch_size = config["training"]["batch_size"]

    if use_predicted_velocity:

        if velocity_model is None:
            raise ValueError(
                "Pressure uses predicted velocity, but no "
                "trained velocity model was provided."
            )

        if device is None:
            raise ValueError(
                "device is required when predicted velocity "
                "features are enabled."
            )

        velocity_encoding = config[
            "networks"
        ]["velocity"]["features"]

    for split_name in ("train", "val", "test"):

        for simulation in normalized[split_name]:

            simulation_id = simulation[
                "simulation_id"
            ]

            # -----------------------------------------------
            # Base features:
            # u(t), v(t), p(t), delta_t
            # -----------------------------------------------

            features = gdata.build_features(
                pressure_encoding,
                simulation["X"],
                simulation["dt"],
            )

            # -----------------------------------------------
            # Add predicted u(t+1), v(t+1)
            # -----------------------------------------------

            if use_predicted_velocity:

                predicted_velocity = (
                    predict_velocity_for_simulation(
                        velocity_model,
                        simulation,
                        velocity_encoding,
                        device,
                    )
                )

                if (
                    predicted_velocity.shape[:2]
                    != features.shape[:2]
                ):
                    raise ValueError(
                        f"Simulation {simulation_id}: "
                        "predicted velocity and pressure "
                        "features have incompatible shapes."
                    )

                features = np.concatenate(
                    [
                        features,
                        predicted_velocity,
                    ],
                    axis=-1,
                )

            pressure_dataset = gdata.create_bsms_dataset(
                features,
                simulation["Y"],
                delta_t=simulation["dt_physical"],
            )

            sample = pressure_dataset[0]

            print(
                f"DEBUG | simulation {simulation_id} | "
                f"items={len(sample)}"
            )

            pressure_loaders[
                split_name
            ][simulation_id] = TensorDataLoader(
                pressure_dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
            )

            print(
                f"Pressure {split_name} | "
                f"simulation {simulation_id}: "
                f"{len(pressure_dataset)} samples | "
                f"{simulation['X'].shape[1]} nodes | "
                f"{features.shape[-1]} features"
            )

    return pressure_loaders


# =====================================================================
# Training
# =====================================================================

# Hyperparameters recorded in the sweep table and in the checkpoint
SWEEP_KEYS = (
    "architecture",
    "num_neurons",
    "num_layers",
    "dropout",
    "learning_rate",
    "weight_decay"
)


def train_one_network(
    network_name,
    network,
    network_loaders,
    bsms_hierarchies,
    config,
    criterion,
    device,
    verbose
):
    """
    Train every hyperparameter combination and keep the best.

    Returns (best, sweep_rows), where best carries the winning weights,
    the concrete configuration that produced them and its loss history.
    """

    architecture = network["architecture"]

    columns = gdata.TARGET_COLUMNS[network["predicts"]]

    num_in = gdata.FEATURE_SIZES[
        network["features"]
    ]

    if network.get(
        "use_predicted_velocity",
        False,
    ):
        num_in += 2

    num_out = len(range(*columns.indices(3)))

    num_epochs = config["training"]["num_epochs"]

    combinations = expand_grid(network)

    print(f"\n{'=' * 60}")
    print(f"NETWORK {network_name!r} -> {network['predicts']}")
    print(
        f"{len(combinations)} configuration(s), "
        f"{num_in} in -> {num_out} out"
    )
    print("=" * 60)

    best = None
    sweep_rows = []

    for index, concrete in enumerate(combinations, start=1):

        if len(combinations) > 1:
            print(
                f"\n[{index}/{len(combinations)}] "
                f"neurons={concrete['num_neurons']} "
                f"layers={concrete['num_layers']} "
                f"lr={concrete['learning_rate']} "
                f"dropout={concrete['dropout']} "
                f"wd={concrete['weight_decay']}"
            )

        set_seeds(config["seed"])

        model_kwargs = {
            "architecture": concrete["architecture"],
            "num_in": num_in,
            "num_out": num_out,
            "num_neurons": concrete["num_neurons"],
            "num_layers": concrete["num_layers"],
            "dropout": concrete["dropout"],
        }

        if concrete["architecture"] == "bsms":

            model_kwargs.update(
                {
                    "unet_depth": concrete["unet_depth"],
                    "hidden_layers": concrete["hidden_layers"],
                    "pos_dim": 2,
                }
            )

        model = build_model(**model_kwargs).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=concrete["learning_rate"],
            weight_decay=concrete["weight_decay"]
        )

        if architecture == "bsms":

            state, train_history, val_history = (
                train_bsms_multi_simulation(
                    model,
                    network_loaders["train"],
                    network_loaders["val"],
                    bsms_hierarchies,
                    criterion,
                    optimizer,
                    num_epochs,
                    device,
                    target_columns=columns,
                    verbose=verbose,
                )
            )

        else:

            state, train_history, val_history = (
                train_network(
                    model,
                    network_loaders["train"],
                    network_loaders["val"],
                    criterion,
                    optimizer,
                    num_epochs,
                    device,
                    target_columns=columns,
                    verbose=verbose,
                )
            )

        best_val = min(val_history)

        sweep_rows.append(
            {
                "network": network_name,
                "val_loss": best_val,
                **{key: concrete[key] for key in SWEEP_KEYS},
                "unet_depth": concrete.get("unet_depth"),
                "hidden_layers": concrete.get("hidden_layers"),
            }
        )

        print(f"Best validation loss: {best_val:.6e}")

        if best is None or best_val < best["val_loss"]:
            best = {
                "val_loss": best_val,
                "state": state,
                "config": concrete,
                "model": model,
                "train_history": train_history,
                "val_history": val_history
            }

    # Carried along so the caller does not recompute them
    best["columns"] = columns
    best["num_in"] = num_in
    best["num_out"] = num_out

    return best, sweep_rows


def train_all_networks(
    config,
    normalized,
    velocity_loaders,
    bsms_hierarchies,
    normalizer,
    run_dir,
    criterion,
    device,
    verbose,
):
    """Train every network in the config, save it, and score it."""

    trained = {}
    sweep_rows = []
    pressure_loaders = None
    for network_name, network in config["networks"].items():

        architecture = network["architecture"]

        if architecture == "bsms":

            if network.get(
                "use_predicted_velocity",
                False,
            ):

                if "velocity" not in trained:
                    raise RuntimeError(
                        "Pressure requires predicted velocity, "
                        "so the velocity network must be trained first."
                    )

                print(
                    "\nGenerating predicted velocity "
                    "features for pressure..."
                )

                pressure_loaders = build_pressure_loaders(
                    normalized,
                    config,
                    velocity_model=trained["velocity"]["model"],
                    device=device,
                )

            else:

                pressure_loaders = build_pressure_loaders(
                    normalized,
                    config,
                )

            network_loaders = pressure_loaders

        else:

            network_loaders = velocity_loaders

        best, rows = train_one_network(
            network_name,
            network,
            network_loaders,
            bsms_hierarchies,
            config,
            criterion,
            device,
            verbose
        )

        sweep_rows.extend(rows)

        columns = best["columns"]

        # Restore the best weights of the best configuration
        best["model"].load_state_dict(best["state"])
        best["model"].eval()

        checkpoint_path = run_dir / f"{network_name}.pth"

        save_checkpoint(
            checkpoint_path,
            best["model"],
            normalizer,
            metadata={
                "experiment": config["name"],
                "network": network_name,
                "predicts": network["predicts"],
                "features": network["features"],
                "use_predicted_velocity": network.get(
                    "use_predicted_velocity",
                    False),
                "num_in": best["num_in"],
                "num_out": best["num_out"],
                "split_mode": config["split"]["mode"],
                "num_epochs": config["training"]["num_epochs"],
                "batch_size": config["training"]["batch_size"],

                **{
                    key: best["config"][key]
                    for key in SWEEP_KEYS
                },

                # BSMS-specific parameters.
                # None for architectures that do not use BSMS.
                "unet_depth": best["config"].get("unet_depth"),
                "hidden_layers": best["config"].get("hidden_layers"),
            }
        )

        plots.plot_loss(
            best["train_history"],
            best["val_history"],
            path=run_dir / f"{network_name}_loss.png",
            title=network_name
        )

        if architecture == "bsms":

            test_loss = evaluate_bsms_multi_simulation(
                best["model"],
                network_loaders["test"],
                bsms_hierarchies,
                criterion,
                device,
                target_columns=columns,
            )

        else:

            test_loss = evaluate_dataset(
                best["model"],
                network_loaders["test"],
                criterion,
                device,
                columns,
            )

        trained[network_name] = {
            "model": best["model"],
            "columns": columns,
            "features": network["features"],
            "use_predicted_velocity": network.get(
                "use_predicted_velocity",
                False
            ),
            "test_loss": test_loss,
        }

        print(
            f"Saved {checkpoint_path.name} | "
            f"test loss (normalized) "
            f"{trained[network_name]['test_loss']:.6e}"
        )

    return trained, sweep_rows


# =====================================================================
# Outputs
# =====================================================================

def write_outputs(run_dir, config, splits, trained, sweep_rows):

    # Which simulations feed each block. With a whole-simulation split
    # the three lists are disjoint; with a temporal or random one every
    # simulation appears in all three, because it was cut along its own
    # samples. split_mode is recorded so the lists can be read correctly.
    results = {
        "experiment": config["name"],

        "split_mode": config["split"]["mode"],

        "train_simulations": [
            sim.simulation_id for sim in splits.train
        ],

        "val_simulations": [
            sim.simulation_id for sim in splits.val
        ],

        "test_simulations": [
            sim.simulation_id for sim in splits.test
        ],

        "samples": {
            name: sum(
                sim.num_samples
                for sim in getattr(splits, name)
            )
            for name in ("train", "val", "test")
        },

        "test_loss_normalized": {
            name: spec["test_loss"]
            for name, spec in trained.items()
        },
    }

    with open(run_dir / "metrics.json", "w") as handle:
        json.dump(results, handle, indent=2)

    if len(sweep_rows) > len(config["networks"]):

        import csv

        sweep_path = run_dir / "sweep.csv"

        with open(sweep_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sweep_rows[0])
            writer.writeheader()
            writer.writerows(
                sorted(sweep_rows, key=lambda row: row["val_loss"])
            )

        print(f"Sweep table: {sweep_path}")


# =====================================================================
# The experiment
# =====================================================================

def main():

    args = parse_args()

    config = load_config(args.config)

    set_seeds(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    run_dir = create_run_dir(args.output_root, config)

    print(f"Device:  {device}")
    print(f"Run dir: {run_dir}")

    simulations = load_and_report_simulations(config)

    splits = split_and_report(simulations, config)

    normalizer, dt_mean, dt_std = build_normalizer(splits.train)

    normalized = normalize_all_splits(
        splits, normalizer, dt_mean, dt_std
    )

    bsms_hierarchies = build_bsms_hierarchies(simulations, config)

    velocity_loaders = build_velocity_loaders(normalized, config)
    

    trained, sweep_rows = train_all_networks(
    config,
    normalized,
    velocity_loaders,
    bsms_hierarchies,
    normalizer,
    run_dir,
    nn.MSELoss(),
    device,
    verbose=not args.quiet,
    )

    # NOTE: one-step inference in physical units used to run here, and
    # no longer does - the section was emptied during the multi-geometry
    # work. metrics.json therefore reports only the normalized test
    # loss, and nothing in Pa or m/s. evaluate.predict_next_timestep and
    # per_variable_metrics are still there, unused, ready to be wired
    # back in on an unseen TEST simulation.

    write_outputs(run_dir, config, splits, trained, sweep_rows)

    print(f"\nDone. Everything is in {run_dir}")


if __name__ == "__main__":
    main()
