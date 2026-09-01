#!/usr/bin/env python3
"""
Run one experiment, described by a YAML file in configs/.

    python scripts/run_experiment.py configs/virtual_node.yaml

Everything a run produces goes into outputs/<name>_<timestamp>/:
the resolved configuration, one checkpoint per network, the loss curves,
the sweep table if there was one, and the metrics as JSON.
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
    evaluate_dataset,
    format_metrics,
    per_variable_metrics,
    predict_next_timestep,
    evaluate_bsms_multi_simulation,
)
from gnn_comsol.models import build_model                 # noqa: E402
from gnn_comsol.train import train_network, train_bsms_multi_simulation                # noqa: E402
from gnn_comsol.graph.bsms import BistrideMultiLayerGraph

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

def predict_velocity_features(
    model,
    features,
    edge_index,
    edge_weight,
    device,
):
    """
    Run the trained velocity network on every snapshot and return
    normalized predictions of u(t+1), v(t+1).

    Returns
    -------
    predictions : np.ndarray
        Shape [num_samples, num_nodes, 2].
    """

    model.eval()

    dataset = gdata.create_graph_dataset(
        features,
        np.zeros(
            (features.shape[0], features.shape[1], 3),
            dtype=np.float32,
        ),
        edge_index,
        edge_weight,
    )

    loader = PyGDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    predictions = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            output = model(batch)

            predictions.append(
                output.detach().cpu().numpy()[None, ...]
            )

    return np.concatenate(predictions, axis=0)

def main():

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

    args = parser.parse_args()

    config = load_config(args.config)

    set_seeds(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    run_dir = Path(args.output_root) / (
        f"{config['name']}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    print(f"Device:  {device}")
    print(f"Run dir: {run_dir}")

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------

    simulations = gdata.load_simulations(
        config["dataset"]["paths"],
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

    split_config = config["split"]

    splits = gdata.split_simulations(
        simulations,
        train_fraction=split_config["train_fraction"],
        seed=config["seed"]
    )
    print("\nSimulation split:")

    for name in ("train", "val", "test"):

        split_simulations = getattr(splits, name)

        print(
            f"\n{name.upper()} "
            f"({len(split_simulations)} simulations)"
        )

        for simulation in split_simulations:

            print(
                f"  Simulation {simulation.simulation_id}: "
                f"{simulation.num_nodes} nodes | "
                f"{simulation.num_samples} samples | "
                f"{simulation.file_path}"
            )

    # ---------------------------------------------------------------
    # Scaling, computed on the training block only
    # ---------------------------------------------------------------

    x_mean, x_std, dt_mean, dt_std = (
        gdata.compute_multi_simulation_normalization_parameters(
            splits.train
        )
    )

    normalizer = gdata.StateNormalizer(
        x_mean,
        x_std
    )

    print(f"\n{normalizer}")

    print(
        f"delta_t normalization: "
        f"mean={dt_mean:.6e}, std={dt_std:.6e}"
    )


    blocks = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }

    normalized = {}

    for split_name, split_simulations in blocks.items():

        normalized[split_name] = []

        for simulation in split_simulations:

            normalized_simulation = {
                "X": normalizer.transform(
                    simulation.X_input
                ),

                "Y": normalizer.transform(
                    simulation.Y_target
                ),

                "dt": (
                    simulation.delta_t - dt_mean
                ) / dt_std,

                # Keep the graph belonging to this simulation
                "edge_index": simulation.edge_index,
                "edge_weight": simulation.edge_weight,
                "pos": simulation.pos,

                "simulation_id": simulation.simulation_id,
                "file_path": simulation.file_path,
            }

            normalized[split_name].append(
                normalized_simulation
            )

    
# ---------------------------------------------------------------
# BSMS hierarchies
# ---------------------------------------------------------------

    uses_bsms = any(
        network["architecture"] == "bsms"
        for network in config["networks"].values()
    )

    bsms_hierarchies = {}

    if uses_bsms:

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

            bsms_hierarchies[
                simulation.simulation_id
            ] = {
                "edge_indices": [
                    torch.tensor(
                        edges,
                        dtype=torch.long,
                    )
                    for edges in m_flat_es
                ],

                "pool_indices": [
                    torch.tensor(
                        indices,
                        dtype=torch.long,
                    )
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
    # ---------------------------------------------------------------
    # One dataset per feature encoding, shared by the networks that use
    # it. The target is always the full normalized state; each network
    # selects its own columns inside the training loop.
    # ---------------------------------------------------------------

    

    # ---------------------------------------------------------------
    # One loader set per network.
    #
    # Standard GNNs use PyG batches.
    # BSMS uses dense tensor batches [B, N, F] because every snapshot
    # shares exactly the same mesh hierarchy.
    # ---------------------------------------------------------------
    batch_size = config["training"]["batch_size"]


    # ===============================================================
    # Velocity loaders
    # ===============================================================

    velocity_network = config["networks"]["velocity"]

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


    # ===============================================================
    # Pressure BSMS loaders
    # ===============================================================

    pressure_loaders = {
        "train": {},
        "val": {},
        "test": {},
    }

    if uses_bsms:

        pressure_network = config["networks"]["pressure"]
        pressure_encoding = pressure_network["features"]

        for split_name in ("train", "val", "test"):

            for simulation in normalized[split_name]:

                simulation_id = simulation["simulation_id"]

                features = gdata.build_features(
                    pressure_encoding,
                    simulation["X"],
                    simulation["dt"],
                )

                pressure_dataset = gdata.create_bsms_dataset(
                    features,
                    simulation["Y"],
                )

                pressure_loader = TensorDataLoader(
                    pressure_dataset,
                    batch_size=batch_size,
                    shuffle=(split_name == "train"),
                )

                pressure_loaders[
                    split_name
                ][simulation_id] = pressure_loader

                print(
                    f"Pressure {split_name} | "
                    f"simulation {simulation_id}: "
                    f"{len(pressure_dataset)} samples | "
                    f"{simulation['X'].shape[1]} nodes"
                )
    # ---------------------------------------------------------------
    # Train each network
    # ---------------------------------------------------------------

    criterion = nn.HuberLoss(delta=1.0)

    num_epochs = config["training"]["num_epochs"]

    trained = {}
    sweep_rows = []

    for network_name, network in config["networks"].items():

        # ---------------------------------------------------------
        # TEMPORARY TEST:
        # validate multi-geometry training on velocity first.
        # Pressure / BSMS will be added afterwards.
        # ---------------------------------------------------------


        encoding = network["features"]
        architecture = network["architecture"]

        if architecture == "bsms":
            network_loaders = pressure_loaders
        else:
            network_loaders = velocity_loaders

        use_predicted_velocity = False

        columns = gdata.TARGET_COLUMNS[
            network["predicts"]
        ]

        num_in = gdata.FEATURE_SIZES[
            network["features"]
        ]

        num_out = len(
            range(*columns.indices(3))
        )

        combinations = expand_grid(network)

        print(f"\n{'=' * 60}")
        print(f"NETWORK {network_name!r} -> {network['predicts']}")
        print(
            f"{len(combinations)} configuration(s), "
            f"{num_in} in -> {num_out} out"
        )
        print("=" * 60)

        best = None

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

            model = build_model(
                **model_kwargs
            ).to(device)

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
                        verbose=not args.quiet,
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
                        verbose=not args.quiet,
                    )
                )

            best_val = min(val_history)

            sweep_rows.append(
                {
                    "network": network_name,
                    "val_loss": best_val,
                    **{
                        key: concrete[key]
                        for key in (
                            "architecture",
                            "num_neurons",
                            "num_layers",
                            "dropout",
                            "learning_rate",
                            "weight_decay"
                        )
                    },
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
                "num_in": num_in,
                "num_out": num_out,
                "split_mode": split_config["mode"],
                "num_epochs": num_epochs,
                "batch_size": batch_size,

                **{
                    key: best["config"][key]
                    for key in (
                        "architecture",
                        "num_neurons",
                        "num_layers",
                        "dropout",
                        "learning_rate",
                        "weight_decay"
                    )
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

    # ---------------------------------------------------------------
    # One-step inference on a snapshot from the TEST block
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # One-step inference on an unseen TEST simulation
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Persist the numbers
    # ---------------------------------------------------------------

    results = {
        "experiment": config["name"],

        "train_simulations": [
            sim.simulation_id
            for sim in splits.train
        ],

        "val_simulations": [
            sim.simulation_id
            for sim in splits.val
        ],

        "test_simulations": [
            sim.simulation_id
            for sim in splits.test
        ],

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

    print(f"\nDone. Everything is in {run_dir}")


if __name__ == "__main__":
    main()
