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
from torch_geometric.loader import DataLoader

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
    predict_next_timestep
)
from gnn_comsol.models import build_model                 # noqa: E402
from gnn_comsol.train import train_network                # noqa: E402


def set_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)


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

    raw = gdata.load_data(
        config["dataset"]["path"],
        skip_initial=config["dataset"]["skip_initial"]
    )

    print(
        f"\nDataset: {raw.num_samples} samples, "
        f"{raw.num_nodes} nodes, {raw.num_edges} edges "
        f"({config['dataset']['skip_initial']} initial snapshot(s) "
        f"dropped)"
    )

    # If the solver used a constant step this is ~0 and the time feature
    # carries no information; if it is large the step varies a lot.
    print(
        f"Time step: mean={raw.delta_t.mean():.4e} "
        f"min={raw.delta_t.min():.4e} "
        f"max={raw.delta_t.max():.4e} "
        f"| coeff. of variation="
        f"{raw.delta_t.std() / raw.delta_t.mean():.4f}"
    )

    split_config = config["split"]

    splits = gdata.split_dataset(
        raw,
        mode=split_config["mode"],
        train_fraction=split_config["train_fraction"],
        val_fraction=split_config["val_fraction"],
        gap=split_config["gap"],
        seed=config["seed"]
    )

    print(gdata.format_split_statistics(splits))

    # ---------------------------------------------------------------
    # Scaling, computed on the training block only
    # ---------------------------------------------------------------

    x_mean, x_std, dt_mean, dt_std = (
        gdata.compute_normalization_parameters(
            splits.train.X,
            splits.train.dt
        )
    )

    normalizer = gdata.StateNormalizer(x_mean, x_std)

    print(f"\n{normalizer}")

    blocks = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test
    }

    normalized = {
        name: {
            "X": normalizer.transform(block.X),
            "Y": normalizer.transform(block.Y),
            "dt": (block.dt - dt_mean) / dt_std
        }
        for name, block in blocks.items()
    }

    edge_index = gdata.to_tensor(raw.edge_index, dtype=torch.long)
    edge_weight = gdata.to_tensor(raw.edge_weight)

    # ---------------------------------------------------------------
    # One dataset per feature encoding, shared by the networks that use
    # it. The target is always the full normalized state; each network
    # selects its own columns inside the training loop.
    # ---------------------------------------------------------------

    encodings = {
        network["features"]
        for network in config["networks"].values()
    }

    loaders = {}

    batch_size = config["training"]["batch_size"]

    for encoding in encodings:

        loaders[encoding] = {}

        for name in blocks:

            features = gdata.build_features(
                encoding,
                normalized[name]["X"],
                normalized[name]["dt"]
            )

            dataset = gdata.create_graph_dataset(
                features,
                normalized[name]["Y"],
                edge_index,
                edge_weight
            )

            loaders[encoding][name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(name == "train")
            )

        print(
            f"Encoding {encoding!r}: "
            f"{gdata.FEATURE_SIZES[encoding]} input features"
        )

    # ---------------------------------------------------------------
    # Train each network
    # ---------------------------------------------------------------

    criterion = nn.MSELoss()

    num_epochs = config["training"]["num_epochs"]

    trained = {}
    sweep_rows = []

    for network_name, network in config["networks"].items():

        columns = gdata.TARGET_COLUMNS[network["predicts"]]

        num_in = gdata.FEATURE_SIZES[network["features"]]
        num_out = len(range(*columns.indices(3)))

        encoding_loaders = loaders[network["features"]]

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

            model = build_model(
                architecture=concrete["architecture"],
                num_in=num_in,
                num_out=num_out,
                num_neurons=concrete["num_neurons"],
                num_layers=concrete["num_layers"],
                dropout=concrete["dropout"]
            ).to(device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=concrete["learning_rate"],
                weight_decay=concrete["weight_decay"]
            )

            state, train_history, val_history = train_network(
                model,
                encoding_loaders["train"],
                encoding_loaders["val"],
                criterion,
                optimizer,
                num_epochs,
                device,
                target_columns=columns,
                verbose=not args.quiet
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
                    }
                }
            )

            print(f"Best validation MSE: {best_val:.6e}")

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
                }
            }
        )

        plots.plot_loss(
            best["train_history"],
            best["val_history"],
            path=run_dir / f"{network_name}_loss.png",
            title=network_name
        )

        trained[network_name] = {
            "model": best["model"],
            "columns": columns,
            "features": network["features"],
            "test_loss": evaluate_dataset(
                best["model"],
                encoding_loaders["test"],
                criterion,
                device,
                columns
            )
        }

        print(
            f"Saved {checkpoint_path.name} | "
            f"test MSE (normalized) "
            f"{trained[network_name]['test_loss']:.6e}"
        )

    # ---------------------------------------------------------------
    # One-step inference on a snapshot from the TEST block
    # ---------------------------------------------------------------

    test_indices = splits.test.indices

    timestep = int(test_indices[len(test_indices) // 2])

    print(f"\nInference on timestep {timestep} (test block)")

    Y_pred = predict_next_timestep(
        trained,
        raw.X_input[timestep],
        raw.delta_t[timestep],
        edge_index,
        edge_weight,
        normalizer,
        dt_mean,
        dt_std,
        device
    )

    metrics = per_variable_metrics(raw.Y_target[timestep], Y_pred)

    print(format_metrics(metrics))

    # ---------------------------------------------------------------
    # Persist the numbers
    # ---------------------------------------------------------------

    results = {
        "experiment": config["name"],
        "timestep": timestep,
        "test_loss_normalized": {
            name: spec["test_loss"]
            for name, spec in trained.items()
        },
        "inference_metrics_physical": metrics
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
