#!/usr/bin/env python3

"""
One-step evaluation of a trained run on held-out data, in physical units.

    python scripts/evaluate_test.py outputs/<run>

Everything the script needs it takes from the run directory: which
simulations were held out, how the inputs were scaled, and which extra
features the pressure network was fed. Nothing about a specific run is
written in this file.

That is the point. The previous version carried the run directory, the
test dataset, the simulation id and the time-step scaling as constants
pasted into the source. They went stale as soon as a different
experiment was run - and a stale delta_t scaling does not raise, it just
feeds the network the wrong number and reports confident nonsense.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make src/ importable
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from gnn_comsol import data as gdata                        # noqa: E402
from gnn_comsol.checkpoints import read_checkpoint          # noqa: E402
from gnn_comsol.graph.bsms import BistrideMultiLayerGraph   # noqa: E402
from gnn_comsol.models import build_model                   # noqa: E402


# ===============================================================
# Arguments
# ===============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "run_dir",
        help="a run directory under outputs/"
    )

    parser.add_argument(
        "--simulation-id",
        type=int,
        default=None,
        help=(
            "which held-out simulation to evaluate. Defaults to the "
            "first one the run put in its test block."
        )
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "evaluate this .mat instead of the one the run held out. "
            "Only for a genuinely unseen file: the default is derived "
            "from the run itself, and is the safe choice."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="where to write figures and arrays (default: the run dir)"
    )

    parser.add_argument(
        "--dt-normalization",
        nargs=2,
        type=float,
        metavar=("MEAN", "STD"),
        default=None,
        help=(
            "time-step scaling, for checkpoints written before it was "
            "stored in them. Take the two numbers from the 'delta_t "
            "normalization' line of that run's output; guessing them "
            "produces plausible but wrong predictions."
        )
    )

    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="skip the pressure-error animation, which is the slow part"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="open the figures in a window as well as saving them"
    )

    return parser.parse_args()


# ===============================================================
# What the run held out
# ===============================================================

def read_run(run_dir):
    """The resolved config and the metrics of a finished run."""

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"

    for path in (config_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: {run_dir} does not look like a "
                "finished run directory."
            )

    with open(config_path) as handle:
        config = json.load(handle)

    with open(metrics_path) as handle:
        metrics = json.load(handle)

    return config, metrics


def choose_test_simulation(config, metrics, simulation_id):
    """
    Which .mat to evaluate, and which simulation id it is.

    Derived from the run rather than typed in, because the failure it
    prevents is silent: evaluating a geometry that was in the training
    block reports a training error and calls it a test error.
    """

    dataset = config["dataset"]

    paths = dataset.get("paths") or [dataset["path"]]

    test_ids = metrics["test_simulations"]

    if not test_ids:
        raise ValueError(
            "This run has no test simulations recorded in metrics.json."
        )

    if simulation_id is None:
        simulation_id = test_ids[0]

    elif simulation_id not in test_ids:
        raise ValueError(
            f"Simulation {simulation_id} was not in the test block of "
            f"this run: the held-out simulations are {test_ids}. "
            "Evaluating on a simulation the model was trained on does "
            "not measure generalisation."
        )

    if simulation_id >= len(paths):
        raise ValueError(
            f"Simulation {simulation_id} is out of range: the run used "
            f"{len(paths)} dataset(s)."
        )

    return paths[simulation_id], simulation_id


def held_out_samples(raw, config, split_mode):
    """
    The part of a simulation the run did NOT train on.

    With a whole-simulation split the answer is "all of it". With a
    temporal or random split the simulation was cut along its own
    samples and most of it WAS used for training, so the same split has
    to be recomputed here - otherwise the numbers below are largely a
    training error wearing a test label.

    Returns (evaluation dataset, indices into the full simulation).
    """

    if split_mode in ("simulation", "group"):
        return raw, np.arange(raw.num_samples)

    split = config["split"]

    _, _, test_indices = gdata.compute_split_indices(
        raw.num_samples,
        mode=split_mode,
        train_fraction=split["train_fraction"],
        val_fraction=split["val_fraction"],
        gap=split["gap"],
        seed=config["seed"]
    )

    return gdata.subset_simulation(raw, test_indices), test_indices


# ===============================================================
# Rebuilding a trained network
# ===============================================================

def build_from_checkpoint(checkpoint_path, device):
    """
    Rebuild a model from its own checkpoint.

    The metadata carries everything the constructor needs, so nothing
    here has to know which experiment produced the file.
    """

    # First pass: metadata and scalings, before the model exists
    bundle = read_checkpoint(checkpoint_path, device=device)

    metadata = bundle.metadata

    architecture = metadata["architecture"]

    model_kwargs = {
        "architecture": architecture,
        "num_in": metadata["num_in"],
        "num_out": metadata["num_out"],
        "num_neurons": metadata["num_neurons"],
        "num_layers": metadata["num_layers"],
        "dropout": metadata["dropout"],
    }

    if architecture == "bsms":

        model_kwargs.update(
            {
                "unet_depth": metadata["unet_depth"],
                "hidden_layers": metadata["hidden_layers"],
                "pos_dim": 2,
            }
        )

    model = build_model(**model_kwargs).to(device)

    # Second pass: the weights
    bundle = read_checkpoint(
        checkpoint_path,
        model=model,
        device=device
    )

    bundle.model.eval()

    return bundle


def agree(first_values, second_values, what, first, second):

    if not np.allclose(first_values, second_values):
        raise ValueError(
            f"{first} and {second} were saved with different {what}, "
            "so they cannot have come from the same run."
        )


def resolve_dt_normalization(bundle, override, checkpoint_path):
    """
    The time-step scaling, from the checkpoint or given explicitly.

    Never guessed. The scaling of delta_t is part of how the model was
    fed, and a wrong one changes every prediction without changing the
    shape of anything.
    """

    if override is not None:

        mean, std = override

        print(
            f"delta_t normalization: mean={mean:.6e}, std={std:.6e} "
            "(given on the command line)"
        )

        return mean, std

    if bundle.dt_mean is None or bundle.dt_std is None:
        raise ValueError(
            f"{checkpoint_path} does not carry the delta_t scaling: it "
            "was written before that was stored in checkpoints. Pass "
            "--dt-normalization MEAN STD, taking the two numbers from "
            "the 'delta_t normalization' line printed by that run, or "
            "retrain so the checkpoint carries them itself."
        )

    print(
        f"delta_t normalization: mean={bundle.dt_mean:.6e}, "
        f"std={bundle.dt_std:.6e} (from the checkpoint)"
    )

    return bundle.dt_mean, bundle.dt_std


# ===============================================================
# Inference
# ===============================================================

def evaluate(
    evaluation,
    velocity,
    pressure,
    normalizer,
    dt_mean,
    dt_std,
    hierarchy,
    device
):
    """
    One-step prediction for every held-out sample, in physical units.

    Returns per-sample RMSE for u, v and p, plus the absolute pressure
    error at every node.
    """

    X_norm = normalizer.transform(evaluation.X_input)

    dt_norm = (evaluation.delta_t - dt_mean) / dt_std

    velocity_features = gdata.build_features(
        velocity.metadata["features"],
        X_norm,
        dt_norm
    )

    # -----------------------------------------------------------
    # Physics-derived features, scaled the way training scaled them
    # -----------------------------------------------------------

    uses_physics = pressure.metadata.get("use_physics_features", False)

    physics_norm = None

    if uses_physics:

        if evaluation.physics_features is None:
            raise ValueError(
                "The pressure network was trained on physics-derived "
                "features, but this dataset does not contain any. It "
                "expects "
                f"{pressure.metadata.get('physics_feature_names')}."
            )

        if pressure.physics_normalizer is None:
            raise ValueError(
                "The pressure checkpoint says it uses physics features "
                "but does not carry their scaling, so they cannot be "
                "reproduced. Retrain with the current code."
            )

        physics_norm = pressure.physics_normalizer.transform(
            evaluation.physics_features
        )

    uses_predicted_velocity = pressure.metadata.get(
        "use_predicted_velocity",
        False
    )

    edge_index = torch.as_tensor(
        evaluation.edge_index,
        dtype=torch.long,
        device=device
    )

    edge_weight = torch.as_tensor(
        evaluation.edge_weight,
        dtype=torch.float32,
        device=device
    )

    num_samples = evaluation.num_samples

    rmse = {
        name: np.zeros(num_samples)
        for name in gdata.VARIABLE_NAMES
    }

    pressure_absolute_error = np.zeros(
        (num_samples, evaluation.num_nodes),
        dtype=np.float32
    )

    from torch_geometric.data import Data

    print(f"\nEvaluating {num_samples} held-out samples...")

    with torch.no_grad():

        for timestep in range(num_samples):

            # ---------------------------------------------------
            # Velocity
            # ---------------------------------------------------

            velocity_graph = Data(
                x=torch.as_tensor(
                    velocity_features[timestep],
                    dtype=torch.float32,
                    device=device
                ),
                edge_index=edge_index,
                edge_weight=edge_weight
            )

            velocity_pred_norm = velocity.model(velocity_graph)

            # ---------------------------------------------------
            # Pressure
            #
            # The feature vector is assembled by the same function the
            # training loaders use, so its layout cannot drift from
            # what the weights expect. Appending the blocks by hand
            # here was how the order could silently disagree.
            # ---------------------------------------------------

            predicted_velocity = None

            if uses_predicted_velocity:

                predicted_velocity = (
                    velocity_pred_norm.cpu().numpy()[None, ...]
                )

            pressure_features = gdata.build_pressure_features(
                pressure.metadata["features"],
                X_norm[timestep:timestep + 1],
                dt_norm[timestep:timestep + 1],
                physics_features=(
                    physics_norm[timestep:timestep + 1]
                    if physics_norm is not None
                    else None
                ),
                predicted_velocity=predicted_velocity
            )[0]

            expected = pressure.metadata["num_in"]

            if pressure_features.shape[-1] != expected:
                raise ValueError(
                    f"The pressure model expects {expected} input "
                    f"features, but {pressure_features.shape[-1]} were "
                    "assembled."
                )

            pressure_pred_norm = pressure.model(
                torch.as_tensor(
                    pressure_features,
                    dtype=torch.float32,
                    device=device
                ),
                hierarchy["pool_indices"],
                hierarchy["edge_indices"],
                hierarchy["pos"]
            )

            # ---------------------------------------------------
            # Back to physical units
            # ---------------------------------------------------

            Y_pred = normalizer.inverse_transform(
                torch.cat(
                    [velocity_pred_norm, pressure_pred_norm],
                    dim=-1
                )
            ).cpu().numpy()

            error = Y_pred - evaluation.Y_target[timestep]

            pressure_absolute_error[timestep] = np.abs(error[:, 2])

            for column, name in enumerate(gdata.VARIABLE_NAMES):
                rmse[name][timestep] = np.sqrt(
                    np.mean(error[:, column] ** 2)
                )

            if timestep % 100 == 0 or timestep == num_samples - 1:
                print(
                    f"  {timestep + 1}/{num_samples} | "
                    + " | ".join(
                        f"RMSE {name}={rmse[name][timestep]:.3e}"
                        for name in gdata.VARIABLE_NAMES
                    )
                )

    return rmse, pressure_absolute_error


def report(rmse, time):

    print("\n" + "=" * 60)
    print("HELD-OUT RMSE - PHYSICAL UNITS")
    print("=" * 60)

    for name, values in rmse.items():

        worst = int(np.argmax(values))

        print(f"\n{name}")
        print(f"  Mean RMSE:     {values.mean():.6e}")
        print(f"  Median RMSE:   {np.median(values):.6e}")
        print(f"  Min RMSE:      {values.min():.6e}")
        print(f"  Max RMSE:      {values.max():.6e}")
        print(f"  Max at sample: {worst}")
        print(f"  Max at time:   {time[worst]:.6e}")


# ===============================================================
# Figures
# ===============================================================

def plot_rmse(plt, rmse, time, output_dir):

    for name, values in rmse.items():

        for scale in ("linear", "log"):

            figure, axes = plt.subplots(figsize=(10, 5))

            axes.plot(time, values)

            if scale == "log":
                axes.set_yscale("log")

            axes.set_xlabel("Time")
            axes.set_ylabel(f"RMSE {name}")

            axes.set_title(
                f"Held-out set - {name} RMSE over time"
                + (" (log scale)" if scale == "log" else "")
            )

            axes.grid(True)

            figure.tight_layout()

            suffix = "_log" if scale == "log" else ""

            figure.savefig(
                output_dir / f"test_rmse_{name}{suffix}.png",
                dpi=200
            )


def mesh_triangulation(mtri, evaluation):
    """
    A triangulation that follows the mesh instead of the convex hull.

    Delaunay on the node positions alone happily draws triangles across
    a cavity or an obstacle, which paints error where there is no fluid.
    A triangle is kept only if all three of its edges exist in the graph.
    """

    triangulation = mtri.Triangulation(
        evaluation.pos[:, 0],
        evaluation.pos[:, 1]
    )

    edge_index = np.asarray(evaluation.edge_index)

    if edge_index.shape[0] != 2:
        edge_index = edge_index.T

    edges = {
        (int(min(i, j)), int(max(i, j)))
        for i, j in edge_index.T
    }

    def has_edge(i, j):
        return (min(i, j), max(i, j)) in edges

    triangles = triangulation.triangles

    mask = np.array(
        [
            not (
                has_edge(i, j)
                and has_edge(j, k)
                and has_edge(k, i)
            )
            for i, j, k in triangles
        ]
    )

    triangulation.set_mask(mask)

    print(
        f"Visualization triangulation: {len(triangles)} triangles | "
        f"{int(mask.sum())} removed | {int((~mask).sum())} retained"
    )

    return triangulation


def animate_pressure_error(
    plt,
    mtri,
    animation_module,
    evaluation,
    error,
    rmse_p,
    time,
    output_dir
):

    triangulation = mesh_triangulation(mtri, evaluation)

    # The 99th percentile keeps a handful of extreme nodes from washing
    # out the rest of the field.
    vmin, vmax = 0.0, float(np.percentile(error, 99))

    print(f"Color scale: {vmin:.3e} -> {vmax:.3e}")

    figure, axes = plt.subplots(figsize=(12, 5))

    first = axes.tripcolor(
        triangulation,
        error[0],
        shading="flat",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis"
    )

    colorbar = figure.colorbar(first, ax=axes)
    colorbar.set_label("Absolute pressure error")

    def update(frame):

        axes.clear()

        plot = axes.tripcolor(
            triangulation,
            error[frame],
            shading="flat",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis"
        )

        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        axes.set_title(
            f"Pressure absolute error | t = {time[frame]:.3f} | "
            f"RMSE = {rmse_p[frame]:.3e}"
        )

        return plot,

    figure_animation = animation_module.FuncAnimation(
        figure,
        update,
        frames=range(len(error)),
        interval=50,
        blit=False
    )

    path = output_dir / "pressure_error_animation.gif"

    figure_animation.save(
        path,
        writer=animation_module.PillowWriter(fps=15)
    )

    plt.close(figure)

    print(f"Pressure error animation saved to: {path}")


# ===============================================================
# The evaluation
# ===============================================================

def main():

    args = parse_args()

    import matplotlib

    if not args.show:
        # plt.show() blocks, which makes an unattended run hang forever
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib import animation as animation_module

    run_dir = Path(args.run_dir)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device:  {device}")
    print(f"Run dir: {run_dir}")

    config, metrics = read_run(run_dir)

    split_mode = metrics["split_mode"]

    print(f"Split:   {split_mode}")
    print(f"Held-out simulations: {metrics['test_simulations']}")

    # -----------------------------------------------------------
    # The trained networks
    # -----------------------------------------------------------

    velocity = build_from_checkpoint(run_dir / "velocity.pth", device)
    pressure = build_from_checkpoint(run_dir / "pressure.pth", device)

    print("\nVelocity metadata:", velocity.metadata)
    print("\nPressure metadata:", pressure.metadata)

    agree(
        velocity.normalizer.mean,
        pressure.normalizer.mean,
        "state means",
        "velocity.pth",
        "pressure.pth"
    )

    agree(
        velocity.normalizer.std,
        pressure.normalizer.std,
        "state standard deviations",
        "velocity.pth",
        "pressure.pth"
    )

    normalizer = velocity.normalizer

    dt_mean, dt_std = resolve_dt_normalization(
        velocity,
        args.dt_normalization,
        run_dir / "velocity.pth"
    )

    if args.dt_normalization is None and pressure.dt_mean is not None:

        agree(
            [velocity.dt_mean, velocity.dt_std],
            [pressure.dt_mean, pressure.dt_std],
            "delta_t scalings",
            "velocity.pth",
            "pressure.pth"
        )

    # -----------------------------------------------------------
    # The held-out data
    # -----------------------------------------------------------

    if args.dataset is not None:

        dataset_path = args.dataset
        simulation_id = args.simulation_id

        print(
            f"\nEvaluating {dataset_path} (given on the command line; "
            "it is on you that the run never saw it)"
        )

    else:

        dataset_path, simulation_id = choose_test_simulation(
            config,
            metrics,
            args.simulation_id
        )

        print(f"\nEvaluating held-out simulation {simulation_id}")

    raw = gdata.load_data(
        dataset_path,
        skip_initial=config["dataset"]["skip_initial"],
        simulation_id=simulation_id
    )

    time_axis = np.cumsum(raw.delta_t)

    evaluation, indices = held_out_samples(raw, config, split_mode)

    time = time_axis[indices]

    print(
        f"  {dataset_path}\n"
        f"  {raw.num_nodes} nodes | {raw.num_edges} edges | "
        f"{raw.num_samples} samples, of which "
        f"{evaluation.num_samples} were held out"
    )

    round_trip = np.max(
        np.abs(
            normalizer.inverse_transform(
                normalizer.transform(evaluation.Y_target)
            )
            - evaluation.Y_target
        )
    )

    print(f"  Normalization round-trip max error: {round_trip:.6e}")

    # -----------------------------------------------------------
    # The BSMS hierarchy of this mesh
    # -----------------------------------------------------------

    multi_layer_graph = BistrideMultiLayerGraph(
        evaluation.edge_index,
        pressure.metadata["unet_depth"],
        evaluation.num_nodes,
        evaluation.pos
    )

    _, flat_edges, pool_ids = multi_layer_graph.get_multi_layer_graphs()

    hierarchy = {
        "edge_indices": [
            torch.as_tensor(edges, dtype=torch.long, device=device)
            for edges in flat_edges
        ],
        "pool_indices": [
            torch.as_tensor(ids, dtype=torch.long, device=device)
            for ids in pool_ids
        ],
        "pos": torch.as_tensor(
            evaluation.pos,
            dtype=torch.float32,
            device=device
        ),
    }

    # -----------------------------------------------------------
    # Run it
    # -----------------------------------------------------------

    rmse, pressure_absolute_error = evaluate(
        evaluation,
        velocity,
        pressure,
        normalizer,
        dt_mean,
        dt_std,
        hierarchy,
        device
    )

    report(rmse, time)

    np.savez(
        output_dir / "test_rmse_over_time.npz",
        time=time,
        indices=indices,
        simulation_id=simulation_id,
        dataset=str(dataset_path),
        pressure_absolute_error=pressure_absolute_error,
        **{f"rmse_{name}": values for name, values in rmse.items()}
    )

    plot_rmse(plt, rmse, time, output_dir)

    if not args.no_animation:

        print("\nCreating pressure error animation...")

        animate_pressure_error(
            plt,
            mtri,
            animation_module,
            evaluation,
            pressure_absolute_error,
            rmse["p"],
            time,
            output_dir
        )

    if args.show:
        plt.show()

    print(f"\nDone. Everything is in {output_dir}")


if __name__ == "__main__":
    main()
