"""
Figures.

Every function saves to a file and only optionally opens a window:
plt.show() is blocking, which stalls an unattended run.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def plot_loss(train_losses, val_losses, path=None, title="Training"):
    """Training and validation loss against epoch, on a log scale."""

    epochs = range(1, len(train_losses) + 1)

    figure, axes = plt.subplots(figsize=(8, 5))

    axes.plot(epochs, train_losses, label="Training loss")
    axes.plot(epochs, val_losses, label="Validation loss")

    axes.set_xlabel("Epoch")
    axes.set_ylabel("MSE loss (normalized units)")
    axes.set_title(f"{title}: training and validation loss")

    # Losses span orders of magnitude; a linear axis hides the tail
    axes.set_yscale("log")

    axes.legend()
    axes.grid(True, which="both", alpha=0.3)

    figure.tight_layout()

    if path is not None:
        figure.savefig(path, dpi=150)

    plt.close(figure)

    return path


def plot_pressure_statistics(X, path=None):
    """
    Mean, min and max pressure over time.

    Worth looking at before choosing how to scale the pressure target: if
    the mean level drifts, it may be a gauge offset carrying no physical
    information, and it will still dominate an MSE computed on absolute
    pressure.
    """

    p = X[:, :, 2]

    timesteps = np.arange(X.shape[0])

    figure, axes = plt.subplots(figsize=(9, 5))

    axes.plot(timesteps, np.mean(p, axis=1), label="Mean pressure")
    axes.plot(timesteps, np.max(p, axis=1), label="Max pressure")
    axes.plot(timesteps, np.min(p, axis=1), label="Min pressure")

    axes.set_xlabel("Timestep")
    axes.set_ylabel("Pressure")
    axes.set_title("Pressure statistics over time")

    axes.legend()
    axes.grid(True, alpha=0.3)

    figure.tight_layout()

    if path is not None:
        figure.savefig(path, dpi=150)

    plt.close(figure)

    return path
