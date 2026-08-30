"""
Saving and loading trained models.

A checkpoint is not just weights: it also carries the StateNormalizer
used to build the targets it was trained on, and a description of the
run. Inference reads the scaling from the checkpoint instead of assuming
one.

This is what prevents the class of bug where a network trained on
physical pressure was de-standardized at inference: the two halves of
the transform used to live in different files and could drift apart
without anything failing. A checkpoint with no normalizer is rejected
loudly, because a wrong scaling produces plausible numbers rather than
an error, which is the worst way to fail.
"""

import torch

from .data.normalization import StateNormalizer


def save_checkpoint(path, model, normalizer, metadata=None):
    """
    Save weights, normalizer and run description.

    metadata must contain only primitive types, so the checkpoint stays
    loadable in weights-only mode.
    """

    if normalizer is None:
        raise ValueError(
            "save_checkpoint requires the normalizer used for training: "
            "without it the predictions of this model cannot be brought "
            "back to physical units."
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "normalizer": normalizer.to_dict(),
            "metadata": metadata or {}
        },
        path
    )


def load_checkpoint(path, model=None, device=None):
    """
    Load a checkpoint and return (model, normalizer, metadata).

    Raises on checkpoints saved before the normalizer was stored: those
    are raw state dictionaries whose target scaling is unknown, and
    guessing it is exactly the bug this module exists to prevent.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if not isinstance(checkpoint, dict) or "normalizer" not in checkpoint:
        raise ValueError(
            f"{path} has no normalizer stored in it: it is a legacy "
            "checkpoint, saved when the target scaling lived in the "
            "training script. The scaling of its predictions cannot be "
            "recovered - retrain, or load it manually if you know which "
            "convention produced it."
        )

    normalizer = StateNormalizer.from_dict(checkpoint["normalizer"])

    if model is not None:
        model.load_state_dict(checkpoint["state_dict"])

    return model, normalizer, checkpoint.get("metadata", {})
