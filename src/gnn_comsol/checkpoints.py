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
from .data.normalization import (
    PhysicsNormalizer,
    StateNormalizer,
)

def save_checkpoint(
    path,
    model,
    normalizer,
    metadata=None,
    physics_normalizer=None,
):
    """
    Save weights, state normalizer, optional physics normalizer
    and run description.

    The physics normalizer is required by models that use
    physics-derived input features.
    """

    if normalizer is None:
        raise ValueError(
            "save_checkpoint requires the normalizer used for training: "
            "without it the predictions of this model cannot be brought "
            "back to physical units."
        )

    checkpoint = {
        "state_dict": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "metadata": metadata or {},
    }

    if physics_normalizer is not None:
        checkpoint["physics_normalizer"] = (
            physics_normalizer.to_dict()
        )

    torch.save(
        checkpoint,
        path,
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

def load_physics_normalizer(
    path,
    device=None,
    required=False,
):
    """
    Load the PhysicsNormalizer stored in a checkpoint.

    Parameters
    ----------
    required : bool
        If True, raise an error when the checkpoint does not
        contain a physics normalizer.

        If False, return None for checkpoints/models that do
        not use physics-derived features.
    """

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    state = checkpoint.get(
        "physics_normalizer"
    )

    if state is None:

        if required:
            raise ValueError(
                f"{path} has no physics normalizer stored "
                "in it, but this model requires "
                "physics-derived input features."
            )

        return None

    return PhysicsNormalizer.from_dict(
        state
    )