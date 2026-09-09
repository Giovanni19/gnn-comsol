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

from dataclasses import dataclass

import torch
from .data.normalization import (
    GeometryNormalizer,
    PhysicsNormalizer,
    StateNormalizer,
)


@dataclass
class CheckpointBundle:
    """
    Everything a checkpoint knows about how its model was fed.

    Weights alone are not enough to run a model: every scaling applied
    to build its inputs and its targets has to come back with them, or
    whoever loads it has to guess - and a wrong scaling produces
    plausible numbers instead of an error.

    normalizer : StateNormalizer
        Scaling of (u, v, p). Always present.

    physics_normalizer : PhysicsNormalizer or None
        Scaling of the physics-derived features. None for models that
        were not fed them.

    geometry_normalizer : GeometryNormalizer or None
        Scaling of the static per-node geometry features. None for
        models that were not fed them.

    delta_normalizer : StateNormalizer or None
        Scaling of the one-step increment `Y_target - X_input`. Only
        present for a model trained with `predict_delta: true`; it is a
        StateNormalizer because the increment has the same 3-column
        (u, v, p) shape as the state, just fitted on different data.

    dt_mean, dt_std : float or None
        Scaling of the time step. None for checkpoints written before
        these were stored; callers must refuse to guess them.

    metadata : dict
        How the model was built and what it was trained on.
    """

    model: object
    normalizer: StateNormalizer
    physics_normalizer: PhysicsNormalizer | None
    geometry_normalizer: GeometryNormalizer | None
    delta_normalizer: StateNormalizer | None
    dt_mean: float | None
    dt_std: float | None
    metadata: dict


def save_checkpoint(
    path,
    model,
    normalizer,
    metadata=None,
    physics_normalizer=None,
    geometry_normalizer=None,
    delta_normalizer=None,
    dt_normalization=None,
):
    """
    Save weights, every scaling used to build the inputs and targets,
    and a description of the run.

    Parameters
    ----------
    physics_normalizer : PhysicsNormalizer or None
        Required by models fed physics-derived input features.

    geometry_normalizer : GeometryNormalizer or None
        Required by models fed the static per-node geometry features.

    delta_normalizer : StateNormalizer or None
        Required by models trained with `predict_delta: true`, to bring
        their normalized increment prediction back to a physical
        increment before it is added to the input state.

    dt_normalization : (mean, std) or None
        Scaling of the time-step feature. It belongs here for exactly
        the same reason the state scaling does: it used to live only in
        the stdout of a training run, and the evaluation script carried
        a copy of it pasted into the source, which silently went stale
        the moment the dataset list changed.
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

    if geometry_normalizer is not None:
        checkpoint["geometry_normalizer"] = (
            geometry_normalizer.to_dict()
        )

    if delta_normalizer is not None:
        checkpoint["delta_normalizer"] = (
            delta_normalizer.to_dict()
        )

    if dt_normalization is not None:

        dt_mean, dt_std = dt_normalization

        if not dt_std > 0:
            raise ValueError(
                f"dt_std must be strictly positive, got {dt_std}."
            )

        checkpoint["dt_normalization"] = {
            "mean": float(dt_mean),
            "std": float(dt_std),
        }

    torch.save(
        checkpoint,
        path,
    )


def read_checkpoint(path, model=None, device=None):
    """
    Read a checkpoint once and return everything it carries.

    Raises on checkpoints saved before the normalizer was stored: those
    are raw state dictionaries whose target scaling is unknown, and
    guessing it is exactly the bug this module exists to prevent.

    Returns
    -------
    CheckpointBundle
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

    physics_state = checkpoint.get("physics_normalizer")

    physics_normalizer = (
        PhysicsNormalizer.from_dict(physics_state)
        if physics_state is not None
        else None
    )

    geometry_state = checkpoint.get("geometry_normalizer")

    geometry_normalizer = (
        GeometryNormalizer.from_dict(geometry_state)
        if geometry_state is not None
        else None
    )

    delta_state = checkpoint.get("delta_normalizer")

    delta_normalizer = (
        StateNormalizer.from_dict(delta_state)
        if delta_state is not None
        else None
    )

    dt_normalization = checkpoint.get("dt_normalization") or {}

    if model is not None:
        model.load_state_dict(checkpoint["state_dict"])

    return CheckpointBundle(
        model=model,
        normalizer=normalizer,
        physics_normalizer=physics_normalizer,
        geometry_normalizer=geometry_normalizer,
        delta_normalizer=delta_normalizer,
        dt_mean=dt_normalization.get("mean"),
        dt_std=dt_normalization.get("std"),
        metadata=checkpoint.get("metadata", {}),
    )


def load_checkpoint(path, model=None, device=None):
    """
    (model, normalizer, metadata) - the short form of read_checkpoint.

    Use read_checkpoint when the physics or time-step scalings matter.
    """

    bundle = read_checkpoint(path, model=model, device=device)

    return bundle.model, bundle.normalizer, bundle.metadata


def load_physics_normalizer(
    path,
    device=None,
    required=False,
):
    """
    The PhysicsNormalizer stored in a checkpoint, or None.

    Parameters
    ----------
    required : bool
        If True, raise when the checkpoint does not carry one.
        If False, return None for models that do not use
        physics-derived features.
    """

    physics_normalizer = read_checkpoint(
        path,
        device=device,
    ).physics_normalizer

    if physics_normalizer is None and required:
        raise ValueError(
            f"{path} has no physics normalizer stored in it, but this "
            "model requires physics-derived input features."
        )

    return physics_normalizer


def load_geometry_normalizer(
    path,
    device=None,
    required=False,
):
    """
    The GeometryNormalizer stored in a checkpoint, or None.

    Parameters
    ----------
    required : bool
        If True, raise when the checkpoint does not carry one.
        If False, return None for models that do not use
        geometry-derived features.
    """

    geometry_normalizer = read_checkpoint(
        path,
        device=device,
    ).geometry_normalizer

    if geometry_normalizer is None and required:
        raise ValueError(
            f"{path} has no geometry normalizer stored in it, but this "
            "model requires geometry-derived input features."
        )

    return geometry_normalizer


def load_delta_normalizer(
    path,
    device=None,
    required=False,
):
    """
    The delta StateNormalizer stored in a checkpoint, or None.

    Parameters
    ----------
    required : bool
        If True, raise when the checkpoint does not carry one.
        If False, return None for models trained to predict the
        absolute state rather than the one-step increment.
    """

    delta_normalizer = read_checkpoint(
        path,
        device=device,
    ).delta_normalizer

    if delta_normalizer is None and required:
        raise ValueError(
            f"{path} has no delta normalizer stored in it, but this "
            "model was trained with predict_delta: true."
        )

    return delta_normalizer