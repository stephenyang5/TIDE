"""Batch-level masking for excluded feature channels (matches training/eval)."""

from __future__ import annotations


def mask_excluded_features(batch: dict, exclude_idxs: list[int]) -> dict:
    """Zero out values and point_mask for excluded feature indices.

    The model sees these features as completely unobserved for every patient.
    """
    if not exclude_idxs:
        return batch
    vals = batch["values"].clone()
    pm = batch["point_mask"].clone()
    for idx in exclude_idxs:
        vals[:, idx, :, :] = 0.0
        pm[:, idx, :, :] = 0.0
    return {**batch, "values": vals, "point_mask": pm}
