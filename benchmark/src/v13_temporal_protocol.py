from __future__ import annotations

import numpy as np


SECONDS_PER_DAY = 86_400.0


def last_observed_elapsed_days(mask: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
    """Return one prospective prediction anchor per sequence.

    The anchor is the elapsed time of the last retained observed event.  A
    sequence with no retained event is an admission-time/static-only update.
    """
    observed = np.asarray(mask).astype(bool).any(axis=2)
    times = np.asarray(elapsed, dtype=np.float64)
    if observed.shape != times.shape:
        raise ValueError("mask/elapsed anchor inputs are not aligned")
    safe = np.where(observed, times, -np.inf)
    anchor_seconds = np.max(safe, axis=1)
    anchor_seconds[~observed.any(axis=1)] = 0.0
    return (np.maximum(anchor_seconds, 0.0) / SECONDS_PER_DAY).astype(np.float32)


def remaining_los_labels(
    total_los_days: np.ndarray,
    mask: np.ndarray,
    elapsed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.asarray(total_los_days, dtype=np.float32)
    anchors = last_observed_elapsed_days(mask, elapsed)
    if total.shape != anchors.shape:
        raise ValueError("LOS/anchor inputs are not aligned")
    remaining = np.maximum(total - anchors, 0.0).astype(np.float32)
    return remaining, anchors


def deterministic_event_anchor_indices(
    mask: np.ndarray,
    elapsed: np.ndarray,
    *,
    minimum_elapsed_hours: float = 2.0,
    maximum_anchors: int = 4,
) -> list[int]:
    """Select up to four ordered event anchors without using outcomes."""
    observed_rows = np.asarray(mask).astype(bool).any(axis=1)
    times = np.asarray(elapsed, dtype=np.float64)
    candidates = np.flatnonzero(
        observed_rows & (times >= float(minimum_elapsed_hours) * 3600.0)
    )
    if not len(candidates):
        candidates = np.flatnonzero(observed_rows)
    if not len(candidates):
        return [-1]
    if len(candidates) <= maximum_anchors:
        return [int(index) for index in candidates]
    positions = np.linspace(0, len(candidates) - 1, maximum_anchors).round().astype(int)
    return [int(candidates[position]) for position in np.unique(positions)]


def retain_through_anchor(
    values: np.ndarray,
    mask: np.ndarray,
    anchor_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    values_out = np.asarray(values, dtype=np.float32).copy()
    mask_out = (np.asarray(mask) > 0).astype(np.uint8, copy=True)
    if anchor_index >= 0:
        mask_out[anchor_index + 1 :, :] = 0
    else:
        mask_out[:, :] = 0
    values_out[mask_out == 0] = np.nan
    return values_out, mask_out
