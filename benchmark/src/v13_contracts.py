from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


SUMMARY_NAMES = ("last", "mean", "max", "min", "std")


def strict_flat_summaries(
    values: np.ndarray,
    mask: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return exactly 5 x F raw summaries, with no static or observedness fields."""
    if values.ndim != 3 or mask.shape != values.shape:
        raise ValueError("values and mask must have matching [N, T, F] shapes")
    if values.shape[2] != len(feature_names):
        raise ValueError("feature name count does not match the dynamic dimension")

    observed = mask > 0
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    counts = observed.sum(axis=1).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    sums = (clean * observed).sum(axis=1)
    means = sums / safe_counts
    sum_squares = ((clean * clean) * observed).sum(axis=1)
    stds = np.sqrt(np.maximum(sum_squares / safe_counts - means * means, 0.0))

    last = np.zeros((values.shape[0], values.shape[2]), dtype=np.float32)
    maxima = np.full_like(last, -np.inf)
    minima = np.full_like(last, np.inf)
    for time_index in range(values.shape[1]):
        seen = observed[:, time_index, :]
        row = clean[:, time_index, :]
        last = np.where(seen, row, last)
        maxima = np.where(seen, np.maximum(maxima, row), maxima)
        minima = np.where(seen, np.minimum(minima, row), minima)
    never = counts == 0
    maxima[never] = 0.0
    minima[never] = 0.0
    means[never] = 0.0
    stds[never] = 0.0

    matrix = np.concatenate([last, means, maxima, minima, stds], axis=1).astype(np.float32)
    names = [f"{summary}__{feature}" for summary in SUMMARY_NAMES for feature in feature_names]
    expected = 5 * len(feature_names)
    if matrix.shape[1] != expected or len(names) != expected:
        raise AssertionError("strict XGBOOST-FLAT contract did not produce 5 x F columns")
    if any("observ" in name.lower() or "missing" in name.lower() for name in names):
        raise AssertionError("observedness leaked into strict XGBOOST-FLAT")
    return matrix, names


def subject_grouped_partitions(
    stay_ids: list[Any],
    stay_subjects: dict[Any, Any],
    mortality_labels: dict[Any, int],
    seed: int,
) -> dict[str, list[Any]]:
    subject_stays: dict[Any, list[Any]] = defaultdict(list)
    for stay_id in stay_ids:
        subject_stays[stay_subjects[stay_id]].append(stay_id)
    subjects = sorted(subject_stays, key=lambda value: str(value))
    subject_labels = [max(int(mortality_labels[stay]) for stay in subject_stays[sub]) for sub in subjects]
    train_subjects, remainder = train_test_split(
        subjects,
        train_size=0.60,
        random_state=seed,
        stratify=subject_labels,
    )
    remainder_labels = [max(int(mortality_labels[stay]) for stay in subject_stays[sub]) for sub in remainder]
    validation_subjects, test_subjects = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=remainder_labels,
    )
    subject_sets = {
        "train": set(train_subjects),
        "validation": set(validation_subjects),
        "test": set(test_subjects),
    }
    if subject_sets["train"] & subject_sets["validation"]:
        raise AssertionError("train/validation subject overlap")
    if subject_sets["train"] & subject_sets["test"]:
        raise AssertionError("train/test subject overlap")
    if subject_sets["validation"] & subject_sets["test"]:
        raise AssertionError("validation/test subject overlap")
    return {
        split: sorted(
            (stay for subject in split_subjects for stay in subject_stays[subject]),
            key=lambda stay: hashlib.sha256(f"{seed}|{split}|{stay}".encode()).hexdigest(),
        )
        for split, split_subjects in subject_sets.items()
    }


def split_overlap_counts(
    partitions: dict[str, list[Any]],
    stay_subjects: dict[Any, Any],
) -> dict[str, int]:
    groups = {
        split: {stay_subjects[stay] for stay in stays}
        for split, stays in partitions.items()
    }
    return {
        "train_validation": len(groups["train"] & groups["validation"]),
        "train_test": len(groups["train"] & groups["test"]),
        "validation_test": len(groups["validation"] & groups["test"]),
    }
