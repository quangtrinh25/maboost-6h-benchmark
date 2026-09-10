from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_curve


ALLOWED_SOURCE_DATASETS = {"mimic_iv", "eicu"}


def _require_source_validation(source_dataset: str, split: str) -> None:
    if source_dataset not in ALLOWED_SOURCE_DATASETS:
        raise PermissionError("selection/calibration source dataset is not registered")
    if split != "validation":
        raise PermissionError("fusion, calibration, and thresholds require source validation")


def select_two_component_fusion(
    component_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    source_dataset: str,
    split: str,
    grid_points: int = 1001,
) -> dict[str, Any]:
    _require_source_validation(source_dataset, split)
    predictions = np.asarray(component_probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if predictions.ndim != 2 or predictions.shape[1] != 2 or len(predictions) != len(labels):
        raise ValueError("fusion selection requires aligned [N,2] source predictions")
    if np.unique(labels).size != 2:
        raise ValueError("fusion selection requires both mortality classes")
    best_score = -np.inf
    best_weight = 0.0
    for weight in np.linspace(0.0, 1.0, grid_points):
        fused = weight * predictions[:, 0] + (1.0 - weight) * predictions[:, 1]
        score = float(average_precision_score(labels, fused))
        if score > best_score:
            best_score = score
            best_weight = float(weight)
    return {
        "source_dataset": source_dataset,
        "selection_split": "validation",
        "selection_metric": "AUPRC",
        "component_0_weight": best_weight,
        "component_1_weight": 1.0 - best_weight,
        "validation_auprc": best_score,
        "grid_points": int(grid_points),
        "target_information_used": False,
    }


def select_three_component_fusion(
    component_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    source_dataset: str,
    split: str,
    simplex_step: float = 0.05,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Select the frozen-v11 three-head fusion on source validation only."""
    _require_source_validation(source_dataset, split)
    predictions = np.asarray(component_probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if predictions.ndim != 2 or predictions.shape[1] != 3 or len(predictions) != len(labels):
        raise ValueError("tri-fusion selection requires aligned [N,3] source predictions")
    if np.unique(labels).size != 2:
        raise ValueError("tri-fusion selection requires both mortality classes")
    if not 0.0 < simplex_step <= 1.0:
        raise ValueError("simplex_step must be in (0,1]")
    resolution = int(round(1.0 / simplex_step))
    if not np.isclose(resolution * simplex_step, 1.0):
        raise ValueError("simplex_step must evenly divide one")
    best_score = -np.inf
    best = (0.0, 0.0, 1.0)
    candidates = 0
    for first in range(resolution + 1):
        for second in range(resolution - first + 1):
            third = resolution - first - second
            weights = np.asarray([first, second, third], dtype=np.float64) / resolution
            fused = predictions @ weights
            score = float(average_precision_score(
                labels, fused, sample_weight=sample_weight
            ))
            candidates += 1
            if score > best_score:
                best_score = score
                best = tuple(float(value) for value in weights)
    return {
        "source_dataset": source_dataset,
        "selection_split": "validation",
        "selection_metric": "AUPRC",
        "components": ["Hybrid-plain", "XGBOOST-FLAT", "Flat-head"],
        "weights": list(best),
        "validation_auprc": best_score,
        "simplex_step": float(simplex_step),
        "candidate_count": int(candidates),
        "target_information_used": False,
        "sample_weighted": sample_weight is not None,
    }


@dataclass(frozen=True)
class IsotonicCalibration:
    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]
    source_dataset: str

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        return np.interp(
            np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0),
            np.asarray(self.x_thresholds),
            np.asarray(self.y_thresholds),
            left=self.y_thresholds[0],
            right=self.y_thresholds[-1],
        ).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "isotonic",
            "source_dataset": self.source_dataset,
            "selection_split": "validation",
            "x_thresholds": list(self.x_thresholds),
            "y_thresholds": list(self.y_thresholds),
            "target_information_used": False,
        }


def fit_source_isotonic_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    source_dataset: str,
    split: str,
    sample_weight: np.ndarray | None = None,
) -> IsotonicCalibration:
    _require_source_validation(source_dataset, split)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if len(probabilities) != len(labels) or np.unique(labels).size != 2:
        raise ValueError("calibration requires aligned probabilities and both source classes")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probabilities, labels, sample_weight=sample_weight)
    return IsotonicCalibration(
        x_thresholds=tuple(float(value) for value in calibrator.X_thresholds_),
        y_thresholds=tuple(float(value) for value in calibrator.y_thresholds_),
        source_dataset=source_dataset,
    )


def select_source_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    source_dataset: str,
    split: str,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    _require_source_validation(source_dataset, split)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if len(probabilities) != len(labels) or np.unique(labels).size != 2:
        raise ValueError("threshold selection requires both source classes")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, probabilities, sample_weight=sample_weight
    )
    index = int(np.argmax(true_positive_rate - false_positive_rate))
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = 1.0
    return {
        "source_dataset": source_dataset,
        "selection_split": "validation",
        "criterion": "youden_j",
        "threshold": threshold,
        "target_information_used": False,
        "sample_weighted": sample_weight is not None,
    }
