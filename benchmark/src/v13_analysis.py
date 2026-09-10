from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from v13_feature_registry import DYNAMIC_FEATURES, DYNAMIC_METADATA, STATIC_FEATURES
from v13_temporal_protocol import last_observed_elapsed_days


CANONICAL_METRIC_FIELDS = (
    "dataset",
    "direction",
    "population",
    "model",
    "seed",
    "scenario",
    "evaluation_unit",
    "denominator",
    "calibration_status",
    "seed_aggregation_rule",
    "auroc",
    "auprc",
    "brier",
    "ece",
    "calibration_intercept",
    "calibration_slope",
    "observed_risk",
    "mean_predicted_risk",
    "median_predicted_risk",
    "los_mae",
    "los_rmse",
)


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    total = len(labels)
    error = 0.0
    for index in range(bins):
        selected = indices == index
        if not selected.any():
            continue
        error += float(selected.mean()) * abs(
            float(labels[selected].mean()) - float(probabilities[selected].mean())
        )
    return float(error if total else np.nan)


def calibration_intercept_slope(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    if labels.size < 2 or np.unique(labels).size < 2:
        return float("nan"), float("nan")
    predictor = logit(probabilities).reshape(-1, 1)
    model = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000)
    model.fit(predictor, labels)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def canonical_metric_row(
    *,
    dataset: str,
    direction: str,
    population: str,
    model: str,
    seed: int,
    scenario: str,
    mortality_labels: np.ndarray,
    mortality_probabilities: np.ndarray,
    los_labels: np.ndarray,
    los_predictions: np.ndarray,
    calibration_status: str,
) -> dict[str, Any]:
    y = np.asarray(mortality_labels, dtype=np.int8)
    p = np.clip(np.asarray(mortality_probabilities, dtype=np.float64), 0.0, 1.0)
    y_los = np.asarray(los_labels, dtype=np.float64)
    p_los = np.asarray(los_predictions, dtype=np.float64)
    if not (len(y) == len(p) == len(y_los) == len(p_los)):
        raise ValueError("all canonical metrics inputs must be one aligned row per stay")
    if len(y) == 0:
        raise ValueError("canonical metrics cannot be computed on an empty population")
    if not np.isfinite(p).all() or not np.isfinite(p_los).all():
        raise ValueError("canonical metrics reject non-finite predictions")
    intercept, slope = calibration_intercept_slope(y, p)
    two_classes = np.unique(y).size == 2
    row = {
        "dataset": dataset,
        "direction": direction,
        "population": population,
        "model": model,
        "seed": int(seed),
        "scenario": scenario,
        "evaluation_unit": "one_prespecified_row_per_icu_stay_per_scenario",
        "denominator": int(len(y)),
        "calibration_status": calibration_status,
        "seed_aggregation_rule": "per_seed_before_mean_and_sample_sd",
        "auroc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "auprc": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, bins=10),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "observed_risk": float(y.mean()),
        "mean_predicted_risk": float(p.mean()),
        "median_predicted_risk": float(np.median(p)),
        "los_mae": float(mean_absolute_error(y_los, p_los)),
        "los_rmse": float(math.sqrt(mean_squared_error(y_los, p_los))),
    }
    if tuple(row) != CANONICAL_METRIC_FIELDS:
        raise AssertionError("canonical metric schema drift")
    return row


def clustered_paired_bootstrap(
    labels: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    replicates: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    labels = np.asarray(labels)
    predictions_a = np.asarray(predictions_a)
    predictions_b = np.asarray(predictions_b)
    groups = np.asarray(groups)
    if not (len(labels) == len(predictions_a) == len(predictions_b) == len(groups)):
        raise ValueError("clustered bootstrap inputs are not aligned")
    unique_groups, inverse = np.unique(groups.astype(str), return_inverse=True)
    cluster_rows = [np.flatnonzero(inverse == index) for index in range(len(unique_groups))]
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        draw = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
        rows = np.concatenate([cluster_rows[index] for index in draw])
        differences[replicate] = metric(labels[rows], predictions_a[rows]) - metric(
            labels[rows], predictions_b[rows]
        )
    observed = metric(labels, predictions_a) - metric(labels, predictions_b)
    return {
        "difference": float(observed),
        "ci_lower": float(np.quantile(differences, 0.025)),
        "ci_upper": float(np.quantile(differences, 0.975)),
        "replicates": int(replicates),
        "clusters": int(len(unique_groups)),
    }


def paired_sign_permutation(
    paired_differences: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    differences = np.asarray(paired_differences, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if len(differences) != len(groups):
        raise ValueError("paired permutation inputs are not aligned")
    unique_groups = np.unique(groups)
    cluster_differences = np.asarray([
        differences[groups == group].mean() for group in unique_groups
    ])
    observed = float(cluster_differences.mean())
    rng = np.random.default_rng(seed)
    permuted = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(cluster_differences))
        permuted[replicate] = float(np.mean(cluster_differences * signs))
    p_value = (1.0 + float(np.sum(np.abs(permuted) >= abs(observed)))) / (replicates + 1.0)
    return {
        "mean_cluster_difference": observed,
        "p_value_two_sided": float(p_value),
        "replicates": int(replicates),
        "clusters": int(len(unique_groups)),
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be a one-dimensional array in [0,1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def require_one_prediction_per_stay(stay_keys: np.ndarray) -> None:
    values = np.asarray(stay_keys)
    if len(np.unique(values)) != len(values):
        raise ValueError("DeLong and primary metrics require one prediction per stay")


def _scenario_rng(seed: int, scenario: str, row: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{scenario}|{row}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def temporal_stress_view(
    values: np.ndarray,
    mask: np.ndarray,
    tau: np.ndarray,
    elapsed: np.ndarray,
    *,
    scenario: str,
    seed: int,
    source_row_ids: np.ndarray | None = None,
    population_thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create exactly one deterministic stress-view sequence per input stay."""
    x = np.asarray(values, dtype=np.float32).copy()
    m = (np.asarray(mask) > 0).astype(np.uint8, copy=True)
    gaps = np.asarray(tau, dtype=np.float32).copy()
    times = np.asarray(elapsed, dtype=np.float32).copy()
    if x.shape != m.shape or x.shape[:2] != gaps.shape or gaps.shape != times.shape:
        raise ValueError("stress inputs violate aligned [N,T,F]/[N,T] contracts")
    if source_row_ids is None:
        row_ids = np.arange(len(x), dtype=np.int64)
    else:
        row_ids = np.asarray(source_row_ids, dtype=np.int64)
        if row_ids.shape != (len(x),):
            raise ValueError("source_row_ids must provide one stable id per input stay")
    thresholds = population_thresholds or {}
    included = np.ones(x.shape[0], dtype=bool)
    observed_rows = m.any(axis=2)

    if scenario == "full_6h":
        pass
    elif scenario.startswith("early_") and scenario.endswith("h"):
        token = scenario.removeprefix("early_").removesuffix("h").replace("_", ".")
        hours = float(token)
        m[times > hours * 3600.0] = 0
    elif scenario == "random_prefix":
        for row in range(len(x)):
            valid_times = np.flatnonzero(observed_rows[row])
            if len(valid_times):
                rng = _scenario_rng(seed, scenario, int(row_ids[row]))
                cutoff = int(rng.choice(valid_times))
                m[row, cutoff + 1 :, :] = 0
    elif scenario.startswith("random_window_") and scenario.endswith("h"):
        hours = float(scenario.removeprefix("random_window_").removesuffix("h"))
        for row in range(len(x)):
            rng = _scenario_rng(seed, scenario, int(row_ids[row]))
            start = float(rng.uniform(0.0, max(6.0 - hours, 0.0))) * 3600.0
            keep = (times[row] >= start) & (times[row] <= start + hours * 3600.0)
            m[row, ~keep, :] = 0
    elif scenario.startswith("event_budget_"):
        count = int(scenario.removeprefix("event_budget_"))
        for row in range(len(x)):
            # One event token is one observed (timestamp, variable) pair. A
            # timestamp row may contain several variables, so retaining N rows
            # would silently exceed the registered N-event budget.
            positions = np.argwhere(m[row] > 0)
            if len(positions) > count:
                drop = positions[:-count]
                m[row, drop[:, 0], drop[:, 1]] = 0
    elif scenario in {"sparse", "dense"}:
        keep_fraction = 0.25 if scenario == "sparse" else 0.75
        for row in range(len(x)):
            positions = np.argwhere(m[row] > 0)
            if len(positions) <= 1:
                continue
            rng = _scenario_rng(seed, scenario, int(row_ids[row]))
            keep_n = max(1, int(round(len(positions) * keep_fraction)))
            keep = rng.choice(len(positions), size=keep_n, replace=False)
            keep_mask = np.zeros(len(positions), dtype=bool)
            keep_mask[keep] = True
            drop = positions[~keep_mask]
            m[row, drop[:, 0], drop[:, 1]] = 0
    elif scenario in {"high_gap_variance", "low_gap_variance"}:
        for row in range(len(x)):
            valid_times = np.flatnonzero(observed_rows[row])
            if len(valid_times) <= 1:
                continue
            observed_gaps = gaps[row, valid_times].astype(np.float64)
            mean_gap = max(float(observed_gaps.mean()), 1.0)
            if scenario == "low_gap_variance":
                gaps[row, valid_times] = mean_gap
            else:
                factors = np.where(np.arange(len(valid_times)) % 2 == 0, 0.25, 1.75)
                perturbed = np.maximum(observed_gaps * factors, 1.0)
                perturbed *= observed_gaps.sum() / max(perturbed.sum(), 1.0)
                gaps[row, valid_times] = perturbed.astype(np.float32)
    elif scenario == "important_lab_loss":
        names = {
            "lactate", "bicarbonate", "chloride", "calcium", "magnesium",
            "platelet", "alt", "ast", "inr", "pt", "ptt", "procalcitonin",
        }
        m[:, :, [DYNAMIC_FEATURES.index(name) for name in names]] = 0
    elif scenario == "lab_panel_dropout":
        panels = (
            ("ph", "pao2", "paco2", "fio2"),
            ("sodium", "potassium", "chloride", "bicarbonate", "bun", "creatinine", "glucose"),
            ("wbc", "hemoglobin", "hematocrit", "platelet"),
            ("inr", "pt", "ptt"),
            ("bilirubin_total", "alt", "ast"),
        )
        for row in range(len(x)):
            rng = _scenario_rng(seed, scenario, int(row_ids[row]))
            panel = panels[int(rng.integers(0, len(panels)))]
            m[row, :, [DYNAMIC_FEATURES.index(name) for name in panel]] = 0
    elif scenario == "feature_dropout_30pct":
        n_drop = int(round(0.30 * len(DYNAMIC_FEATURES)))
        for row in range(len(x)):
            rng = _scenario_rng(seed, scenario, int(row_ids[row]))
            features = rng.choice(len(DYNAMIC_FEATURES), size=n_drop, replace=False)
            m[row, :, features] = 0
    elif scenario == "event_dropout_50pct":
        for row in range(len(x)):
            positions = np.argwhere(m[row] > 0)
            if len(positions):
                rng = _scenario_rng(seed, scenario, int(row_ids[row]))
                drop = rng.choice(len(positions), size=len(positions) // 2, replace=False)
                chosen = positions[drop]
                m[row, chosen[:, 0], chosen[:, 1]] = 0
    elif scenario == "delayed_update_2h":
        m[times > 4.0 * 3600.0] = 0
    elif scenario == "late_only_5_to_6h":
        keep = (times >= 5.0 * 3600.0) & (times <= 6.0 * 3600.0)
        m[~keep] = 0
    else:
        raise ValueError(f"unknown locked temporal stress scenario: {scenario}")

    x[m == 0] = np.nan
    final_rows = m.any(axis=2)
    anchor_elapsed_days = last_observed_elapsed_days(m, times)
    audit = {
        "scenario": scenario,
        "input_stays": int(len(x)),
        "included_stays": int(included.sum()),
        "one_row_per_included_stay": True,
        "observed_cells": int(m[included].sum()),
        "zero_event_included_stays": int((~final_rows[included].any(axis=1)).sum()),
        "anchor_elapsed_days_mean": float(anchor_elapsed_days[included].mean()),
        "anchor_elapsed_days_median": float(np.median(anchor_elapsed_days[included])),
    }
    return {
        "values": x[included],
        "mask": m[included],
        "tau": gaps[included],
        "elapsed": times[included],
        "anchor_elapsed_days": anchor_elapsed_days[included],
        "source_row_indices": np.flatnonzero(included),
    }, audit


def shap_feature_registry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_display = {
        "last": "Last",
        "mean": "Mean",
        "max": "Maximum",
        "min": "Minimum",
        "std": "Standard deviation of",
    }
    for summary in ("last", "mean", "max", "min", "std"):
        for feature in DYNAMIC_FEATURES:
            display, unit, family = DYNAMIC_METADATA[feature]
            rows.append({
                "machine_name": f"{summary}__{feature}",
                "display_name": f"{summary_display[summary]} {display.lower()}",
                "unit": unit,
                "feature_family": family,
                "summary_statistic": summary,
                "availability": "MIMIC-IV,eICU,CareVue under harmonized contract",
                "explanation_eligibility": "direct_clinical_feature",
                "component_scope": "Hybrid-plain,Flat-head,XGBoost comparators",
            })
    for feature in STATIC_FEATURES:
        rows.append({
            "machine_name": feature,
            "display_name": feature.replace("_", " ").title(),
            "unit": "years" if feature == "age" else "binary",
            "feature_family": (
                "admission-time demographic/context"
            ),
            "summary_statistic": "static",
            "availability": "source-specific mapped value or registered structural zero",
            "explanation_eligibility": "direct_clinical_feature",
            "component_scope": "Hybrid-plain,Flat-head,XGBoost comparators",
        })
    for feature in DYNAMIC_FEATURES:
        display, _, family = DYNAMIC_METADATA[feature]
        rows.append({
            "machine_name": f"observed_rate__{feature}",
            "display_name": f"Observedness rate of {display.lower()}",
            "unit": "proportion",
            "feature_family": f"observedness/{family}",
            "summary_statistic": "observedness_rate",
            "availability": "MIMIC-IV,eICU,CareVue under harmonized contract",
            "explanation_eligibility": "direct_clinical_feature",
            "component_scope": "Flat-head only",
        })
    if len(rows) != 252 or len({row["machine_name"] for row in rows}) != 252:
        raise AssertionError("SHAP clinical registry must resolve exactly 252 direct features")
    if any(row["machine_name"].startswith("f") and row["machine_name"][1:].isdigit() for row in rows):
        raise AssertionError("opaque fN labels are forbidden")
    return rows


def decision_curve_authorized(dataset: str, calibration_status: str) -> None:
    if dataset != "mimic_iv":
        raise PermissionError(
            "decision curves are registered only for MIMIC-IV; eICU zero-shot "
            "probabilities are not sufficiently calibrated"
        )
    if calibration_status != "source_validation_calibrated":
        raise PermissionError("MIMIC-IV decision curve requires source-validation calibration")
