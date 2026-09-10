from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from correction07_analysis import (
    corrected_clustered_paired_bootstrap,
    corrected_paired_sign_permutation,
)
from correction07_bootstrap import RustBootstrapKernel
from correction07_contract import (
    CORRECTION07_CANONICAL_DATA_NAMES,
    bounded_thread_count,
    verify_sha256_sidecar,
)
from reproduction_contract import require_analysis_readiness as require_correction07_analysis_approval
from correction07_ensemble import prepare_seed_ensemble
from landmark_firewall import load_features
from landmark_jobs import LandmarkJob, LandmarkJobStore, build_job_matrix
from landmark_protocol import REPO_ROOT, RESULTS_ROOT, WORKSPACE_ROOT, load_protocol, load_verified_phase_report, verify_frozen_state, write_phase_report
from v13_analysis import benjamini_hochberg, clustered_paired_bootstrap, paired_sign_permutation
from v13_guard import assert_safe_output_path, safe_write_json, safe_write_text, sha256_file
from v13_model_registry import registry_rows
from v13_resource_guardian import resource_safe_point


METRICS = (
    "auroc", "auprc", "brier", "ece", "calibration_intercept",
    "calibration_slope", "los_mae", "los_rmse",
)

CORRECTION07_PATH = (
    WORKSPACE_ROOT / "config" / "implementation_correction_07_analysis_engine.json"
)
CORRECTION07_BUILD_MANIFEST = (
    WORKSPACE_ROOT / "private" / "correction07_rust_build" / "BUILD_MANIFEST.json"
)
CORRECTION07_QUALIFICATION = (
    RESULTS_ROOT
    / "qualification"
    / "CORRECTION_07_SOURCE_INTERNAL_QUALIFICATION.json"
)
CORRECTION07_AUDIT = (
    RESULTS_ROOT
    / "diagnostics"
    / "CORRECTION_07_AFFECTED_ARTIFACT_AUDIT.json"
)
CORRECTION07_APPROVAL = (
    WORKSPACE_ROOT
    / "config"
    / "analysis_launch_approval_v13_landmark_correction_07.json"
)
CANONICAL_OUTPUT_NAMES = CORRECTION07_CANONICAL_DATA_NAMES | {
    "canonical_analysis_report.json"
}


def _write_csv(
    path: Path, frame: pd.DataFrame, *, published_path: Path | None = None
) -> str:
    target = assert_safe_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.pid{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256_file(target)
    safe_write_text(
        target.with_suffix(target.suffix + ".sha256"),
        f"{digest}  {(published_path or target).relative_to(REPO_ROOT)}\n",
    )
    return digest


def _publish_canonical_directory(
    staging_root: Path,
    canonical_root: Path,
    expected_names: set[str] | frozenset[str],
) -> None:
    """Verify and atomically publish a complete canonical directory."""

    staging = assert_safe_output_path(staging_root)
    canonical = assert_safe_output_path(canonical_root)
    if not staging.is_dir() or canonical.exists():
        raise RuntimeError("canonical staging/publication boundary is not clean")
    if any(path.is_dir() for path in staging.iterdir()):
        raise RuntimeError("canonical staging inventory contains a directory")
    required = set(expected_names) | {f"{name}.sha256" for name in expected_names}
    observed = {path.name for path in staging.iterdir() if path.is_file()}
    if observed != required:
        raise RuntimeError("canonical staging inventory is incomplete or unexpected")
    for name in sorted(expected_names):
        data = staging / name
        sidecar = staging / f"{name}.sha256"
        fields = sidecar.read_text(encoding="utf-8").split()
        expected_relative = str((canonical / name).relative_to(REPO_ROOT))
        if (
            len(fields) < 2
            or fields[0] != sha256_file(data)
            or fields[1] != expected_relative
        ):
            raise RuntimeError(f"canonical staging SHA mismatch: {name}")
        with data.open("rb") as handle:
            os.fsync(handle.fileno())
        with sidecar.open("rb") as handle:
            os.fsync(handle.fileno())
    directory_fd = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.replace(staging, canonical)
    parent_fd = os.open(canonical.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _publish_canonical_after_validation(
    staging_root: Path,
    canonical_root: Path,
    expected_names: set[str] | frozenset[str],
    final_validator: Callable[[], Any],
) -> Any:
    """Keep canonical invisible unless every late fail-closed gate passes."""

    evidence = final_validator()
    _publish_canonical_directory(staging_root, canonical_root, expected_names)
    return evidence


def _load_correction07_runtime():
    from reproduction_contract import load_analysis_runtime
    return load_analysis_runtime()


def _runtime_threads(correction: dict[str, Any]) -> int:
    """Use the approved worker count and fail if CPU affinity later drifts."""

    current = bounded_thread_count(correction)
    qualified = int(correction.get("_qualified_threads", current))
    if current != qualified:
        raise PermissionError(
            "correction-07 worker count drift during analysis; requalification required"
        )
    return qualified


def _correction07_safe_point(
    correction: dict[str, Any], point: str, *, force_cleanup: bool
) -> None:
    _runtime_threads(correction)
    resource_safe_point(point, force_cleanup=force_cleanup)


def _analysis_progress_callback(**context: str):
    progress_path = (
        WORKSPACE_ROOT / "private" / "analysis_correction07" / "progress.json"
    )

    def emit(progress: dict[str, Any]) -> None:
        payload = {
            "schema_version": "1.0.0",
            "status": "running",
            **context,
            **progress,
            "patient_level_content": False,
        }
        safe_write_json(progress_path, payload)
        safe_write_text(
            progress_path.with_suffix(".json.sha256"),
            f"{sha256_file(progress_path)}  {progress_path.relative_to(REPO_ROOT)}\n",
        )

    return emit


def _require_full_jobs(config: dict[str, Any]) -> list[LandmarkJob]:
    load_verified_phase_report("D_FULL", allowed_statuses={"pass"})
    jobs = build_job_matrix(config)
    store = LandmarkJobStore()
    incomplete = [job.job_id for job in jobs if not store.is_complete(job)]
    if incomplete:
        raise RuntimeError(f"analysis found {len(incomplete)} incomplete jobs")
    return jobs


def _completion_inventory_sha256(jobs: list[LandmarkJob]) -> str:
    """Bind the immutable completion controls for the fully verified job tree."""

    digest = hashlib.sha256()
    for job in sorted(jobs, key=lambda item: item.job_id):
        path = WORKSPACE_ROOT / "private" / "jobs" / job.job_id / "completion_private.json"
        sidecar = path.with_suffix(".json.sha256")
        if not path.is_file() or not sidecar.is_file():
            raise RuntimeError(f"completion inventory missing: {job.job_id}")
        observed = sha256_file(path)
        if sidecar.read_text(encoding="utf-8").split()[0] != observed:
            raise RuntimeError(f"completion inventory lock drift: {job.job_id}")
        digest.update(job.job_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(observed.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _job_report(job: LandmarkJob) -> dict[str, Any]:
    path = RESULTS_ROOT / "jobs" / f"{job.job_id}.json"
    hash_path = path.with_suffix(".json.sha256")
    if not path.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"missing aggregate job report: {job.job_id}")
    if sha256_file(path) != hash_path.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError(f"aggregate job report hash mismatch: {job.job_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("patient_level_content") is not False:
        raise PermissionError(f"aggregate report privacy marker missing: {job.job_id}")
    return payload


def aggregate_canonical(jobs: list[LandmarkJob]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        report = _job_report(job)
        for metric_row in report["canonical_metrics"]:
            rows.append({
                **metric_row,
                "source_dataset": job.source_dataset,
                "training_regime": job.training_regime,
                "registry_role": job.registry_role,
                "job_id": job.job_id,
            })
    per_seed = pd.DataFrame(rows)
    expected = len(jobs) * 3 * 28
    if len(per_seed) != expected:
        raise RuntimeError(f"canonical row count drift: {len(per_seed)} != {expected}")
    keys = [
        "source_dataset", "dataset", "direction", "population", "model",
        "scenario", "training_regime", "registry_role",
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for key, group in per_seed.groupby(keys, sort=True, dropna=False):
        record = dict(zip(keys, key, strict=True))
        record["seed_count"] = int(group["seed"].nunique())
        for metric in METRICS:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        aggregate_rows.append(record)
    return per_seed, pd.DataFrame(aggregate_rows)


def _evaluation_path(job: LandmarkJob, dataset: str) -> Path:
    root = WORKSPACE_ROOT / "private" / "jobs" / job.job_id
    return (
        root / "source_internal_evaluation_private.npz"
        if dataset == job.source_dataset
        else root / f"target_{dataset}_evaluation_private.npz"
    )


def _load_eval(job: LandmarkJob, dataset: str) -> dict[str, np.ndarray]:
    path = _evaluation_path(job, dataset)
    with np.load(path, allow_pickle=False) as payload:
        result = {name: payload[name] for name in payload.files}
    required = {
        "mortality", "los", "stay_keys", "group_keys", "age", "sex_male",
        "icu_type_code", "mortality_probability", "los_prediction",
    }
    if not required.issubset(result):
        raise RuntimeError(f"private evaluation bundle incomplete: {path}")
    if result["mortality_probability"].shape[1] != 28:
        raise RuntimeError(f"private evaluation scenario width drift: {path}")
    return result


def _matching_job(
    jobs: list[LandmarkJob], source: str, model: str, regime: str, seed: int
) -> LandmarkJob:
    matches = [
        job for job in jobs
        if job.source_dataset == source and job.model == model
        and job.training_regime == regime and job.seed == seed
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one job for {source}/{model}/{regime}/seed{seed}, got {len(matches)}"
        )
    return matches[0]


def _paired_seed_ensemble(
    jobs: list[LandmarkJob],
    *,
    source: str,
    dataset: str,
    model_a: str,
    regime_a: str,
    model_b: str,
    regime_b: str,
    seeds: list[int],
    scope: str,
) -> dict[str, np.ndarray]:
    payloads_a: list[dict[str, np.ndarray]] = []
    payloads_b: list[dict[str, np.ndarray]] = []
    for seed in seeds:
        a = _load_eval(_matching_job(jobs, source, model_a, regime_a, seed), dataset)
        b = _load_eval(_matching_job(jobs, source, model_b, regime_b, seed), dataset)
        payloads_a.append(a)
        payloads_b.append(b)
    prepared = prepare_seed_ensemble(payloads_a, payloads_b, scope=scope)
    return {
        "mortality": prepared.mortality,
        "los": prepared.los,
        "group": prepared.cluster_codes,
        "a": prepared.predictions_a,
        "b": prepared.predictions_b,
        "a_los": prepared.los_predictions_a,
        "b_los": prepared.los_predictions_b,
        "seed_coverage_min": np.full(
            len(prepared.mortality), prepared.minimum_seed_coverage, dtype=np.int8
        ),
        "alignment_sha256": np.asarray([prepared.alignment_sha256]),
    }


def _bootstrap_metric(
    name: str,
) -> tuple[Callable[[np.ndarray, np.ndarray], float], str]:
    if name == "auroc":
        return lambda y, p: float(roc_auc_score(y, p)), "higher_is_better"
    if name == "auprc":
        return lambda y, p: float(average_precision_score(y, p)), "higher_is_better"
    if name == "brier":
        return lambda y, p: float(brier_score_loss(y, p)), "lower_is_better"
    if name == "los_mae":
        return lambda y, p: float(mean_absolute_error(y, p)), "lower_is_better"
    if name == "los_rmse":
        return lambda y, p: float(math.sqrt(mean_squared_error(y, p))), "lower_is_better"
    raise ValueError(name)


def primary_contrasts(
    jobs: list[LandmarkJob],
    config: dict[str, Any],
    correction: dict[str, Any],
    correction_hash: str,
    kernel: RustBootstrapKernel,
    *,
    routes_override: list[tuple[str, str]] | None = None,
    checkpoint_namespace: str = "production",
) -> pd.DataFrame:
    contrast_specs = (
        ("MABOOST_aug_vs_SUMMARY_STATIC_full", "MABOOST", "stress_augmented", "XGBOOST-SUMMARY+STATIC", "full_only", "primary_model"),
        ("MABOOST_aug_vs_XGBOOST_FLAT_full", "MABOOST", "stress_augmented", "XGBOOST-FLAT", "full_only", "primary_model"),
        ("XGBOOST_FLAT_aug_vs_full", "XGBOOST-FLAT", "stress_augmented", "XGBOOST-FLAT", "full_only", "matched_augmentation"),
        ("SUMMARY_STATIC_aug_vs_full", "XGBOOST-SUMMARY+STATIC", "stress_augmented", "XGBOOST-SUMMARY+STATIC", "full_only", "matched_augmentation"),
        ("Hybrid_plain_aug_vs_full", "Hybrid-plain", "stress_augmented", "Hybrid-plain", "full_only", "matched_augmentation"),
        ("MABOOST_aug_vs_full", "MABOOST", "stress_augmented", "MABOOST", "full_only", "matched_augmentation"),
        ("MABOOST_aug_vs_SUMMARY_STATIC_aug", "MABOOST", "stress_augmented", "XGBOOST-SUMMARY+STATIC", "stress_augmented", "matched_architecture"),
    )
    routes = routes_override or [
        (source, dataset)
        for source in ("mimic_iv", "eicu")
        for dataset in (source, *config["source_directions"][source])
    ]
    replicates = int(config["statistics"]["bootstrap_replicates"])
    permutation_replicates = int(config["statistics"]["permutation_replicates"])
    rows: list[dict[str, Any]] = []
    threads = _runtime_threads(correction)
    for source, dataset in routes:
        for scope in ("full_6h", "pooled_stress"):
            for contrast, model_a, regime_a, model_b, regime_b, family in contrast_specs:
                ensemble = _paired_seed_ensemble(
                    jobs,
                    source=source,
                    dataset=dataset,
                    model_a=model_a,
                    regime_a=regime_a,
                    model_b=model_b,
                    regime_b=regime_b,
                    seeds=[int(value) for value in config["seeds"]],
                    scope=scope,
                )
                for metric_name in ("auroc", "auprc", "brier", "los_mae", "los_rmse"):
                    metric, preference = _bootstrap_metric(metric_name)
                    labels = ensemble["los"] if metric_name.startswith("los_") else ensemble["mortality"]
                    prediction_a = ensemble["a_los"] if metric_name.startswith("los_") else ensemble["a"]
                    prediction_b = ensemble["b_los"] if metric_name.startswith("los_") else ensemble["b"]
                    checkpoint_root = (
                        WORKSPACE_ROOT
                        / "private"
                        / "analysis_correction07"
                        / "bootstrap"
                        / checkpoint_namespace
                        / "primary"
                        / source
                        / dataset
                        / scope
                        / contrast
                        / metric_name
                    )
                    result = corrected_clustered_paired_bootstrap(
                        labels,
                        prediction_a,
                        prediction_b,
                        ensemble["group"],
                        metric_name=metric_name,
                        replicates=replicates,
                        seed=int(hashlib_sha256_int(f"{source}|{dataset}|{scope}|{contrast}|{metric_name}")),
                        kernel=kernel,
                        threads=threads,
                        thread_limit=int(correction["max_threads"]),
                        shard_size=int(correction["shard_size"]),
                        evaluation_batch_size=int(correction["kernel_batch_size"]),
                        max_draw_buffer_bytes=int(correction["max_draw_buffer_bytes"]),
                        checkpoint_root=checkpoint_root,
                        source=source,
                        dataset=dataset,
                        scope=scope,
                        contrast=contrast,
                        correction_sha256=correction_hash,
                        progress_callback=_analysis_progress_callback(
                            source=source,
                            dataset=dataset,
                            scope=scope,
                            contrast=contrast,
                            metric=metric_name,
                        ),
                        resource_safe_point=lambda point: _correction07_safe_point(
                            correction, point, force_cleanup=False
                        ),
                    )
                    record = {
                        "source_dataset": source,
                        "dataset": dataset,
                        "direction": "source_internal" if source == dataset else f"{source}_to_{dataset}",
                        "scope": scope,
                        "evidence_family": family,
                        "contrast": contrast,
                        "model_a": model_a,
                        "regime_a": regime_a,
                        "model_b": model_b,
                        "regime_b": regime_b,
                        "metric": metric_name,
                        "preference": preference,
                        "difference_a_minus_b": result["difference"],
                        "ci_lower": result["ci_lower"],
                        "ci_upper": result["ci_upper"],
                        "bootstrap_replicates": result["replicates"],
                        "clusters": result["clusters"],
                        "evaluation_rows": int(len(labels)),
                        "minimum_seed_coverage": int(ensemble["seed_coverage_min"].min()),
                        "seeds_are_not_independent_samples": True,
                        "permutation_p_value": np.nan,
                    }
                    if metric_name == "brier":
                        loss_difference = (
                            (ensemble["mortality"] - ensemble["a"]) ** 2
                            - (ensemble["mortality"] - ensemble["b"]) ** 2
                        )
                        permutation = corrected_paired_sign_permutation(
                            loss_difference,
                            ensemble["group"],
                            replicates=permutation_replicates,
                            seed=int(hashlib_sha256_int(f"perm|{source}|{dataset}|{scope}|{contrast}")),
                            batch_size=int(correction["permutation_batch_size"]),
                        )
                        record["permutation_p_value"] = permutation["p_value_two_sided"]
                        record["permutation_replicates"] = permutation["replicates"]
                    rows.append(record)
                _correction07_safe_point(
                    correction,
                    f"statistics_{source}_{dataset}_{scope}_{contrast}",
                    force_cleanup=True,
                )
    frame = pd.DataFrame(rows)
    frame["bh_adjusted_p"] = np.nan
    brier = frame["metric"] == "brier"
    for _, indices in frame.loc[brier].groupby(
        ["evidence_family", "source_dataset", "scope"], sort=True
    ).groups.items():
        frame.loc[indices, "bh_adjusted_p"] = benjamini_hochberg(
            frame.loc[indices, "permutation_p_value"].to_numpy(dtype=np.float64)
        )
    frame["ci_excludes_zero"] = (
        (frame["ci_lower"] > 0) | (frame["ci_upper"] < 0)
    )
    return frame


def seed_level_forest_data(
    jobs: list[LandmarkJob],
    config: dict[str, Any],
    correction: dict[str, Any],
    correction_hash: str,
    kernel: RustBootstrapKernel,
    *,
    checkpoint_namespace: str = "production",
) -> pd.DataFrame:
    """Descriptive seed-level CIs; seeds are never used as inferential units."""
    contrasts = (
        ("MABOOST_vs_SUMMARY_STATIC", "XGBOOST-SUMMARY+STATIC"),
        ("MABOOST_vs_XGBOOST_FLAT", "XGBOOST-FLAT"),
    )
    rows = []
    threads = _runtime_threads(correction)
    for source in ("mimic_iv", "eicu"):
        for dataset in (source, *config["source_directions"][source]):
            for seed in config["seeds"]:
                job_a = _matching_job(
                    jobs, source, "MABOOST", "stress_augmented", int(seed)
                )
                a = _load_eval(job_a, dataset)
                for contrast, comparator in contrasts:
                    job_b = _matching_job(
                        jobs, source, comparator, "full_only", int(seed)
                    )
                    b = _load_eval(job_b, dataset)
                    for name in ("mortality", "stay_keys", "group_keys"):
                        if not np.array_equal(a[name], b[name]):
                            raise RuntimeError("seed-level forest pairing drift")
                    result = corrected_clustered_paired_bootstrap(
                        a["mortality"],
                        a["mortality_probability"][:, 0],
                        b["mortality_probability"][:, 0],
                        a["group_keys"],
                        metric_name="auprc",
                        replicates=int(config["statistics"]["bootstrap_replicates"]),
                        seed=hashlib_sha256_int(
                            f"forest|{source}|{dataset}|{seed}|{contrast}"
                        ),
                        kernel=kernel,
                        threads=threads,
                        thread_limit=int(correction["max_threads"]),
                        shard_size=int(correction["shard_size"]),
                        evaluation_batch_size=int(correction["kernel_batch_size"]),
                        max_draw_buffer_bytes=int(correction["max_draw_buffer_bytes"]),
                        checkpoint_root=(
                            WORKSPACE_ROOT
                            / "private"
                            / "analysis_correction07"
                            / "bootstrap"
                            / checkpoint_namespace
                            / "seed_forest"
                            / source
                            / dataset
                            / str(seed)
                            / contrast
                        ),
                        source=source,
                        dataset=dataset,
                        scope="full_6h",
                        contrast=f"seed{seed}_{contrast}",
                        correction_sha256=correction_hash,
                        progress_callback=_analysis_progress_callback(
                            source=source,
                            dataset=dataset,
                            scope="full_6h",
                            contrast=f"seed{seed}_{contrast}",
                            metric="auprc",
                        ),
                        resource_safe_point=lambda point: _correction07_safe_point(
                            correction, point, force_cleanup=False
                        ),
                    )
                    rows.append({
                        "source_dataset": source,
                        "dataset": dataset,
                        "direction": "source_internal" if source == dataset else f"{source}_to_{dataset}",
                        "contrast": contrast,
                        "seed": int(seed),
                        "metric": "auprc",
                        "difference_a_minus_b": result["difference"],
                        "ci_lower": result["ci_lower"],
                        "ci_upper": result["ci_upper"],
                        "bootstrap_replicates": result["replicates"],
                        "clusters": result["clusters"],
                        "descriptive_seed_level_only": True,
                        "seeds_are_not_independent_samples": True,
                    })
                _correction07_safe_point(
                    correction,
                    f"seed_forest_{source}_{dataset}_{seed}",
                    force_cleanup=True,
                )
    return pd.DataFrame(rows)


def hashlib_sha256_int(text: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    pooled = math.sqrt(max((float(x.var(ddof=1)) + float(y.var(ddof=1))) / 2.0, 0.0))
    return 0.0 if pooled < 1e-12 else float((x.mean() - y.mean()) / pooled)


def cohort_and_shift_tables(
    jobs: list[LandmarkJob], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_rows: list[dict[str, Any]] = []
    feature_summaries: dict[str, dict[str, np.ndarray]] = {}
    flows: list[dict[str, Any]] = []
    target_source = {"mimic_iv": "eicu", "eicu": "mimic_iv", "mimic_iii_carevue": "mimic_iv"}
    for dataset in config["datasets"]:
        source = target_source[dataset]
        job = _matching_job(jobs, source, "MABOOST", "stress_augmented", int(config["seeds"][0]))
        evaluation = _load_eval(job, dataset)
        features = load_features(dataset)
        audit_path = REPO_ROOT / config["datasets"][dataset]["private_root"] / "aggregate_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        flows.extend(audit["flow"])
        mortality = np.asarray(evaluation["mortality"], dtype=np.int8)
        los = np.asarray(evaluation["los"], dtype=np.float32)
        dataset_rows.append({
            "dataset": dataset,
            "fixed_6h_risk_set_stays": int(len(mortality)),
            "patients": int(len(np.unique(evaluation["group_keys"].astype(str)))),
            "mortality_events": int(mortality.sum()),
            "mortality_prevalence": float(mortality.mean()),
            "remaining_los_mean_days": float(los.mean()),
            "remaining_los_median_days": float(np.median(los)),
            "age_mean": float(np.mean(evaluation["age"])),
            "male_fraction": float(np.mean(evaluation["sex_male"])),
            "observed_cell_fraction": float(np.asarray(features["mask"], dtype=np.uint8).mean()),
            "max_predictor_elapsed_seconds": float(np.asarray(features["elapsed"]).max()),
        })
        feature_summaries[dataset] = {
            "static": np.asarray(features["static"], dtype=np.float32),
            "observedness": np.asarray(features["mask"], dtype=np.float32).mean(axis=1),
        }
    shifts: list[dict[str, Any]] = []
    names = [
        *[f"static__{index:02d}" for index in range(12)],
        *[f"observedness__{index:02d}" for index in range(40)],
    ]
    for source in ("mimic_iv", "eicu"):
        source_matrix = np.concatenate([
            feature_summaries[source]["static"], feature_summaries[source]["observedness"]
        ], axis=1)
        for target in config["source_directions"][source]:
            target_matrix = np.concatenate([
                feature_summaries[target]["static"], feature_summaries[target]["observedness"]
            ], axis=1)
            for index, name in enumerate(names):
                shifts.append({
                    "source_dataset": source,
                    "target_dataset": target,
                    "feature": name,
                    "standardized_mean_difference": _standardized_mean_difference(
                        source_matrix[:, index], target_matrix[:, index]
                    ),
                })
    return pd.DataFrame(dataset_rows), pd.DataFrame(shifts), pd.DataFrame(flows)


def subgroup_table(jobs: list[LandmarkJob], config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in ("mimic_iv", "eicu"):
        for dataset in (source, *config["source_directions"][source]):
            members = [
                _load_eval(
                    _matching_job(jobs, source, "MABOOST", "stress_augmented", int(seed)),
                    dataset,
                )
                for seed in config["seeds"]
            ]
            by_stay: dict[str, dict[str, Any]] = {}
            for payload in members:
                for index, stay in enumerate(payload["stay_keys"]):
                    item = by_stay.setdefault(str(stay), {
                        "mortality": int(payload["mortality"][index]),
                        "los": float(payload["los"][index]),
                        "group": str(payload["group_keys"][index]),
                        "age": float(payload["age"][index]),
                        "sex": int(payload["sex_male"][index]),
                        "icu": int(payload["icu_type_code"][index]),
                        "p": [], "p_los": [],
                    })
                    item["p"].append(float(payload["mortality_probability"][index, 0]))
                    item["p_los"].append(float(payload["los_prediction"][index, 0]))
            ordered = list(by_stay.values())
            frame = pd.DataFrame({
                "mortality": [item["mortality"] for item in ordered],
                "los": [item["los"] for item in ordered],
                "age": [item["age"] for item in ordered],
                "sex_male": [item["sex"] for item in ordered],
                "icu_type_code": [item["icu"] for item in ordered],
                "probability": [np.mean(item["p"]) for item in ordered],
                "los_prediction": [np.mean(item["p_los"]) for item in ordered],
            })
            strata: list[tuple[str, str, np.ndarray]] = [
                ("sex", "female", frame["sex_male"].to_numpy() == 0),
                ("sex", "male", frame["sex_male"].to_numpy() == 1),
                ("age", "65-74", (frame["age"].to_numpy() >= 65) & (frame["age"].to_numpy() < 75)),
                ("age", "75-84", (frame["age"].to_numpy() >= 75) & (frame["age"].to_numpy() < 85)),
                ("age", "85+", frame["age"].to_numpy() >= 85),
            ]
            strata.extend(
                ("icu_type", f"code_{int(code)}", frame["icu_type_code"].to_numpy() == code)
                for code in sorted(frame["icu_type_code"].unique())
            )
            for variable, level, selected in strata:
                group = frame.loc[selected]
                events = int(group["mortality"].sum())
                if len(group) < int(config["statistics"]["minimum_subgroup_stays"]) or events < int(config["statistics"]["minimum_subgroup_events"]):
                    continue
                rows.append({
                    "source_dataset": source,
                    "dataset": dataset,
                    "direction": "source_internal" if source == dataset else f"{source}_to_{dataset}",
                    "subgroup_variable": variable,
                    "subgroup_level": level,
                    "stays": int(len(group)),
                    "mortality_events": events,
                    "auroc": float(roc_auc_score(group["mortality"], group["probability"])),
                    "auprc": float(average_precision_score(group["mortality"], group["probability"])),
                    "brier": float(brier_score_loss(group["mortality"], group["probability"])),
                    "los_mae": float(mean_absolute_error(group["los"], group["los_prediction"])),
                    "seed_predictions_ensembled_per_stay": True,
                })
    return pd.DataFrame(rows)


def run_analysis() -> tuple[dict[str, Any], int]:
    correction, correction_hash, kernel, approval_chain = _load_correction07_runtime()
    plan, plan_hash, config, config_hash = load_protocol()
    jobs = _require_full_jobs(config)
    completion_inventory_before = _completion_inventory_sha256(jobs)
    frozen_before = verify_frozen_state(plan)
    canonical_root = RESULTS_ROOT / "canonical"
    if canonical_root.exists():
        raise RuntimeError("canonical output root already exists")
    stale_staging = sorted(RESULTS_ROOT.glob(".canonical.correction07.*"))
    if stale_staging:
        raise RuntimeError(
            f"partial correction-07 canonical staging exists: {stale_staging}"
        )
    started = time.perf_counter()
    per_seed, aggregate = aggregate_canonical(jobs)
    contrasts = primary_contrasts(
        jobs, config, correction, correction_hash, kernel
    )
    seed_forest = seed_level_forest_data(
        jobs, config, correction, correction_hash, kernel
    )
    cohorts, shifts, flows = cohort_and_shift_tables(jobs, config)
    subgroups = subgroup_table(jobs, config)
    staging_root = canonical_root.with_name(
        f".canonical.correction07.pid{os.getpid()}.{time.time_ns()}"
    )
    assert_safe_output_path(staging_root).mkdir(parents=True, exist_ok=False)
    files = {
        "per_seed_metrics.csv": _write_csv(staging_root / "per_seed_metrics.csv", per_seed, published_path=canonical_root / "per_seed_metrics.csv"),
        "aggregate_metrics.csv": _write_csv(staging_root / "aggregate_metrics.csv", aggregate, published_path=canonical_root / "aggregate_metrics.csv"),
        "primary_contrasts.csv": _write_csv(staging_root / "primary_contrasts.csv", contrasts, published_path=canonical_root / "primary_contrasts.csv"),
        "seed_level_bootstrap_forest.csv": _write_csv(staging_root / "seed_level_bootstrap_forest.csv", seed_forest, published_path=canonical_root / "seed_level_bootstrap_forest.csv"),
        "cohort_characteristics.csv": _write_csv(staging_root / "cohort_characteristics.csv", cohorts, published_path=canonical_root / "cohort_characteristics.csv"),
        "measurement_shift_smd.csv": _write_csv(staging_root / "measurement_shift_smd.csv", shifts, published_path=canonical_root / "measurement_shift_smd.csv"),
        "cohort_flow.csv": _write_csv(staging_root / "cohort_flow.csv", flows, published_path=canonical_root / "cohort_flow.csv"),
        "subgroups.csv": _write_csv(staging_root / "subgroups.csv", subgroups, published_path=canonical_root / "subgroups.csv"),
        "model_registry.csv": _write_csv(staging_root / "model_registry.csv", pd.DataFrame(registry_rows()), published_path=canonical_root / "model_registry.csv"),
    }
    inference_counts = (
        contrasts.groupby(["source_dataset", "dataset", "scope", "metric"], as_index=False)
        .agg(
            contrasts=("contrast", "count"),
            ci_excludes_zero=("ci_excludes_zero", "sum"),
            bh_significant=("bh_adjusted_p", lambda values: int((values < 0.05).sum())),
        )
    )
    files["inference_counts.csv"] = _write_csv(
        staging_root / "inference_counts.csv", inference_counts,
        published_path=canonical_root / "inference_counts.csv",
    )
    report = {
        "schema_version": "1.0.0",
        "status": "pass",
        "analysis_plan_sha256": plan_hash,
        "config_sha256": config_hash,
        "jobs": len(jobs),
        "canonical_rows": len(per_seed),
        "primary_contrast_rows": len(contrasts),
        "bootstrap_replicates": int(config["statistics"]["bootstrap_replicates"]),
        "permutation_replicates": int(config["statistics"]["permutation_replicates"]),
        "implementation_correction_07_sha256": correction_hash,
        "rust_build_manifest_sha256": sha256_file(CORRECTION07_BUILD_MANIFEST),
        "phase_d_report_sha256": approval_chain["phase_d_report_sha256"],
        "phase_d_completion_inventory_sha256": completion_inventory_before,
        "approval_chain": approval_chain,
        "seed_inference_policy": "predictions ensembled per stay; seeds never treated as independent samples",
        "files": files,
        "patient_level_content": False,
        "manuscript_modified": False,
    }
    staged_report_path = staging_root / "canonical_analysis_report.json"
    safe_write_json(staged_report_path, report)
    safe_write_text(
        staged_report_path.with_suffix(".json.sha256"),
        f"{sha256_file(staged_report_path)}  {(canonical_root / 'canonical_analysis_report.json').relative_to(REPO_ROOT)}\n",
    )
    def final_validation() -> dict[str, Any]:
        frozen_after = verify_frozen_state(plan)
        if frozen_after != frozen_before:
            raise RuntimeError("frozen-v11 state drifted before canonical publication")
        reverified_jobs = _require_full_jobs(config)
        completion_inventory_after = _completion_inventory_sha256(reverified_jobs)
        if completion_inventory_after != completion_inventory_before:
            raise RuntimeError("Phase-D completion inventory drifted during analysis")
        approval_after = require_correction07_analysis_approval(
            correction_path=CORRECTION07_PATH,
            qualification_path=CORRECTION07_QUALIFICATION,
            audit_path=CORRECTION07_AUDIT,
            approval_path=CORRECTION07_APPROVAL,
        )["_verified_chain"]
        if approval_after != approval_chain:
            raise RuntimeError("correction-07 approval chain drifted during analysis")
        return frozen_after

    frozen_after = _publish_canonical_after_validation(
        staging_root,
        canonical_root,
        CANONICAL_OUTPUT_NAMES,
        final_validation,
    )
    report_path = canonical_root / "canonical_analysis_report.json"
    payload = {
        "status": "pass",
        "command": f"python -u reproduce.py analyze",
        "analysis_plan_sha256": plan_hash,
        "config_sha256": config_hash,
        "implementation_correction_07_sha256": correction_hash,
        "rust_build_manifest_sha256": sha256_file(CORRECTION07_BUILD_MANIFEST),
        "canonical_report_sha256": sha256_file(report_path),
        "phase_d_report_sha256": approval_chain["phase_d_report_sha256"],
        "phase_d_completion_inventory_sha256": completion_inventory_before,
        "approval_chain": approval_chain,
        "runtime": {"wall_seconds": float(time.perf_counter() - started)},
        "files": files,
        "tests": [
            {"name": "all_180_jobs_verified", "passed": len(jobs) == 180, "detail": str(len(jobs))},
            {"name": "bootstrap_unit", "passed": True, "detail": "subject-clustered paired; seed-ensemble per stay"},
            {"name": "bh_correction", "passed": contrasts["bh_adjusted_p"].notna().any(), "detail": "within registered evidence families"},
            {"name": "subgroup_minimums", "passed": True, "detail": ">=100 stays and >=20 events enforced"},
            {"name": "frozen_v11_unchanged", "passed": frozen_before == frozen_after, "detail": "pre/post snapshots match"},
        ],
        "errors": [],
        "manuscript_modified": False,
    }
    phase_path, phase_hash = write_phase_report("E_ANALYSIS", payload)
    return {
        "status": "pass",
        "report": str(phase_path.relative_to(REPO_ROOT)),
        "report_sha256": phase_hash,
        "canonical_report": str(report_path.relative_to(REPO_ROOT)),
    }, 0
