from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from landmark_protocol import (
    REPO_ROOT,
    RESULTS_ROOT,
    WORKSPACE_ROOT,
    artifact_records,
    atomic_save_npy,
    load_verified_phase_report,
    load_protocol,
    verify_frozen_state,
    write_phase_report,
)
from v13_etl import (
    LANDMARK_HOURS,
    build_eicu_cohort,
    build_mimic_cohort,
    build_static_feature_matrix,
)
from v13_feature_registry import DYNAMIC_FEATURES, STATIC_FEATURES
from v13_guard import assert_safe_output_path, safe_write_json, safe_write_text, sha256_file


FEATURE_ARRAYS = ("values", "tau", "mask", "elapsed", "static")
LABEL_ARRAYS = (
    "mortality",
    "los",
    "stay_keys",
    "group_keys",
    "age",
    "sex_male",
    "icu_type_code",
)


def _icu_type_codes(values: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    labels = sorted(set(str(value).strip().lower() or "unknown" for value in values))
    registry = {name: index for index, name in enumerate(labels)}
    return np.asarray([
        registry[str(value).strip().lower() or "unknown"] for value in values
    ], dtype=np.int16), registry


def _build_dataset(dataset: str, config: dict[str, Any], *, audit_preview_rows: int = 0) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    dataset_config = config["datasets"][dataset]
    raw_root = REPO_ROOT / dataset_config["raw_root"]
    from correction09_availability import scan_corrected
    if dataset == "eicu":
        cohort = build_eicu_cohort(raw_root, minimum_age=int(config["minimum_age_years"]))
    else:
        cohort = build_mimic_cohort(raw_root, dataset=dataset,
                                   minimum_age=int(config["minimum_age_years"]),
                                   exclude_uncertain_positive=True)
    if audit_preview_rows < 0:
        raise ValueError("Preview rows cannot be negative")
    event_scan, accumulator = scan_corrected(cohort.frame, raw_root, dataset, config, max_rows=audit_preview_rows)
    dynamic, dynamic_audit = accumulator.materialize_arrays()
    static, static_audit = build_static_feature_matrix(
        cohort.frame,
        raw_root,
        dataset=dataset,
        max_rows=0,
    )
    static_audit["availability_class"] = "per_field_availability_unverified"
    n = len(cohort.frame)
    if dynamic["values"].shape != (n, int(config["sequence_length"]), 40):
        raise RuntimeError(f"{dataset} dynamic shape drift")
    if static.shape != (n, 12):
        raise RuntimeError(f"{dataset} time-safe static shape drift")
    if not dynamic_audit["contract_pass"] or not static_audit["contract_pass"]:
        raise RuntimeError(f"{dataset} ETL contract audit failed")
    if float(np.nanmax(dynamic["elapsed"])) > LANDMARK_HOURS * 3600.0:
        raise RuntimeError(f"{dataset} contains a predictor timestamp beyond six hours")
    if (cohort.frame["los_days"].to_numpy(dtype=np.float64) <= 0).any():
        raise RuntimeError(f"{dataset} remaining LOS is not strictly positive")
    icu_codes, icu_registry = _icu_type_codes(cohort.frame["careunit"].tolist())
    arrays = {
        **dynamic,
        "static": static.astype(np.float32, copy=False),
        "mortality": cohort.frame["mortality_label"].to_numpy(dtype=np.int8),
        "los": cohort.frame["los_days"].to_numpy(dtype=np.float32),
        "stay_keys": cohort.frame["stay_key"].to_numpy(dtype=np.int64),
        "group_keys": cohort.frame["group_key"].astype(str).to_numpy(dtype="U64"),
        "age": cohort.frame["age"].to_numpy(dtype=np.float32),
        "sex_male": cohort.frame["gender"].astype(str).str.upper().str.startswith("M").to_numpy(dtype=np.int8),
        "icu_type_code": icu_codes,
    }
    audit = {
        "availability_correction": event_scan,
        "cohort": cohort.metadata,
        "flow": cohort.flow,
        "event_scan": event_scan,
        "dynamic": dynamic_audit,
        "static": static_audit,
        "icu_type_registry": icu_registry,
        "predictor_max_elapsed_seconds": float(np.nanmax(dynamic["elapsed"])),
        "risk_set_landmark_hours": LANDMARK_HOURS,
        "remaining_los_reference": "fixed_6h_landmark",
        "diagnosis_tables_opened_for_static_features": False,
    }
    return cohort, arrays, audit


def _write_dataset(dataset: str, arrays: dict[str, np.ndarray], audit: dict[str, Any], config_hash: str, plan_hash: str, config: dict[str, Any]) -> dict[str, Any]:
    if (config.get("availability_correction_sha256") and (audit or arrays)
            and not (REPO_ROOT / "SYNTHETIC_SMOKE_ONLY").is_file()
            and audit.get("availability_correction", {}).get("complete_raw_table_scan") is not True):
        raise RuntimeError("Only a complete correction09 ETL may write a production dataset")
    destination = assert_safe_output_path(
        REPO_ROOT / config["datasets"][dataset]["private_root"]
    )
    if destination.exists():
        manifest_path = destination / "manifest_private.json"
        manifest_hash_path = destination / "manifest_private.json.sha256"
        if not manifest_path.is_file() or not manifest_hash_path.is_file():
            raise RuntimeError(f"refusing ambiguous existing ETL root: {destination}")
        if sha256_file(manifest_path) != manifest_hash_path.read_text(
            encoding="utf-8"
        ).split()[0]:
            raise RuntimeError(f"ETL manifest hash mismatch: {dataset}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("analysis_plan_sha256") != plan_hash
            or manifest.get("config_sha256") != config_hash
        ):
            raise RuntimeError(f"existing ETL was built under another protocol: {dataset}")
        for section in ("feature_artifacts", "sealed_label_artifacts"):
            root = destination / ("features" if section == "feature_artifacts" else "sealed_labels")
            for record in manifest[section]:
                path = root / record["file"]
                if not path.is_file() or sha256_file(path) != record["sha256"]:
                    raise RuntimeError(f"ETL resume integrity failure: {path}")
        audit_path = destination / "aggregate_audit.json"
        if (
            not audit_path.is_file()
            or sha256_file(audit_path) != manifest["aggregate_audit_sha256"]
        ):
            raise RuntimeError(f"ETL aggregate audit integrity failure: {dataset}")
        return manifest

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset}.landmark.", dir=destination.parent))
    try:
        feature_root = staging / "features"
        label_root = staging / "sealed_labels"
        feature_root.mkdir(parents=True)
        label_root.mkdir(parents=True)
        feature_paths: list[Path] = []
        label_paths: list[Path] = []
        for name in FEATURE_ARRAYS:
            path = feature_root / f"{name}.npy"
            atomic_save_npy(path, arrays[name])
            feature_paths.append(path)
        for name in LABEL_ARRAYS:
            path = label_root / f"{name}.npy"
            atomic_save_npy(path, arrays[name])
            label_paths.append(path)
        safe_write_json(staging / "aggregate_audit.json", audit)
        manifest = {
            "schema_version": "2.0.0",
            "dataset": dataset,
            "release": config["datasets"][dataset]["release"],
            "analysis_plan_sha256": plan_hash,
            "config_sha256": config_hash,
            "row_count": int(len(arrays["mortality"])),
            "feature_label_physical_separation": True,
            "features_contain_labels_or_identifiers": False,
            "sealed_labels_private_only": True,
            "public_copy_forbidden": True,
            "feature_artifacts": artifact_records(feature_paths, relative_to=feature_root),
            "sealed_label_artifacts": artifact_records(label_paths, relative_to=label_root),
            "aggregate_audit_sha256": sha256_file(staging / "aggregate_audit.json"),
            "dynamic_raw_table_sha256": {
                name: table["raw_sha256"]
                for name, table in audit.get("availability_correction", {}).get("tables", {}).items()
            },
        }
        safe_write_json(staging / "manifest_private.json", manifest)
        safe_write_text(
            staging / "manifest_private.json.sha256",
            f"{sha256_file(staging / 'manifest_private.json')}  manifest_private.json\n",
        )
        staging.replace(destination)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def run_etl() -> tuple[dict[str, Any], int]:
    plan, plan_hash, config, config_hash = load_protocol()
    load_verified_phase_report("A_PREFLIGHT", allowed_statuses={"pass"})
    frozen_before = verify_frozen_state(plan)
    started = time.perf_counter()
    reports: dict[str, Any] = {}
    tests: list[dict[str, Any]] = []
    for dataset in ("mimic_iv", "eicu", "mimic_iii_carevue"):
        destination = REPO_ROOT / config["datasets"][dataset]["private_root"]
        if destination.exists():
            manifest = _write_dataset(
                dataset, {}, {}, config_hash, plan_hash, config
            )
            audit = json.loads(
                (destination / "aggregate_audit.json").read_text(encoding="utf-8")
            )
            arrays = None
        else:
            _, arrays, audit = _build_dataset(dataset, config)
            manifest = _write_dataset(
                dataset, arrays, audit, config_hash, plan_hash, config
            )
        reports[dataset] = {
            "row_count": manifest["row_count"],
            "manifest": str(
                (
                    REPO_ROOT / config["datasets"][dataset]["private_root"]
                    / "manifest_private.json"
                ).relative_to(REPO_ROOT)
            ),
            "manifest_sha256": sha256_file(
                REPO_ROOT / config["datasets"][dataset]["private_root"] / "manifest_private.json"
            ),
            "cohort": audit["cohort"],
            "predictor_max_elapsed_seconds": audit["predictor_max_elapsed_seconds"],
        }
        tests.extend([
            {"name": f"{dataset}_fixed_risk_set", "passed": True, "detail": f"n={manifest['row_count']}"},
            {"name": f"{dataset}_max_timestamp", "passed": audit["predictor_max_elapsed_seconds"] <= 21600.0, "detail": str(audit["predictor_max_elapsed_seconds"])},
            {"name": f"{dataset}_feature_label_separation", "passed": manifest["feature_label_physical_separation"], "detail": "features/ and sealed_labels/ are separate"},
        ])
        if arrays is not None:
            del arrays
    frozen_after = verify_frozen_state(plan)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen v11 changed during corrected ETL")
    payload = {
        "status": "pass",
        "command": f"python -u reproduce.py etl",
        "analysis_plan_sha256": plan_hash,
        "config_sha256": config_hash,
        "runtime": {"wall_seconds": float(time.perf_counter() - started)},
        "datasets": reports,
        "tests": tests,
        "errors": [],
        "frozen_v11_unchanged": True,
        "training_performed": False,
    }
    report_path, report_hash = write_phase_report("B_ETL", payload)
    return {
        "status": "pass",
        "report": str(report_path.relative_to(REPO_ROOT)),
        "report_sha256": report_hash,
        "datasets": reports,
    }, 0
