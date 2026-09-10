from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auxiliary_los_contract import DEEP_BASELINE_MODELS
from reproduction_contract import require_run
from correction06_artifacts import archive_incomplete_public_job_report
from landmark_firewall import (
    load_features,
    load_source_training_data,
    lock_source_choices,
    lock_target_predictions,
    open_target_labels_for_locked_evaluation,
)
from landmark_jobs import (
    LandmarkJobStore,
    build_job_matrix,
    load_or_create_source_split,
    training_config_for_job,
)
from landmark_protocol import REPO_ROOT, RESULTS_ROOT, WORKSPACE_ROOT, load_protocol, load_verified_phase_report, verify_frozen_state
from v13_analysis import canonical_metric_row, temporal_stress_view
from v13_guard import assert_safe_output_path, safe_write_json, safe_write_text, sha256_file
from v13_inference import ModelPredictor
from v13_model_registry import MODEL_NAMES
from v13_resource_guardian import ResourceGuardian, resource_safe_point
from v13_training import (
    fit_normalization_chunked,
    save_training_payloads,
    set_deterministic_seed,
    train_deep_baseline,
    train_xgb_or_mamba_primary,
)


def _save_npz(path: Path, **arrays: np.ndarray) -> Path:
    target = assert_safe_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.pid{os.getpid()}.tmp")
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return target


def _scenario_predictions(
    predictor: ModelPredictor,
    arrays: dict[str, np.ndarray],
    scenarios: list[str],
    *,
    seed: int,
    rows: np.ndarray | None = None,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    selected = (
        np.arange(len(arrays["values"]), dtype=np.int64)
        if rows is None else np.asarray(rows, dtype=np.int64)
    )
    probabilities = np.empty((len(selected), len(scenarios)), dtype=np.float32)
    los_predictions = np.empty_like(probabilities)
    audits: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        observed_cells = 0
        zero_event_stays = 0
        for start in range(0, len(selected), batch_size):
            source_rows = selected[start:start + batch_size]
            view, audit = temporal_stress_view(
                np.asarray(arrays["values"][source_rows], dtype=np.float32),
                np.asarray(arrays["mask"][source_rows], dtype=np.uint8),
                np.asarray(arrays["tau"][source_rows], dtype=np.float32),
                np.asarray(arrays["elapsed"][source_rows], dtype=np.float32),
                scenario=scenario,
                seed=seed,
                source_row_ids=source_rows,
            )
            if len(view["source_row_indices"]) != len(source_rows):
                raise RuntimeError(f"stress scenario changed the fixed risk set: {scenario}")
            probability, los = predictor.predict(
                view["values"],
                view["tau"],
                view["mask"],
                np.asarray(arrays["static"][source_rows], dtype=np.float32),
            )
            stop = start + len(source_rows)
            probabilities[start:stop, scenario_index] = probability
            los_predictions[start:stop, scenario_index] = los
            observed_cells += int(audit["observed_cells"])
            zero_event_stays += int(audit["zero_event_included_stays"])
            resource_safe_point(
                f"inference_{scenario}_rows_{start}_{stop}", force_cleanup=False
            )
        audits.append({
            "scenario": scenario,
            "stays": int(len(selected)),
            "observed_cells": observed_cells,
            "zero_event_stays": zero_event_stays,
            "risk_set_preserved": True,
        })
        resource_safe_point(f"inference_{scenario}_complete", force_cleanup=True)
    return probabilities, los_predictions, audits


def _metric_rows(
    *,
    dataset: str,
    direction: str,
    model: str,
    seed: int,
    scenarios: list[str],
    mortality: np.ndarray,
    los: np.ndarray,
    probabilities: np.ndarray,
    los_predictions: np.ndarray,
    population: str,
) -> list[dict[str, Any]]:
    return [
        canonical_metric_row(
            dataset=dataset,
            direction=direction,
            population=population,
            model=model,
            seed=seed,
            scenario=scenario,
            mortality_labels=mortality,
            mortality_probabilities=probabilities[:, index],
            los_labels=los,
            los_predictions=los_predictions[:, index],
            calibration_status="source_validation_calibrated_zero_shot",
        )
        for index, scenario in enumerate(scenarios)
    ]


def _prepare_retry_attempt(
    store: LandmarkJobStore,
    job: Any,
    *,
    plan_hash: str,
    config_hash: str,
    retry_correction_sha256: str,
    implementation_correction_sha256: str | None,
    archive_external_residue: Any | None = None,
) -> tuple[Path, Path] | None:
    """Return a clean claimed root, or None for a hash-verified completion."""
    if store.is_complete(job):
        return None
    store.archive_incomplete_attempt(
        job,
        correction_sha256=retry_correction_sha256,
    )
    if archive_external_residue is not None:
        archive_external_residue()
    job_root = store.prepare(
        job,
        plan_hash=plan_hash,
        config_hash=config_hash,
        implementation_correction_sha256=implementation_correction_sha256,
        retry_correction_sha256=retry_correction_sha256,
    )
    return job_root, store.claim(job)


def run_job(job_id: str, *, bounded: bool = False) -> dict[str, Any]:
    if bounded and not (REPO_ROOT / 'SYNTHETIC_SMOKE_ONLY').is_file():
        raise PermissionError('Bounded jobs are restricted to the isolated synthetic smoke checkout')
    plan, plan_hash, config, config_hash = load_protocol()
    load_verified_phase_report("B_ETL", allowed_statuses={"pass"})
    jobs = {job.job_id: job for job in build_job_matrix(config)}
    if job_id not in jobs:
        raise ValueError(f"unknown landmark job: {job_id}")
    job = jobs[job_id]
    store = LandmarkJobStore()
    if store.is_complete(job):
        return {"job_id": job.job_id, "status": "already_complete"}
    run_identity = require_run()
    retry_correction = {"record_sha256": run_identity["record_sha256"]}
    active_deep_correction = retry_correction if job.model in DEEP_BASELINE_MODELS else None
    implementation_corrections = [{"mode": "fresh_public_reproduction", **retry_correction}]
    if not torch.cuda.is_available() and not bounded:
        raise RuntimeError("production landmark training requires a visible CUDA device")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frozen_before = verify_frozen_state(plan)
    prepared = _prepare_retry_attempt(
        store,
        job,
        plan_hash=plan_hash,
        config_hash=config_hash,
        implementation_correction_sha256=(
            active_deep_correction["record_sha256"]
            if active_deep_correction is not None else None
        ),
        retry_correction_sha256=retry_correction["record_sha256"],
        archive_external_residue=lambda: archive_incomplete_public_job_report(
            RESULTS_ROOT / "jobs" / f"{job.job_id}.json",
            job_id=job.job_id,
            correction_sha256=retry_correction["record_sha256"],
        ),
    )
    if prepared is None:
        return {"job_id": job.job_id, "status": "already_complete"}
    job_root, claim = prepared
    guardian = ResourceGuardian(
        component=job.job_id,
        status_path=job_root / "resource_guardian_status.json",
        config=config["guardian"],
    )
    started = time.perf_counter()
    guardian_started = False
    try:
        guardian.start()
        guardian_started = True
        set_deterministic_seed(job.seed)
        arrays = load_source_training_data(job.source_dataset)
        source_manifest = (
            REPO_ROOT / config["datasets"][job.source_dataset]["private_root"]
            / "manifest_private.json"
        )
        indices, split_audit, split_path = load_or_create_source_split(
            job.source_dataset,
            job.seed,
            arrays,
            plan_hash=plan_hash,
            config_hash=config_hash,
            etl_manifest_hash=sha256_file(source_manifest),
            config=config,
        )
        guardian.safe_point("source_split_locked")
        normalization = fit_normalization_chunked(arrays, indices["train"])
        guardian.safe_point("source_normalization_fitted", force_cleanup=True)
        training_config = training_config_for_job(config, job)
        if job.model in MODEL_NAMES[:6]:
            result, source_predictions, model_artifacts = train_xgb_or_mamba_primary(
                job.model,
                arrays,
                indices,
                normalization,
                seed=job.seed,
                max_epochs=(1 if bounded else int(config["training"]["mamba"]["max_epochs"])),
                patience=int(config["training"]["mamba"]["early_stopping_patience"]),
                n_estimators=(2 if bounded else int(config["training"]["xgboost"]["n_estimators"])),
                device=device,
                job_root=job_root,
                training_config=training_config,
                bounded=bounded,
            )
        else:
            result, source_predictions, model_artifacts = train_deep_baseline(
                job.model,
                arrays,
                indices,
                normalization,
                seed=job.seed,
                max_epochs=(1 if bounded else int(config["training"]["deep_baselines"]["max_epochs"])),
                patience=int(config["training"]["deep_baselines"]["early_stopping_patience"]),
                device=device,
                job_root=job_root,
                bounded=bounded,
                correction_sha256=active_deep_correction["record_sha256"],
                los_correction_sha256=None,
                predecessor_correction_sha256=None,
                source_dataset=job.source_dataset,
            )
        normalization_path, source_prediction_path = save_training_payloads(
            job_root, normalization, source_predictions
        )
        choices_path = job_root / "source_choices.json"
        choices_hash = lock_source_choices(
            choices_path,
            result["choices"],
            source=job.source_dataset,
            plan_hash=plan_hash,
            config_hash=config_hash,
        )
        choices_hash_path = choices_path.with_suffix(".json.sha256")
        predictor = ModelPredictor(job.model, job_root, device=device)
        scenarios = [str(value) for value in config["stress_scenarios"]]
        all_metric_rows: list[dict[str, Any]] = []
        private_artifacts: list[Path] = [
            *model_artifacts,
            normalization_path,
            source_prediction_path,
            choices_path,
            choices_hash_path,
        ]

        internal_probability, internal_los_prediction, internal_audit = _scenario_predictions(
            predictor,
            arrays,
            scenarios,
            seed=job.seed,
            rows=indices["test"],
            batch_size=64 if bounded else 512,
        )
        internal_eval_path = _save_npz(
            job_root / "source_internal_evaluation_private.npz",
            mortality=np.asarray(arrays["mortality"])[indices["test"]].astype(np.int8),
            los=np.asarray(arrays["los"])[indices["test"]].astype(np.float32),
            stay_keys=np.asarray(arrays["stay_keys"])[indices["test"]],
            group_keys=np.asarray(arrays["group_keys"])[indices["test"]],
            age=np.asarray(arrays["age"])[indices["test"]].astype(np.float32),
            sex_male=np.asarray(arrays["sex_male"])[indices["test"]].astype(np.int8),
            icu_type_code=np.asarray(arrays["icu_type_code"])[indices["test"]].astype(np.int16),
            mortality_probability=internal_probability,
            los_prediction=internal_los_prediction,
        )
        private_artifacts.append(internal_eval_path)
        all_metric_rows.extend(_metric_rows(
            dataset=job.source_dataset,
            direction="source_internal",
            model=job.model,
            seed=job.seed,
            scenarios=scenarios,
            mortality=np.asarray(arrays["mortality"])[indices["test"]],
            los=np.asarray(arrays["los"])[indices["test"]],
            probabilities=internal_probability,
            los_predictions=internal_los_prediction,
            population="locked_source_test_fixed_6h_risk_set",
        ))

        target_audits: dict[str, Any] = {}
        for target_dataset in job.targets:
            target_features = load_features(target_dataset)
            probability, target_los_prediction, scenario_audit = _scenario_predictions(
                predictor,
                target_features,
                scenarios,
                seed=job.seed,
                batch_size=64 if bounded else 512,
            )
            prediction_path = _save_npz(
                job_root / f"target_{target_dataset}_predictions_private.npz",
                mortality_probability=probability,
                los_prediction=target_los_prediction,
                scenario_names=np.asarray(scenarios, dtype="U48"),
            )
            lock = lock_target_predictions(
                prediction_path,
                dataset=target_dataset,
                source_choices_path=choices_path,
            )
            private_artifacts.extend([
                prediction_path,
                prediction_path.with_suffix(".npz.sha256"),
                prediction_path.with_suffix(".npz.lock.json"),
                prediction_path.with_suffix(".npz.lock.json.sha256"),
            ])
            target_labels = open_target_labels_for_locked_evaluation(
                target_dataset,
                prediction_path=prediction_path,
                source_choices_path=choices_path,
            )
            evaluation_path = _save_npz(
                job_root / f"target_{target_dataset}_evaluation_private.npz",
                mortality=np.asarray(target_labels["mortality"], dtype=np.int8),
                los=np.asarray(target_labels["los"], dtype=np.float32),
                stay_keys=np.asarray(target_labels["stay_keys"]),
                group_keys=np.asarray(target_labels["group_keys"]),
                age=np.asarray(target_labels["age"], dtype=np.float32),
                sex_male=np.asarray(target_labels["sex_male"], dtype=np.int8),
                icu_type_code=np.asarray(target_labels["icu_type_code"], dtype=np.int16),
                mortality_probability=probability,
                los_prediction=target_los_prediction,
            )
            private_artifacts.append(evaluation_path)
            all_metric_rows.extend(_metric_rows(
                dataset=target_dataset,
                direction=f"{job.source_dataset}_to_{target_dataset}",
                model=job.model,
                seed=job.seed,
                scenarios=scenarios,
                mortality=target_labels["mortality"],
                los=target_labels["los"],
                probabilities=probability,
                los_predictions=target_los_prediction,
                population="full_target_fixed_6h_risk_set",
            ))
            target_audits[target_dataset] = {
                "row_count": int(len(target_labels["mortality"])),
                "prediction_lock": lock,
                "scenarios": scenario_audit,
                "labels_opened_only_after_prediction_lock": True,
            }
            del target_features, target_labels, probability, target_los_prediction
            guardian.safe_point(f"target_{target_dataset}_complete", force_cleanup=True)

        runtime = {
            "wall_seconds": float(time.perf_counter() - started),
            "device": device,
            "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "peak_torch_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }
        public_report = {
            "schema_version": "1.0.0",
            "job_id": job.job_id,
            "source_dataset": job.source_dataset,
            "model": job.model,
            "seed": job.seed,
            "training_regime": job.training_regime,
            "registry_role": job.registry_role,
            "analysis_plan_sha256": plan_hash,
            "config_sha256": config_hash,
            "source_etl_manifest_sha256": sha256_file(source_manifest),
            "source_split_sha256": sha256_file(split_path),
            "source_choices_sha256": choices_hash,
            "source_split_audit": split_audit,
            "training_audit": result["choices"].get("training_audit", {}),
            "target_firewall_audit": target_audits,
            "source_internal_scenario_audit": internal_audit,
            "canonical_metrics": all_metric_rows,
            "runtime": runtime,
            "target_information_used_for_training_or_selection": False,
            "patient_level_content": False,
            "manuscript_modified": False,
            "implementation_corrections": implementation_corrections,
        }
        public_path = RESULTS_ROOT / "jobs" / f"{job.job_id}.json"
        safe_write_json(public_path, public_report)
        safe_write_text(
            public_path.with_suffix(".json.sha256"),
            f"{sha256_file(public_path)}  {public_path.relative_to(REPO_ROOT)}\n",
        )
        private_artifacts.extend([
            job_root / "resource_guardian_status.json",
        ])
        guardian.stop("complete")
        guardian_started = False
        store.complete(job, private_artifacts)
        frozen_after = verify_frozen_state(plan)
        if frozen_before != frozen_after:
            raise RuntimeError("frozen v11 changed during landmark job")
        return {
            "job_id": job.job_id,
            "status": "complete",
            "report": str(public_path.relative_to(REPO_ROOT)),
            "report_sha256": sha256_file(public_path),
            "runtime": runtime,
        }
    except BaseException:
        if guardian_started:
            guardian.stop("failed")
        if claim.exists():
            claim.unlink()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--bounded", action="store_true")
    args = parser.parse_args()
    result = run_job(args.job_id, bounded=args.bounded)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
