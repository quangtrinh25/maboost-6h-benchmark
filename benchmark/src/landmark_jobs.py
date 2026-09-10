from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from landmark_protocol import REPO_ROOT, WORKSPACE_ROOT, artifact_records
from v13_guard import assert_safe_output_path, safe_write_json, safe_write_text, sha256_file
from v13_model_registry import MODEL_NAMES


MATCHED_MODELS = (
    "XGBOOST-FLAT",
    "XGBOOST-SUMMARY+STATIC",
    "Hybrid-plain",
    "MABOOST",
)
SOURCES = ("mimic_iv", "eicu")


def model_slug(model: str) -> str:
    return model.lower().replace("+", "plus").replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class LandmarkJob:
    job_id: str
    source_dataset: str
    model: str
    seed: int
    training_regime: str
    registry_role: str
    targets: tuple[str, ...]


def build_job_matrix(config: dict[str, Any], *, seeds: Iterable[int] | None = None) -> list[LandmarkJob]:
    active_seeds = [int(value) for value in (seeds or config["seeds"])]
    configured_seeds = {int(value) for value in config["seeds"]}
    if not set(active_seeds).issubset(configured_seeds):
        raise ValueError("requested seed is outside the locked registry")
    jobs: list[LandmarkJob] = []
    for source in SOURCES:
        targets = tuple(str(value) for value in config["source_directions"][source])
        for seed in active_seeds:
            for model in MODEL_NAMES:
                regime = "stress_augmented" if model == "MABOOST" else "full_only"
                job_id = (
                    f"{source}__{model_slug(model)}__{regime}__seed{seed}"
                )
                jobs.append(LandmarkJob(
                    job_id=job_id,
                    source_dataset=source,
                    model=model,
                    seed=seed,
                    training_regime=regime,
                    registry_role="primary_registry",
                    targets=targets,
                ))
            alternatives = {
                "MABOOST": "full_only",
                "Hybrid-plain": "stress_augmented",
                "XGBOOST-FLAT": "stress_augmented",
                "XGBOOST-SUMMARY+STATIC": "stress_augmented",
            }
            for model, regime in alternatives.items():
                job_id = (
                    f"{source}__{model_slug(model)}__{regime}__seed{seed}"
                )
                jobs.append(LandmarkJob(
                    job_id=job_id,
                    source_dataset=source,
                    model=model,
                    seed=seed,
                    training_regime=regime,
                    registry_role="matched_factorial",
                    targets=targets,
                ))
    expected = len(active_seeds) * len(SOURCES) * (len(MODEL_NAMES) + 4)
    if len(jobs) != expected or len({job.job_id for job in jobs}) != expected:
        raise AssertionError(f"landmark job matrix must contain {expected} unique jobs")
    return jobs


def _group_score(group: str, seed: int) -> float:
    digest = hashlib.sha256(f"landmark-split|{seed}|{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def load_or_create_source_split(
    dataset: str,
    seed: int,
    arrays: dict[str, np.ndarray],
    *,
    plan_hash: str,
    config_hash: str,
    etl_manifest_hash: str,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    if dataset not in SOURCES:
        raise PermissionError("only development sources receive train/validation/test splits")
    path = assert_safe_output_path(
        WORKSPACE_ROOT / "private" / "splits" / f"{dataset}_seed{seed}.json"
    )
    hash_path = path.with_suffix(".json.sha256")
    if path.is_file():
        if not hash_path.is_file() or sha256_file(path) != hash_path.read_text(
            encoding="utf-8"
        ).split()[0]:
            raise RuntimeError(f"source split lock integrity failure: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        group_keys = np.asarray(arrays["group_keys"]).astype(str)
        unique_groups = sorted(set(group_keys.tolist()))
        train_cut = float(config["splits"]["train_fraction"])
        validation_cut = train_cut + float(config["splits"]["validation_fraction"])
        group_partition: dict[str, str] = {}
        for group in unique_groups:
            score = _group_score(group, seed)
            group_partition[group] = (
                "train" if score < train_cut
                else "validation" if score < validation_cut
                else "test"
            )
        partitions = {
            split: np.flatnonzero(
                np.asarray([group_partition[value] == split for value in group_keys])
            ).astype(np.int64).tolist()
            for split in ("train", "validation", "test")
        }
        payload = {
            "schema_version": "1.0.0",
            "dataset": dataset,
            "seed": int(seed),
            "analysis_plan_sha256": plan_hash,
            "config_sha256": config_hash,
            "etl_manifest_sha256": etl_manifest_hash,
            "unit": "patient",
            "fractions": {
                "train": float(config["splits"]["train_fraction"]),
                "validation": float(config["splits"]["validation_fraction"]),
                "test": float(config["splits"]["test_fraction"]),
            },
            "label_stratification": False,
            "target_labels_used": False,
            "partitions": partitions,
        }
        safe_write_json(path, payload)
        safe_write_text(
            hash_path,
            f"{sha256_file(path)}  {path.relative_to(REPO_ROOT)}\n",
        )
    required_bindings = {
        "dataset": dataset,
        "seed": int(seed),
        "analysis_plan_sha256": plan_hash,
        "config_sha256": config_hash,
        "etl_manifest_sha256": etl_manifest_hash,
        "target_labels_used": False,
    }
    for field, expected in required_bindings.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"stale source split binding: {field}")
    indices = {
        split: np.asarray(payload["partitions"][split], dtype=np.int64)
        for split in ("train", "validation", "test")
    }
    all_rows = np.concatenate(list(indices.values()))
    if len(all_rows) != len(arrays["values"]) or len(np.unique(all_rows)) != len(all_rows):
        raise RuntimeError("source split does not partition the cohort exactly once")
    group_sets = {
        split: set(np.asarray(arrays["group_keys"])[rows].astype(str).tolist())
        for split, rows in indices.items()
    }
    overlap = {
        "train_validation": len(group_sets["train"] & group_sets["validation"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "validation_test": len(group_sets["validation"] & group_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"patient split leakage: {overlap}")
    for split, rows in indices.items():
        classes = np.unique(np.asarray(arrays["mortality"])[rows])
        if len(rows) == 0 or classes.size != 2:
            raise RuntimeError(f"source {split} lacks rows or both outcome classes")
    audit = {
        "counts": {key: int(len(value)) for key, value in indices.items()},
        "patient_counts": {key: int(len(value)) for key, value in group_sets.items()},
        "group_overlap": overlap,
        "label_stratification": False,
        "target_information_used": False,
        "split_sha256": sha256_file(path),
    }
    return indices, audit, path


class LandmarkJobStore:
    def __init__(
        self,
        *,
        root: Path | None = None,
        attempt_archive_root: Path | None = None,
    ) -> None:
        self.root = assert_safe_output_path(
            root if root is not None else WORKSPACE_ROOT / "private" / "jobs"
        )
        self.attempt_archive_root = assert_safe_output_path(
            attempt_archive_root
            if attempt_archive_root is not None
            else (
                WORKSPACE_ROOT
                / "superseded"
                / "implementation_correction_06_retry_idempotency"
                / "job_attempts"
            )
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def root_for(self, job: LandmarkJob) -> Path:
        return assert_safe_output_path(self.root / job.job_id)

    def prepare(
        self,
        job: LandmarkJob,
        *,
        plan_hash: str,
        config_hash: str,
        implementation_correction_sha256: str | None = None,
        retry_correction_sha256: str | None = None,
    ) -> Path:
        root = self.root_for(job)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "job_spec.json"
        payload: dict[str, Any] = {
            **asdict(job),
            "targets": list(job.targets),
            "analysis_plan_sha256": plan_hash,
            "config_sha256": config_hash,
        }
        if implementation_correction_sha256 is not None:
            payload["implementation_correction_sha256"] = (
                implementation_correction_sha256
            )
        if retry_correction_sha256 is not None:
            payload["retry_correction_sha256"] = retry_correction_sha256
        if path.is_file():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise RuntimeError(f"immutable landmark job spec mismatch: {job.job_id}")
        else:
            safe_write_json(path, payload)
            safe_write_text(
                path.with_suffix(".json.sha256"),
                f"{sha256_file(path)}  job_spec.json\n",
            )
        return root

    def is_complete(self, job: LandmarkJob) -> bool:
        root = self.root_for(job)
        path = root / "completion_private.json"
        if not path.is_file():
            return False
        hash_path = path.with_suffix(".json.sha256")
        if not hash_path.is_file() or sha256_file(path) != hash_path.read_text(
            encoding="utf-8"
        ).split()[0]:
            raise RuntimeError(f"completion lock mismatch: {job.job_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload["artifacts"]:
            artifact = root / record["file"]
            if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
                raise RuntimeError(f"completed artifact drift: {job.job_id}/{record['file']}")
        return True

    @staticmethod
    def _claim_pid(path: Path) -> int:
        try:
            return int(path.read_text(encoding="utf-8").split("=", 1)[1])
        except (IndexError, ValueError) as error:
            raise RuntimeError(
                f"malformed job claim: {path.parent.name}"
            ) from error

    def _assert_no_live_claim(self, root: Path) -> None:
        claim = root / "RUNNING.lock"
        if claim.is_file():
            pid = self._claim_pid(claim)
            if Path("/proc", str(pid)).exists():
                raise RuntimeError(f"job already active: {root.name} pid={pid}")

    def archive_incomplete_attempt(
        self,
        job: LandmarkJob,
        *,
        correction_sha256: str,
    ) -> dict[str, Any] | None:
        """Atomically quarantine one incomplete job root with per-file hashes."""
        root = self.root_for(job)
        if not root.exists():
            return None
        if self.is_complete(job):
            raise PermissionError(
                f"refusing to archive completed job: {job.job_id}"
            )
        self._assert_no_live_claim(root)

        parent = assert_safe_output_path(
            self.attempt_archive_root / job.job_id
        )
        attempt = 1
        while (parent / f"attempt_{attempt:03d}").exists():
            attempt += 1
        destination = assert_safe_output_path(
            parent / f"attempt_{attempt:03d}"
        )
        paths = sorted(path for path in root.rglob("*") if path.is_file())
        records = [
            {
                "original_path": str(path.relative_to(REPO_ROOT)),
                "archived_path": str(
                    (destination / path.relative_to(root)).relative_to(REPO_ROOT)
                ),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in paths
        ]

        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(root, destination)
        for record in records:
            archived = REPO_ROOT / record["archived_path"]
            if (
                not archived.is_file()
                or sha256_file(archived) != record["sha256"]
            ):
                raise RuntimeError(
                    f"archived artifact hash mismatch: {archived}"
                )

        manifest_path = destination / "ARCHIVE_MANIFEST.json"
        safe_write_json(
            manifest_path,
            {
                "schema_version": "1.0.0",
                "status": "archived_hash_verified",
                "correction_id": (
                    "implementation_correction_06_retry_idempotency"
                ),
                "correction_sha256": correction_sha256,
                "job_id": job.job_id,
                "attempt": attempt,
                "records": records,
                "file_count": len(records),
                "byte_count": int(
                    sum(record["bytes"] for record in records)
                ),
                "all_original_paths_inactive": True,
                "all_archived_hashes_verified": True,
                "delete_performed": False,
                "contains_patient_rows": True,
                "publishable": False,
                "target_domain_information_used": False,
            },
        )
        safe_write_text(
            manifest_path.with_suffix(".json.sha256"),
            (
                f"{sha256_file(manifest_path)}  "
                f"{manifest_path.relative_to(REPO_ROOT)}\n"
            ),
        )
        return {
            "status": "archived_hash_verified",
            "archive_root": str(destination),
            "manifest_path": str(manifest_path),
            "all_archived_hashes_verified": True,
        }

    def claim(self, job: LandmarkJob) -> Path:
        root = self.root_for(job)
        path = root / "RUNNING.lock"
        if path.is_file():
            self._assert_no_live_claim(root)
            raise RuntimeError(
                f"stale job claim must be archived before claim: {job.job_id}"
            )
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def complete(self, job: LandmarkJob, artifacts: Iterable[Path]) -> Path:
        root = self.root_for(job)
        path = root / "completion_private.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite completion: {job.job_id}")
        records = artifact_records(artifacts, relative_to=root)
        safe_write_json(path, {
            "schema_version": "1.0.0",
            "job_id": job.job_id,
            "status": "complete",
            "artifacts": records,
        })
        safe_write_text(
            path.with_suffix(".json.sha256"),
            f"{sha256_file(path)}  completion_private.json\n",
        )
        claim = root / "RUNNING.lock"
        if claim.exists():
            claim.unlink()
        return path


def training_config_for_job(config: dict[str, Any], job: LandmarkJob) -> dict[str, Any]:
    augmented = job.training_regime == "stress_augmented"
    augmentation = config["matched_augmentation"]
    if augmented and job.model not in MATCHED_MODELS:
        raise PermissionError(f"stress augmentation is not registered for {job.model}")
    return {
        "mamba": dict(config["training"]["mamba"]),
        "deep_baselines": dict(config["training"]["deep_baselines"]),
        "xgboost": dict(config["training"]["xgboost"]),
        "selection_and_calibration": {
            "fusion_simplex_step": float(config["selection"]["fusion_simplex_step"]),
        },
        "temporal_robust_stage2": {
            "enabled": augmented,
            "scenarios": [
                str(value) for value in augmentation["views"]
                if str(value) != "full_6h"
            ],
            "weights": dict(augmentation["group_weights"]),
            "max_train_rows_per_view": int(augmentation["max_train_rows_per_view"]),
            "max_validation_rows_per_view": int(augmentation["max_validation_rows_per_view"]),
            "max_total_rows": int(augmentation["max_total_rows"]),
            "minimum_free_cache_bytes": int(augmentation["minimum_free_cache_bytes"]),
            "selection_metric": str(augmentation["selection_metric"]),
            "target_information_used": False,
        },
        "training_regime": job.training_regime,
    }
