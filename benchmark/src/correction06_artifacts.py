from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from landmark_protocol import REPO_ROOT, RESULTS_ROOT, WORKSPACE_ROOT
from v13_guard import (
    assert_safe_output_path,
    safe_write_json,
    safe_write_text,
    sha256_file,
)


CORRECTION_ID = "implementation_correction_06_retry_idempotency"
FAILED_JOB_ID = "eicu__mamba_z_only__full_only__seed2025"
ATTEMPT_ARCHIVE_ROOT = (
    WORKSPACE_ROOT / "superseded" / CORRECTION_ID / "job_attempts"
)
INITIAL_ARCHIVE_SUMMARY = (
    RESULTS_ROOT / "diagnostics" / "CORRECTION_06_INITIAL_ARCHIVE_SUMMARY.json"
)


def _first_unused_path(parent: Path, pattern: str) -> Path:
    index = 1
    while (parent / pattern.format(index=index)).exists():
        index += 1
    return parent / pattern.format(index=index)


def archive_incomplete_job_log(
    log_path: Path,
    *,
    job_id: str,
    correction_sha256: str,
    archive_root: Path | None = None,
) -> dict[str, Any] | None:
    """Atomically archive and hash an existing attempt log."""
    source = assert_safe_output_path(log_path)
    if not source.exists():
        return None
    if not source.is_file():
        raise RuntimeError(f"job log is not a regular file: {source}")
    destination_root = assert_safe_output_path(
        archive_root
        if archive_root is not None
        else ATTEMPT_ARCHIVE_ROOT / job_id / "logs"
    )
    destination = assert_safe_output_path(
        _first_unused_path(destination_root, "log_attempt_{index:03d}.log")
    )
    source_hash = sha256_file(source)
    source_bytes = int(source.stat().st_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    if (
        not destination.is_file()
        or sha256_file(destination) != source_hash
        or int(destination.stat().st_size) != source_bytes
    ):
        raise RuntimeError(f"archived log integrity failure: {destination}")
    sidecar = destination.with_suffix(".log.sha256")
    safe_write_text(
        sidecar,
        f"{source_hash}  {destination.relative_to(REPO_ROOT)}\n",
    )
    return {
        "status": "archived_hash_verified",
        "correction_id": CORRECTION_ID,
        "correction_sha256": correction_sha256,
        "job_id": job_id,
        "original_path": str(source.relative_to(REPO_ROOT)),
        "archived_path": str(destination),
        "archived_relative_path": str(destination.relative_to(REPO_ROOT)),
        "sha256": source_hash,
        "bytes": source_bytes,
        "sidecar": str(sidecar.relative_to(REPO_ROOT)),
        "delete_performed": False,
        "target_domain_information_used": False,
    }


def archive_incomplete_public_job_report(
    report_path: Path,
    *,
    job_id: str,
    correction_sha256: str,
    archive_root: Path | None = None,
) -> dict[str, Any] | None:
    """Archive a report written by an attempt that never completed."""
    source = assert_safe_output_path(report_path)
    sidecar = source.with_suffix(".json.sha256")
    sources = [path for path in (source, sidecar) if path.exists()]
    if not sources:
        return None
    if any(not path.is_file() for path in sources):
        raise RuntimeError(f"incomplete public report is not regular: {source}")
    destination_parent = assert_safe_output_path(
        archive_root
        if archive_root is not None
        else ATTEMPT_ARCHIVE_ROOT / job_id / "public_reports"
    )
    destination = assert_safe_output_path(
        _first_unused_path(
            destination_parent, "public_report_attempt_{index:03d}"
        )
    )
    records = [
        {
            "original_path": str(path.relative_to(REPO_ROOT)),
            "archived_path": str((destination / path.name).relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for path in sources
    ]
    destination.mkdir(parents=True, exist_ok=False)
    for path in sources:
        os.replace(path, destination / path.name)
    for record in records:
        archived = REPO_ROOT / record["archived_path"]
        if not archived.is_file() or sha256_file(archived) != record["sha256"]:
            raise RuntimeError(f"archived public report hash mismatch: {archived}")
    manifest = destination / "ARCHIVE_MANIFEST.json"
    safe_write_json(
        manifest,
        {
            "schema_version": "1.0.0",
            "status": "archived_hash_verified",
            "correction_id": CORRECTION_ID,
            "correction_sha256": correction_sha256,
            "job_id": job_id,
            "records": records,
            "file_count": len(records),
            "byte_count": int(sum(record["bytes"] for record in records)),
            "all_archived_hashes_verified": True,
            "delete_performed": False,
            "contains_patient_rows": False,
            "publishable": False,
            "target_domain_information_used": False,
        },
    )
    safe_write_text(
        manifest.with_suffix(".json.sha256"),
        f"{sha256_file(manifest)}  {manifest.relative_to(REPO_ROOT)}\n",
    )
    return {
        "status": "archived_hash_verified",
        "archive_root": str(destination),
        "manifest_path": str(manifest),
        "file_count": len(records),
    }


def verify_attempt_archive_manifest(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"attempt archive manifest or sidecar is missing: {path}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != expected:
        raise RuntimeError(f"attempt archive manifest hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "archived_hash_verified",
        "correction_id": CORRECTION_ID,
        "all_archived_hashes_verified": True,
        "delete_performed": False,
        "publishable": False,
        "target_domain_information_used": False,
    }
    for field, value in required.items():
        if payload.get(field) != value:
            raise RuntimeError(f"invalid attempt archive binding: {field}")
    records = payload.get("records", [])
    if payload.get("file_count") != len(records):
        raise RuntimeError(f"attempt archive file count mismatch: {path}")
    byte_count = 0
    for record in records:
        archived = REPO_ROOT / record["archived_path"]
        if not archived.is_file() or sha256_file(archived) != record["sha256"]:
            raise RuntimeError(f"archived artifact hash mismatch: {archived}")
        if int(archived.stat().st_size) != int(record["bytes"]):
            raise RuntimeError(f"archived artifact size mismatch: {archived}")
        byte_count += int(record["bytes"])
    if payload.get("byte_count") != byte_count:
        raise RuntimeError(f"attempt archive byte count mismatch: {path}")
    return {"manifest_sha256": expected, **payload}


def verify_archived_log(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".log.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"archived log or sidecar is missing: {path}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != expected:
        raise RuntimeError(f"archived log hash mismatch: {path}")
    return {
        "file": str(path.relative_to(REPO_ROOT)),
        "sha256": expected,
        "bytes": int(path.stat().st_size),
    }


def audit_attempt_archives(*, require_initial_attempt: bool = True) -> dict[str, Any]:
    manifests = sorted(ATTEMPT_ARCHIVE_ROOT.glob("*/attempt_*/ARCHIVE_MANIFEST.json"))
    if require_initial_attempt and not manifests:
        raise RuntimeError("correction-06 attempt archive is missing")
    verified_manifests = [verify_attempt_archive_manifest(path) for path in manifests]
    logs = sorted(ATTEMPT_ARCHIVE_ROOT.glob("*/logs/log_attempt_*.log"))
    verified_logs = [verify_archived_log(path) for path in logs]
    public_manifests = sorted(
        ATTEMPT_ARCHIVE_ROOT.glob(
            "*/public_reports/public_report_attempt_*/ARCHIVE_MANIFEST.json"
        )
    )
    verified_public_manifests = [
        verify_attempt_archive_manifest(path) for path in public_manifests
    ]
    return {
        "status": "pass",
        "correction_id": CORRECTION_ID,
        "attempt_manifests_verified": len(verified_manifests),
        "archived_job_files_verified": int(
            sum(item["file_count"] for item in verified_manifests)
        ),
        "archived_job_bytes_verified": int(
            sum(item["byte_count"] for item in verified_manifests)
        ),
        "archived_logs_verified": len(verified_logs),
        "archived_incomplete_public_reports_verified": len(
            verified_public_manifests
        ),
        "manifest_sha256": [
            item["manifest_sha256"] for item in verified_manifests
        ],
        "log_records": verified_logs,
        "all_archived_hashes_verified": True,
        "mixed_pre_post_correction_artifacts": False,
        "delete_performed": False,
        "target_domain_information_used": False,
    }
