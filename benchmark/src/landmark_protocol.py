from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from v13_guard import (
    PLAN_HASH_PATH,
    PLAN_PATH,
    REPO_ROOT,
    RESULTS_ROOT,
    WORKSPACE_ROOT,
    safe_write_json,
    safe_write_text,
    sha256_file,
    snapshot_tree,
)


CONFIG_PATH = WORKSPACE_ROOT / "config" / "landmark_v13.json"
CONFIG_HASH_PATH = WORKSPACE_ROOT / "config" / "landmark_v13.sha256"
PHASE_REPORT_ROOT = RESULTS_ROOT / "phase_reports"

REQUIRED_RAW_FILES: dict[str, tuple[str, ...]] = {
    "mimic_iv": (
        "icustays.csv.gz",
        "admissions.csv.gz",
        "patients.csv.gz",
        "chartevents.csv.gz",
        "labevents.csv.gz",
        "outputevents.csv.gz",
    ),
    "eicu": (
        "patient.csv.gz",
        "vitalPeriodic.csv.gz",
        "vitalAperiodic.csv.gz",
        "nurseCharting.csv.gz",
        "lab.csv.gz",
        "respiratoryCharting.csv.gz",
        "intakeOutput.csv.gz",
    ),
    "mimic_iii_carevue": (
        "ICUSTAYS.csv.gz",
        "ADMISSIONS.csv.gz",
        "PATIENTS.csv.gz",
        "CHARTEVENTS.csv.gz",
        "LABEVENTS.csv.gz",
        "OUTPUTEVENTS.csv.gz",
    ),
}

OFFICIAL_CAREVUE_RESOURCE = {
    "title": "MIMIC-III Clinical Database CareVue subset",
    "version": "1.4",
    "published": "2022-09-21",
    "content_url": "https://physionet.org/content/mimic3-carevue/1.4/",
    "version_doi": "10.13026/8a4q-w170",
    "non_overlap_method": (
        "CareVue dbsource stays followed by removal of subjects admitted during "
        "the MIMIC-IV v2.0 collection period"
    ),
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_expected_hash(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").split()[0]
    except (FileNotFoundError, IndexError) as error:
        raise FileNotFoundError(f"missing hash lock: {path}") from error


def load_protocol() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    plan_expected = _read_expected_hash(PLAN_HASH_PATH)
    plan_observed = sha256_file(PLAN_PATH)
    if plan_expected != plan_observed:
        raise RuntimeError(
            f"landmark amendment hash mismatch: {plan_expected} != {plan_observed}"
        )
    config_expected = _read_expected_hash(CONFIG_HASH_PATH)
    config_observed = sha256_file(CONFIG_PATH)
    if config_expected != config_observed:
        raise RuntimeError(
            f"landmark config hash mismatch: {config_expected} != {config_observed}"
        )
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("analysis_plan_sha256") != plan_observed:
        raise RuntimeError("landmark config is not bound to the active amendment")
    if config.get("gates", {}).get("manuscript_modification") is not False:
        raise PermissionError("computational config must keep manuscript gate closed")
    return plan, plan_observed, config, config_observed


def verify_frozen_state(plan):
    """Compatibility name: public runs protect code/configuration, not author v11 files."""
    from reproduction_contract import verify_code
    return verify_code()


def raw_file_probe(path: Path, *, deep_hash: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return record
    record["bytes"] = int(path.stat().st_size)
    record["mtime_ns"] = int(path.stat().st_mtime_ns)
    try:
        with gzip.open(path, "rb") as handle:
            header_sample = handle.read(65536)
        record["gzip_header_and_sample_ok"] = True
        record["decompressed_probe_bytes"] = len(header_sample)
        record["csv_header_present"] = b"," in header_sample.splitlines()[0]
    except (OSError, EOFError) as error:
        record["gzip_header_and_sample_ok"] = False
        record["gzip_error"] = f"{type(error).__name__}: {error}"
    if deep_hash:
        record["sha256"] = sha256_file(path)
    return record


def raw_manifest(config: dict[str, Any], *, deep_hash: bool = False) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset, required in REQUIRED_RAW_FILES.items():
        raw_root = REPO_ROOT / config["datasets"][dataset]["raw_root"]
        files = [raw_file_probe(raw_root / name, deep_hash=deep_hash) for name in required]
        datasets[dataset] = {
            "release": config["datasets"][dataset]["release"],
            "raw_root": str(raw_root.relative_to(REPO_ROOT)),
            "root_exists": raw_root.is_dir(),
            "required_files": files,
            "complete": bool(
                raw_root.is_dir()
                and all(
                    item.get("exists")
                    and item.get("gzip_header_and_sample_ok")
                    and item.get("csv_header_present")
                    for item in files
                )
            ),
        }
    return {
        "schema_version": "1.0.0",
        "created_at_utc": now_utc(),
        "deep_hash": bool(deep_hash),
        "datasets": datasets,
        "all_required_datasets_complete": all(
            item["complete"] for item in datasets.values()
        ),
        "contains_patient_rows": False,
        "contains_identifiers": False,
    }


def carevue_official_identity_probe(config: dict[str, Any]) -> dict[str, Any]:
    """Check local content invariants documented for the official v1.4 subset.

    The credentialed PhysioNet file endpoint does not expose publisher checksums
    to an unauthenticated preflight.  The gate therefore binds the exact official
    version URL/DOI and verifies the key content invariant available in the
    downloaded tables: every ICU stay is a CareVue stay.  No identifier or row is
    copied into the report.
    """
    dataset = config["datasets"]["mimic_iii_carevue"]
    raw_root = REPO_ROOT / dataset["raw_root"]
    path = raw_root / "ICUSTAYS.csv.gz"
    result: dict[str, Any] = {
        "resource": OFFICIAL_CAREVUE_RESOURCE,
        "configured_release": str(dataset["release"]),
        "configured_raw_root": str(raw_root.relative_to(REPO_ROOT)),
        "icustays_present": path.is_file(),
        "icu_stay_rows": 0,
        "dbsource_values": [],
        "all_stays_are_carevue": False,
        "publisher_checksum_available_to_preflight": False,
        "contains_identifiers_or_patient_rows": False,
        "verification_basis": (
            "locked official URL/DOI/version plus aggregate DBSOURCE invariant"
        ),
    }
    if not path.is_file():
        result["verified"] = False
        result["error"] = "ICUSTAYS.csv.gz is missing"
        return result
    try:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            field_lookup = {
                str(name).strip().lower(): str(name) for name in (reader.fieldnames or [])
            }
            dbsource_field = field_lookup.get("dbsource")
            if dbsource_field is None:
                raise ValueError("ICUSTAYS is missing DBSOURCE")
            sources: set[str] = set()
            count = 0
            for row in reader:
                count += 1
                sources.add(str(row.get(dbsource_field, "")).strip().lower())
        result["icu_stay_rows"] = int(count)
        result["dbsource_values"] = sorted(sources)
        result["all_stays_are_carevue"] = bool(count and sources == {"carevue"})
        result["verified"] = bool(
            str(dataset["release"]) == OFFICIAL_CAREVUE_RESOURCE["version"]
            and raw_root.relative_to(REPO_ROOT).as_posix()
            == "data/raw/mimic3-carevue/1.4"
            and result["all_stays_are_carevue"]
        )
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        result["verified"] = False
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def atomic_save_npy(path: Path, array: Any) -> None:
    import numpy as np

    from v13_guard import assert_safe_output_path

    path = assert_safe_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        temp = Path(temp_name)
        if temp.exists():
            temp.unlink()


def artifact_records(paths: Iterable[Path], *, relative_to: Path) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        records.append({
            "file": path.relative_to(relative_to).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return records


def write_phase_report(phase: str, payload: dict[str, Any]) -> tuple[Path, str]:
    report = {
        "schema_version": "1.0.0",
        "phase": phase,
        "created_at_utc": now_utc(),
        "manuscript_modified": False,
        **payload,
    }
    path = PHASE_REPORT_ROOT / f"PHASE_REPORT_{phase}.json"
    safe_write_json(path, report)
    digest = sha256_file(path)
    safe_write_text(
        path.with_suffix(".json.sha256"),
        f"{digest}  {path.relative_to(REPO_ROOT)}\n",
    )
    md = [
        f"# Phase report: {phase}",
        "",
        f"- Status: `{report.get('status', 'unknown')}`",
        f"- Analysis plan SHA-256: `{report.get('analysis_plan_sha256', 'n/a')}`",
        f"- Config SHA-256: `{report.get('config_sha256', 'n/a')}`",
        f"- Command: `{report.get('command', 'n/a')}`",
        f"- Manuscript modified: `{report.get('manuscript_modified', False)}`",
        "",
        "## Tests",
        "",
    ]
    tests = report.get("tests", [])
    md.extend(
        f"- [{'x' if item.get('passed') else ' '}] {item.get('name')}: "
        f"{item.get('detail', '')}"
        for item in tests
    )
    md.extend(["", "## Runtime and resource use", ""])
    runtime = report.get("runtime", {})
    if runtime:
        md.append("```json")
        md.append(json.dumps(runtime, indent=2, sort_keys=True))
        md.append("```")
    else:
        md.append("- No runtime-intensive stage was executed.")
    md.extend(["", "## Files and artifact hashes", ""])
    artifact_summary = {
        key: report[key]
        for key in (
            "raw_manifest", "raw_manifest_sha256", "files", "figures", "tables",
            "datasets", "jobs", "concordance_report", "canonical_report",
            "explainability_report",
        )
        if key in report
    }
    if artifact_summary:
        md.append("```json")
        md.append(json.dumps(artifact_summary, indent=2, sort_keys=True, default=str))
        md.append("```")
    else:
        md.append("- See the paired JSON phase report for the verified artifact list.")
    md.extend(["", "## Remaining errors", ""])
    errors = report.get("errors", [])
    md.extend(f"- {error}" for error in errors)
    if not errors:
        md.append("- None")
    md_path = PHASE_REPORT_ROOT / f"PHASE_REPORT_{phase}.md"
    safe_write_text(md_path, "\n".join(md) + "\n")
    return path, digest


def load_verified_phase_report(
    phase: str,
    *,
    allowed_statuses: set[str] | None = None,
) -> dict[str, Any]:
    path = PHASE_REPORT_ROOT / f"PHASE_REPORT_{phase}.json"
    hash_path = path.with_suffix(".json.sha256")
    if not path.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"verified phase report is missing: {phase}")
    observed = sha256_file(path)
    expected = hash_path.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise RuntimeError(f"phase report hash mismatch: {phase}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("phase") != phase:
        raise RuntimeError(f"phase report identity mismatch: {phase}")
    plan_hash = _read_expected_hash(PLAN_HASH_PATH)
    config_hash = _read_expected_hash(CONFIG_HASH_PATH)
    for field, locked in (
        ("analysis_plan_sha256", plan_hash),
        ("config_sha256", config_hash),
    ):
        if field in payload and payload[field] != locked:
            raise RuntimeError(f"phase report protocol binding mismatch: {phase}/{field}")
    if allowed_statuses is not None and payload.get("status") not in allowed_statuses:
        raise PermissionError(
            f"phase {phase} status {payload.get('status')!r} is outside "
            f"the allowed gate states {sorted(allowed_statuses)}"
        )
    return payload
