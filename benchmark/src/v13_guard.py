from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "benchmark"
RESULTS_ROOT = REPO_ROOT / "results" / "benchmark"
PLAN_PATH = WORKSPACE_ROOT / "analysis_plan_v13_landmark_amendment_02.json"
PLAN_HASH_PATH = WORKSPACE_ROOT / "analysis_plan_v13_landmark_amendment_02.sha256"
# Compatibility aliases for the inherited v13 modules.  The fixed-landmark
# amendment is the only active protocol in this isolated workspace.
DATA_PROVENANCE_AMENDMENT_PATH = PLAN_PATH
DATA_PROVENANCE_AMENDMENT_HASH_PATH = PLAN_HASH_PATH
MODEL_PROTOCOL_AMENDMENT_PATH = PLAN_PATH
MODEL_PROTOCOL_AMENDMENT_HASH_PATH = PLAN_HASH_PATH
PROTECTED_ROOTS = (
    REPO_ROOT / "checkpoints_pre_v2",
    REPO_ROOT / "results" / "frozen_pre_v2",
    REPO_ROOT / "maboost_github_release" / "github_reproducibility_v11",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_safe_output_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if any(_is_relative_to(resolved, root.resolve(strict=False)) for root in PROTECTED_ROOTS):
        raise PermissionError(f"v13 write rejected for protected v11 path: {resolved}")
    allowed = (WORKSPACE_ROOT.resolve(strict=False), RESULTS_ROOT.resolve(strict=False))
    if not any(_is_relative_to(resolved, root) for root in allowed):
        raise PermissionError(f"v13 output must remain in an isolated v13 root: {resolved}")
    return resolved


def safe_write_bytes(path: Path, payload: bytes) -> None:
    target = assert_safe_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        temp_path = Path(temp_name)
        if temp_path.exists():
            temp_path.unlink()


def _json_default(value: Any) -> Any:
    # Pandas reductions return NumPy booleans, which the stdlib encoder
    # rejects. Preserve their boolean type; unsupported objects still fail.
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def safe_write_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n").encode("utf-8")
    safe_write_bytes(path, encoded)


def safe_write_text(path: Path, payload: str) -> None:
    safe_write_bytes(path, payload.encode("utf-8"))


def load_and_verify_plan() -> tuple[dict[str, Any], str]:
    expected = PLAN_HASH_PATH.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(PLAN_PATH)
    if observed != expected:
        raise RuntimeError(
            f"analysis plan hash mismatch: expected={expected}, observed={observed}"
        )
    return json.loads(PLAN_PATH.read_text(encoding="utf-8")), observed


def load_and_verify_data_provenance_amendment(
    analysis_plan_sha256: str,
) -> tuple[dict[str, Any], str]:
    expected = DATA_PROVENANCE_AMENDMENT_HASH_PATH.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(DATA_PROVENANCE_AMENDMENT_PATH)
    if observed != expected:
        raise RuntimeError(
            "data provenance amendment hash mismatch: "
            f"expected={expected}, observed={observed}"
        )
    payload = json.loads(DATA_PROVENANCE_AMENDMENT_PATH.read_text(encoding="utf-8"))
    if observed != analysis_plan_sha256:
        raise RuntimeError("active landmark amendment hash changed during execution")
    return payload, observed


def load_and_verify_model_protocol_amendment(
    analysis_plan_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify the prospective, append-only model-protocol clarification.

    The registered analysis plan is never rewritten.  Material clarifications
    made before the first full training job are recorded in a separately hashed
    amendment and all downstream configs bind to that hash.
    """
    expected = MODEL_PROTOCOL_AMENDMENT_HASH_PATH.read_text(
        encoding="utf-8"
    ).split()[0]
    observed = sha256_file(MODEL_PROTOCOL_AMENDMENT_PATH)
    if observed != expected:
        raise RuntimeError(
            "model protocol amendment hash mismatch: "
            f"expected={expected}, observed={observed}"
        )
    payload = json.loads(MODEL_PROTOCOL_AMENDMENT_PATH.read_text(encoding="utf-8"))
    if observed != analysis_plan_sha256:
        raise RuntimeError("active landmark amendment hash changed during execution")
    if payload.get("status") != (
        "public_reproduction_recipe"
    ):
        raise RuntimeError("landmark amendment registration status is invalid")
    return payload, observed


def snapshot_tree(root: Path, include_files: bool = False) -> dict[str, Any]:
    resolved = root.resolve(strict=False)
    if not resolved.exists():
        return {
            "exists": False,
            "file_count": 0,
            "byte_count": 0,
            "tree_sha256": None,
            **({"files": []} if include_files else {}),
        }

    tree_hash = hashlib.sha256()
    files: list[dict[str, Any]] = []
    byte_count = 0
    paths = sorted(path for path in resolved.rglob("*") if path.is_file())
    for path in paths:
        relative = path.relative_to(resolved).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        tree_hash.update(relative.encode("utf-8"))
        tree_hash.update(b"\0")
        tree_hash.update(file_hash.encode("ascii"))
        tree_hash.update(b"\0")
        byte_count += size
        if include_files:
            files.append({"path": relative, "bytes": size, "sha256": file_hash})
    result: dict[str, Any] = {
        "exists": True,
        "file_count": len(paths),
        "byte_count": byte_count,
        "tree_sha256": tree_hash.hexdigest(),
    }
    if include_files:
        result["files"] = files
    return result


def verify_frozen_v11(plan: dict[str, Any], include_files: bool = False) -> dict[str, Any]:
    expected_roots = plan["protected_v11"]["roots"]
    observed: dict[str, Any] = {}
    for relative, expected in expected_roots.items():
        current = snapshot_tree(REPO_ROOT / relative, include_files=include_files)
        comparable = {key: current[key] for key in ("exists", "file_count", "byte_count", "tree_sha256")}
        if comparable != expected:
            raise RuntimeError(
                f"frozen v11 mismatch at {relative}: expected={expected}, observed={comparable}"
            )
        observed[relative] = current
    return observed


@dataclass(frozen=True)
class ReverseTransferFirewall:
    source_dataset: str = "eicu"
    target_datasets: tuple[str, ...] = ("mimic_iv", "mimic_iii_carevue")

    def authorize(
        self,
        *,
        dataset: str,
        artifact_role: str,
        stage: str,
        predictions_locked: bool = False,
        source_choices_locked: bool = False,
    ) -> None:
        dataset = dataset.lower()
        artifact_role = artifact_role.lower()
        stage = stage.lower()
        target = dataset in self.target_datasets
        target_sensitive_roles = {
            "labels",
            "prevalence",
            "calibration",
            "feature_selection",
            "fusion_selection",
            "threshold_selection",
        }
        if target and artifact_role in target_sensitive_roles:
            allowed_eval = (
                artifact_role == "labels"
                and stage == "evaluation_metrics"
                and predictions_locked
                and source_choices_locked
            )
            if not allowed_eval:
                raise PermissionError(
                    "reverse-transfer firewall rejected target information before locked evaluation"
                )


def require_no_identifier_fields(fields: Iterable[str]) -> None:
    forbidden = {
        "subject_id",
        "stay_id",
        "hadm_id",
        "icustay_id",
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
    }
    overlap = forbidden.intersection(field.lower() for field in fields)
    if overlap:
        raise ValueError(f"patient-level identifiers are forbidden in aggregate artifacts: {sorted(overlap)}")
