from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from landmark_protocol import REPO_ROOT, WORKSPACE_ROOT
from v13_guard import safe_write_json, safe_write_text, sha256_file


CORRECTION_ID = "implementation_correction_03_auxiliary_los_stability"
CONTRACT_VERSION = "normalized_bounded_log_los_v3"
LEGACY_CONTRACT_VERSION = "legacy_unbounded_log_los_v2"
LOG_LOS_BOUND = 8.0
SATURATION_RELATIVE_THRESHOLD = 0.999
DEGENERATE_RANGE_TOLERANCE = 1e-6
MAX_QUALIFICATION_SATURATION_FRACTION = 0.05
CORRECTION_PATH = (
    WORKSPACE_ROOT
    / "config"
    / "implementation_correction_03_auxiliary_los_stability.json"
)
DEEP_BASELINE_MODELS = (
    "GRU-D",
    "LSTM",
    "Transformer",
    "TCN",
    "SAnD",
    "STraTS",
    "InterpNet",
    "DuETT",
)


def bounded_log_los(
    raw_log_los: torch.Tensor,
    *,
    bound: float = LOG_LOS_BOUND,
) -> torch.Tensor:
    """Apply the correction-03 differentiable output contract.

    Scaling the tanh argument by the same bound keeps a unit derivative at the
    origin while making the represented log-LOS finite by construction.  This
    is part of the trained model, not an inference-time clip or coercion.
    """
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("log-LOS bound must be positive and finite")
    return float(bound) * torch.tanh(raw_log_los / float(bound))


def verify_implementation_correction_03(
    plan_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    path = CORRECTION_PATH
    hash_path = path.with_suffix(".json.sha256")
    if not path.is_file() or not hash_path.is_file():
        raise RuntimeError("implementation correction 03 or its sidecar is missing")
    expected_hash = hash_path.read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != expected_hash:
        raise RuntimeError("implementation correction 03 hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "correction_id": CORRECTION_ID,
        "status": "post_failure_correction_registered_before_corrected_training",
        "analysis_plan_sha256": plan_hash,
        "config_sha256": config_hash,
        "affected_models": list(DEEP_BASELINE_MODELS),
        "target_domain_information_used": False,
        "corrected_training_started_before_registration": False,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"invalid correction-03 binding: {field}")
    contract = payload.get("los_contract", {})
    if contract.get("version") != CONTRACT_VERSION:
        raise RuntimeError("correction-03 LOS contract version mismatch")
    if float(contract.get("log_los_bound", float("nan"))) != LOG_LOS_BOUND:
        raise RuntimeError("correction-03 log-LOS bound mismatch")
    for relative_path, expected in payload.get("code_sha256", {}).items():
        code_path = WORKSPACE_ROOT / relative_path
        if not code_path.is_file() or sha256_file(code_path) != expected:
            raise RuntimeError(
                f"correction-03 code binding mismatch: {relative_path}"
            )
    return {"record_sha256": expected_hash, **payload}


@dataclass
class _RunningStats:
    count: int = 0
    finite_count: int = 0
    nonfinite_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    total: float = 0.0
    total_squares: float = 0.0
    saturation_count: int = 0

    def update(
        self,
        values: torch.Tensor | float,
        *,
        saturation_bound: float | None = None,
    ) -> None:
        tensor = torch.as_tensor(values).detach().float().reshape(-1).cpu()
        self.count += int(tensor.numel())
        finite_mask = torch.isfinite(tensor)
        finite = tensor[finite_mask].double()
        self.finite_count += int(finite.numel())
        self.nonfinite_count += int((~finite_mask).sum().item())
        if finite.numel() == 0:
            return
        current_min = float(finite.min().item())
        current_max = float(finite.max().item())
        self.minimum = (
            current_min if self.minimum is None else min(self.minimum, current_min)
        )
        self.maximum = (
            current_max if self.maximum is None else max(self.maximum, current_max)
        )
        self.total += float(finite.sum().item())
        self.total_squares += float(torch.square(finite).sum().item())
        if saturation_bound is not None:
            threshold = float(saturation_bound) * SATURATION_RELATIVE_THRESHOLD
            self.saturation_count += int((finite.abs() >= threshold).sum().item())

    def summary(self) -> dict[str, Any]:
        mean = self.total / self.finite_count if self.finite_count else None
        if mean is None:
            standard_deviation = None
        else:
            variance = max(
                self.total_squares / self.finite_count - mean * mean,
                0.0,
            )
            standard_deviation = math.sqrt(variance)
        finite_range = (
            self.maximum - self.minimum
            if self.minimum is not None and self.maximum is not None
            else None
        )
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "minimum_finite": self.minimum,
            "maximum_finite": self.maximum,
            "mean_finite": mean,
            "standard_deviation_finite": standard_deviation,
            "range_finite": finite_range,
            "saturation_count": self.saturation_count,
            "saturation_fraction": (
                self.saturation_count / self.finite_count
                if self.finite_count else None
            ),
        }


class AggregateNumericalDiagnostics:
    """Persist aggregate-only numerical evidence, including failure snapshots."""

    def __init__(
        self,
        path: Path,
        *,
        job_id: str,
        model: str,
        seed: int,
        source_dataset: str,
        correction_sha256: str,
        contract_version: str = CONTRACT_VERSION,
    ) -> None:
        self.path = path
        self.job_id = job_id
        self.model = model
        self.seed = int(seed)
        self.source_dataset = source_dataset
        self.correction_sha256 = correction_sha256
        self.contract_version = contract_version
        self._stages: dict[str, dict[str, _RunningStats]] = {}
        self._epochs: list[dict[str, Any]] = []

    def update(
        self,
        stage: str,
        metric: str,
        values: torch.Tensor | float,
        *,
        saturation_bound: float | None = None,
    ) -> None:
        stage_metrics = self._stages.setdefault(stage, {})
        stats = stage_metrics.setdefault(metric, _RunningStats())
        stats.update(values, saturation_bound=saturation_bound)

    def record_epoch(self, payload: dict[str, Any]) -> None:
        allowed = {
            "epoch",
            "train_loss",
            "validation_auprc",
            "gradient_norm_max",
            "last_epoch_checkpoint_sha256",
        }
        if not set(payload).issubset(allowed):
            raise ValueError("epoch diagnostics contain an unregistered field")
        self._epochs.append(dict(payload))

    def payload(
        self,
        *,
        status: str,
        failure: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "correction_id": CORRECTION_ID,
            "correction_sha256": self.correction_sha256,
            "contract_version": self.contract_version,
            "log_los_bound": LOG_LOS_BOUND,
            "job_id": self.job_id,
            "model": self.model,
            "seed": self.seed,
            "source_dataset": self.source_dataset,
            "status": status,
            "failure": failure,
            "aggregate_only": True,
            "contains_identifiers": False,
            "contains_patient_rows": False,
            "contains_patient_level_predictions": False,
            "target_domain_information_used": False,
            "epochs": list(self._epochs),
            "stages": {
                stage: {
                    metric: stats.summary()
                    for metric, stats in sorted(metrics.items())
                }
                for stage, metrics in sorted(self._stages.items())
            },
        }

    def persist(self, *, status: str, failure: str | None = None) -> str:
        safe_write_json(self.path, self.payload(status=status, failure=failure))
        digest = sha256_file(self.path)
        safe_write_text(
            self.path.with_suffix(".json.sha256"),
            f"{digest}  {self.path.relative_to(REPO_ROOT)}\n",
        )
        return digest


def qualification_gate(
    diagnostics: dict[str, Any],
    *,
    require_final_splits: bool = True,
) -> dict[str, Any]:
    failures: list[str] = []
    stages = diagnostics.get("stages", {})
    if diagnostics.get("status") != "complete":
        failures.append("numerical diagnostics are not complete")
    for stage, metrics in stages.items():
        for metric, summary in metrics.items():
            if int(summary.get("nonfinite_count", 0)):
                failures.append(f"{stage}/{metric} contains non-finite values")
        bounded = metrics.get("bounded_log_los")
        if bounded is not None:
            minimum = bounded.get("minimum_finite")
            maximum = bounded.get("maximum_finite")
            if minimum is None or maximum is None:
                failures.append(f"{stage}/bounded_log_los has no finite values")
            elif minimum < -LOG_LOS_BOUND - 1e-6 or maximum > LOG_LOS_BOUND + 1e-6:
                failures.append(f"{stage}/bounded_log_los violates its bound")
            saturation = bounded.get("saturation_fraction")
            if saturation is not None and saturation > MAX_QUALIFICATION_SATURATION_FRACTION:
                failures.append(
                    f"{stage}/bounded_log_los saturation {saturation:.6f} exceeds "
                    f"{MAX_QUALIFICATION_SATURATION_FRACTION:.6f}"
                )
    required = ("final_validation", "final_test") if require_final_splits else ()
    for stage in required:
        metrics = stages.get(stage)
        if metrics is None:
            failures.append(f"missing required stage: {stage}")
            continue
        bounded = metrics.get("bounded_log_los", {})
        finite_range = bounded.get("range_finite")
        standard_deviation = bounded.get("standard_deviation_finite")
        if finite_range is None or finite_range <= DEGENERATE_RANGE_TOLERANCE:
            failures.append(f"{stage}/bounded_log_los is range-degenerate")
        if (
            standard_deviation is None
            or standard_deviation <= DEGENERATE_RANGE_TOLERANCE
        ):
            failures.append(f"{stage}/bounded_log_los is variance-degenerate")
    return {
        "pass": not failures,
        "failures": failures,
        "criteria": {
            "nonfinite_count": 0,
            "absolute_log_los_bound": LOG_LOS_BOUND,
            "maximum_saturation_fraction": MAX_QUALIFICATION_SATURATION_FRACTION,
            "minimum_final_split_range_exclusive": DEGENERATE_RANGE_TOLERANCE,
            "minimum_final_split_standard_deviation_exclusive": DEGENERATE_RANGE_TOLERANCE,
        },
        "target_domain_information_used": False,
    }
