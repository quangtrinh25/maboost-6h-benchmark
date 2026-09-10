from __future__ import annotations


import hashlib


import json


import os


from pathlib import Path


from typing import Any


from landmark_protocol import (
    REPO_ROOT,
    RESULTS_ROOT,
    WORKSPACE_ROOT,
    load_protocol,
    load_verified_phase_report,
)


from v13_guard import sha256_file


CORRECTION_ID = "implementation_correction_07_analysis_engine"


RUST_ABI = "maboost.analysis.bootstrap.v1"


RUST_ABI_VERSION = 1


METRIC_CODES = {
    "auroc": 1,
    "auprc": 2,
    "brier": 3,
    "los_mae": 4,
    "los_rmse": 5,
}


CORRECTION_PATH = (
    WORKSPACE_ROOT / "config" / "implementation_correction_07_analysis_engine.json"
)


QUALIFICATION_PATH = (
    RESULTS_ROOT / "qualification" / "CORRECTION_07_SOURCE_INTERNAL_QUALIFICATION.json"
)


AUDIT_PATH = (
    RESULTS_ROOT / "diagnostics" / "CORRECTION_07_AFFECTED_ARTIFACT_AUDIT.json"
)


APPROVAL_PATH = (
    WORKSPACE_ROOT / "config" / "analysis_launch_approval_v13_landmark_correction_07.json"
)


CORRECTION07_CANONICAL_DATA_NAMES = frozenset(
    {
        "per_seed_metrics.csv",
        "aggregate_metrics.csv",
        "primary_contrasts.csv",
        "seed_level_bootstrap_forest.csv",
        "cohort_characteristics.csv",
        "measurement_shift_smd.csv",
        "cohort_flow.csv",
        "subgroups.csv",
        "model_registry.csv",
        "inference_counts.csv",
    }
)


CORRECTION07_PHASE_F_FILE_NAMES = frozenset(
    {
        "shap_feature_registry.csv",
        "shap_importance_per_seed.csv",
        "shap_importance_aggregate.csv",
        "fusion_pathway_contributions.csv",
        "temporal_feature_occlusion.csv",
        "table_7_shap_pathway_summary.csv",
    }
)


CORRECTION07_PHASE_F_FIGURE_NAMES = frozenset(
    {"figure_5_shap_pathways_occlusion.png", "figure_5_shap_pathways_occlusion.pdf"}
)


CORRECTION07_PHASE_G_FIGURE_NAMES = frozenset(
    {
        "figure_1_landmark_cohort_flow.pdf",
        "figure_2_architecture_landmark_v13.pdf",
        "figure_3_stress_group_effect_coverage.pdf",
        "figure_4_reliability_calibration.pdf",
        "figure_6_seed_level_bootstrap_forest.pdf",
    }
)


CORRECTION07_PHASE_G_TABLE_PATHS = {
    "table_1.csv": "table_1_cohort_shift.csv",
    "table_2.csv": "table_2_model_registry.csv",
    "table_3A.csv": "table_3A_mortality.csv",
    "table_3B.csv": "table_3B_mortality.csv",
    **{f"table_{number}.csv": f"table_{number}.csv" for number in (4, 5, 6, 7, 8)},
    **{
        name: name
        for name in (
            "supplement_carevue.csv",
            "supplement_stress_dictionary.csv",
            "supplement_subgroups.csv",
            "supplement_complete_stress.csv",
            "reliability_bins.csv",
        )
    },
    **{
        f"table_{number}.tex": f"table_{number}.tex"
        for number in ("1", "2", "3A", "3B", "4", "5", "6", "7", "8")
    },
}


def bounded_thread_count(
    correction: dict[str, Any], *, available_cpus: int | None = None
) -> int:
    """Apply the one registered thread rule in every execution route."""

    if available_cpus is None:
        try:
            cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cpus = os.cpu_count()
    else:
        cpus = int(available_cpus)
    cpus = max(1, int(cpus or 1))
    return min(int(correction["max_threads"]), max(1, cpus - 2))


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_sha256_sidecar(path: Path) -> str:
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"hash-locked artifact unavailable: {path}")
    fields = sidecar.read_text(encoding="utf-8").split()
    if len(fields) < 2:
        raise RuntimeError(f"malformed SHA sidecar: {sidecar}")
    observed = sha256_file(path)
    try:
        expected_path = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError as exc:
        raise PermissionError(f"artifact is outside repository: {path}") from exc
    if fields[0] != observed or fields[1] != expected_path:
        raise RuntimeError(f"SHA sidecar mismatch: {path}")
    return observed
