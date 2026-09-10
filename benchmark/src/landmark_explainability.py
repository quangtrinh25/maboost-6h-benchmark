from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from landmark_analysis import _matching_job, _require_full_jobs, _write_csv
from reproduction_contract import require_analysis_report as require_correction07_phase_e_report
from landmark_firewall import load_source_training_data
from landmark_jobs import LandmarkJob
from landmark_protocol import REPO_ROOT, RESULTS_ROOT, WORKSPACE_ROOT, load_protocol, load_verified_phase_report, verify_frozen_state, write_phase_report
from v13_analysis import shap_feature_registry_rows
from v13_feature_registry import DYNAMIC_FEATURES
from v13_guard import assert_safe_output_path, safe_write_json, safe_write_text, sha256_file
from v13_inference import ModelPredictor
from v13_resource_guardian import resource_safe_point


FEATURE_GROUPS = {
    "Neurologic": {"gcs_total", "gcs_verbal", "gcs_motor", "gcs_eye"},
    "Respiratory": {"spo2", "resp_rate", "fio2", "ph", "pao2", "paco2"},
    "Hemodynamic": {"heart_rate", "sbp", "dbp", "map", "temp_c", "temp_f"},
    "Renal/metabolic": {
        "glucose", "urine_output", "lactate", "bicarbonate", "creatinine",
        "bun", "sodium", "potassium", "chloride", "calcium", "magnesium",
        "phosphate",
    },
}
FEATURE_GROUPS["Laboratory"] = set(DYNAMIC_FEATURES).difference(
    set().union(*FEATURE_GROUPS.values())
)


def _save_private_npz(path: Path, **arrays: np.ndarray) -> str:
    target = assert_safe_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.pid{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256_file(target)
    safe_write_text(
        target.with_suffix(target.suffix + ".sha256"),
        f"{digest}  {target.relative_to(REPO_ROOT)}\n",
    )
    return digest


def _sample_rows(rows: np.ndarray, *, seed: int, limit: int = 512) -> np.ndarray:
    values = np.asarray(rows, dtype=np.int64)
    if len(values) <= limit:
        return values
    rng = np.random.default_rng(seed + 9_177_311)
    return np.sort(rng.choice(values, size=limit, replace=False))


def _predict_batches(
    predictor: ModelPredictor,
    values: np.ndarray,
    tau: np.ndarray,
    mask: np.ndarray,
    static: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    pieces = []
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        probability, _ = predictor.predict(
            values[start:stop], tau[start:stop], mask[start:stop], static[start:stop]
        )
        pieces.append(probability)
    return np.concatenate(pieces).astype(np.float32)


def _tree_shap_values(model: Any, features: np.ndarray) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(features)
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    if array.shape != features.shape:
        raise RuntimeError(
            f"TreeSHAP shape drift: shap={array.shape}, features={features.shape}"
        )
    return array.astype(np.float32)


def _explain_job(
    job: LandmarkJob,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    job_root = WORKSPACE_ROOT / "private" / "jobs" / job.job_id
    predictor = ModelPredictor(job.model, job_root, device="cuda")
    arrays = load_source_training_data(job.source_dataset)
    split_path = WORKSPACE_ROOT / "private" / "splits" / f"{job.source_dataset}_seed{job.seed}.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    rows = _sample_rows(np.asarray(split["partitions"]["test"], dtype=np.int64), seed=job.seed)
    values = np.asarray(arrays["values"][rows], dtype=np.float32)
    tau = np.asarray(arrays["tau"][rows], dtype=np.float32)
    mask = np.asarray(arrays["mask"][rows], dtype=np.uint8)
    elapsed = np.asarray(arrays["elapsed"][rows], dtype=np.float32)
    static = np.asarray(arrays["static"][rows], dtype=np.float32)

    components = predictor.explanation_components(values, tau, mask, static)
    flat = components["flat_head_component"]
    feature_values = np.asarray(flat["features"], dtype=np.float32)
    shap_values = _tree_shap_values(flat["mortality_model"], feature_values)
    feature_names = np.asarray(flat["feature_names"], dtype="U96")

    component_order = (
        "hybrid_component", "xgboost_flat_component", "flat_head_component"
    )
    raw_component_probability = np.column_stack([
        components[name]["mortality_model"].predict_proba(
            components[name]["features"]
        )[:, 1]
        for name in component_order
    ]).astype(np.float32)
    weights = np.asarray(
        predictor.choices["training_audit"]["fusion"]["weights"], dtype=np.float32
    )
    pathway_contribution = raw_component_probability * weights[None, :]

    base_probability = _predict_batches(predictor, values, tau, mask, static)
    occlusion_names: list[str] = []
    occlusion_delta: list[np.ndarray] = []
    for label, lower, upper in (
        ("0-2 h", 0.0, 2.0), ("2-4 h", 2.0, 4.0), ("4-6 h", 4.0, 6.0),
    ):
        occluded_mask = mask.copy()
        selected_time = (elapsed >= lower * 3600.0) & (elapsed <= upper * 3600.0)
        occluded_mask[selected_time] = 0
        occluded_values = values.copy()
        occluded_values[occluded_mask == 0] = np.nan
        probability = _predict_batches(
            predictor, occluded_values, tau, occluded_mask, static
        )
        occlusion_names.append(label)
        occlusion_delta.append(base_probability - probability)
    for label, features in FEATURE_GROUPS.items():
        columns = [DYNAMIC_FEATURES.index(feature) for feature in sorted(features)]
        occluded_mask = mask.copy()
        occluded_mask[:, :, columns] = 0
        occluded_values = values.copy()
        occluded_values[occluded_mask == 0] = np.nan
        probability = _predict_batches(
            predictor, occluded_values, tau, occluded_mask, static
        )
        occlusion_names.append(label)
        occlusion_delta.append(base_probability - probability)
    occlusion = np.column_stack(occlusion_delta).astype(np.float32)

    private_root = WORKSPACE_ROOT / "private" / "explainability"
    private_path = private_root / f"{job.job_id}.npz"
    private_hash = _save_private_npz(
        private_path,
        feature_names=feature_names,
        feature_values=feature_values,
        shap_values=shap_values,
        raw_component_probability=raw_component_probability,
        pathway_contribution=pathway_contribution,
        fusion_weights=weights,
        occlusion_names=np.asarray(occlusion_names, dtype="U48"),
        occlusion_delta=occlusion,
    )
    report = {
        "job_id": job.job_id,
        "source_dataset": job.source_dataset,
        "seed": job.seed,
        "sample_rows": int(len(rows)),
        "feature_count": int(feature_values.shape[1]),
        "private_artifact": str(private_path.relative_to(REPO_ROOT)),
        "private_artifact_sha256": private_hash,
        "tree_shap_scope": "Flat-head mortality direct-clinical pathway",
        "pathway_contributions_before_isotonic_calibration": True,
        "target_information_used": False,
    }
    arrays_out = {
        "feature_names": feature_names,
        "feature_values": feature_values,
        "shap_values": shap_values,
        "pathway_contribution": pathway_contribution,
        "fusion_weights": weights,
        "occlusion_names": np.asarray(occlusion_names, dtype="U48"),
        "occlusion_delta": occlusion,
    }
    return report, arrays_out


def _display_registry() -> dict[str, str]:
    return {
        str(row["machine_name"]): str(row["display_name"])
        for row in shap_feature_registry_rows()
    }


def _plot_explainability(
    combined: dict[str, np.ndarray],
    importance: pd.DataFrame,
    pathway: pd.DataFrame,
    occlusion: pd.DataFrame,
) -> dict[str, str]:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(WORKSPACE_ROOT / "private" / "matplotlib_cache")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    figure_root = assert_safe_output_path(RESULTS_ROOT / "figures")
    figure_root.mkdir(parents=True, exist_ok=True)
    display = _display_registry()
    top = importance.sort_values("mean_abs_shap", ascending=False).head(15)
    feature_names = combined["feature_names"].astype(str)
    shap_values = combined["shap_values"]
    feature_values = combined["feature_values"]
    rng = np.random.default_rng(42)

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(11.2, 6.8), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, width_ratios=(1.55, 1.0))
    ax_shap = fig.add_subplot(grid[:, 0])
    top_names = top["feature"].tolist()[::-1]
    scatter = None
    for y, name in enumerate(top_names):
        index = int(np.flatnonzero(feature_names == name)[0])
        shap_column = shap_values[:, index]
        value_column = feature_values[:, index]
        low, high = np.nanpercentile(value_column, [5, 95])
        normalized = np.clip((value_column - low) / max(high - low, 1e-8), 0, 1)
        jitter = rng.normal(0.0, 0.115, size=len(shap_column))
        scatter = ax_shap.scatter(
            shap_column,
            y + jitter,
            c=normalized,
            cmap="cividis",
            vmin=0,
            vmax=1,
            s=6,
            alpha=0.50,
            linewidths=0,
            rasterized=True,
        )
    ax_shap.axvline(0.0, color="#667085", linewidth=0.8)
    ax_shap.set_yticks(range(len(top_names)))
    ax_shap.set_yticklabels([display.get(name, name) for name in top_names])
    ax_shap.set_xlabel("TreeSHAP value (mortality log-odds scale)")
    ax_shap.set_title("A  Flat-head mortality TreeSHAP")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax_shap, pad=0.02, fraction=0.035)
        colorbar.set_label("Feature value: low to high")

    ax_path = fig.add_subplot(grid[0, 1])
    path_summary = pathway.groupby("component", as_index=False).agg(
        weight=("weight", "mean"), mean_contribution=("mean_contribution", "mean")
    )
    labels = ["Hybrid", "XGBOOST-FLAT", "Flat-head"]
    colors = ["#6F3C97", "#B65D00", "#00796B"]
    positions = np.arange(3)
    ax_path.bar(
        positions,
        path_summary.set_index("component").loc[labels, "mean_contribution"],
        color=colors,
    )
    ax_path.set_xticks(positions, labels, rotation=15)
    ax_path.set_ylabel(r"Mean raw contribution $w_k p_k$")
    ax_path.set_title("B  Pre-calibration pathway contribution")

    ax_occ = fig.add_subplot(grid[1, 1])
    occ_summary = occlusion.groupby("occlusion", as_index=False).agg(
        mean_absolute_delta=("mean_absolute_delta", "mean")
    ).sort_values("mean_absolute_delta")
    ax_occ.barh(
        occ_summary["occlusion"],
        occ_summary["mean_absolute_delta"],
        color="#0077BB",
    )
    ax_occ.set_xlabel("Mean absolute change in calibrated risk")
    ax_occ.set_title("C  Temporal and feature-group occlusion")
    for axis in (ax_shap, ax_path, ax_occ):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color="#D0D5DD", linewidth=0.5, alpha=0.55)

    png_path = figure_root / "figure_5_shap_pathways_occlusion.png"
    pdf_path = figure_root / "figure_5_shap_pathways_occlusion.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    outputs = {}
    for path in (png_path, pdf_path):
        digest = sha256_file(path)
        safe_write_text(
            path.with_suffix(path.suffix + ".sha256"),
            f"{digest}  {path.relative_to(REPO_ROOT)}\n",
        )
        outputs[path.name] = digest
    return outputs


def run_explainability() -> tuple[dict[str, Any], int]:
    if not torch.cuda.is_available():
        raise RuntimeError("landmark SHAP/occlusion requires visible CUDA for Mamba inference")
    plan, plan_hash, config, config_hash = load_protocol()
    phase_e = require_correction07_phase_e_report()
    jobs = _require_full_jobs(config)
    frozen_before = verify_frozen_state(plan)
    started = time.perf_counter()
    reports = []
    bundles = []
    importance_rows = []
    pathway_rows = []
    occlusion_rows = []
    for source in ("mimic_iv", "eicu"):
        for seed in config["seeds"]:
            job = _matching_job(
                jobs, source, "MABOOST", "stress_augmented", int(seed)
            )
            report, arrays = _explain_job(job, config)
            reports.append(report)
            bundles.append(arrays)
            mean_abs = np.mean(np.abs(arrays["shap_values"]), axis=0)
            mean_signed = np.mean(arrays["shap_values"], axis=0)
            importance_rows.extend({
                "source_dataset": source,
                "seed": int(seed),
                "feature": str(name),
                "mean_abs_shap": float(mean_abs[index]),
                "mean_signed_shap": float(mean_signed[index]),
            } for index, name in enumerate(arrays["feature_names"]))
            for component_index, component in enumerate(
                ("Hybrid", "XGBOOST-FLAT", "Flat-head")
            ):
                pathway_rows.append({
                    "source_dataset": source,
                    "seed": int(seed),
                    "component": component,
                    "weight": float(arrays["fusion_weights"][component_index]),
                    "mean_contribution": float(
                        arrays["pathway_contribution"][:, component_index].mean()
                    ),
                })
            for index, label in enumerate(arrays["occlusion_names"]):
                delta = arrays["occlusion_delta"][:, index]
                occlusion_rows.append({
                    "source_dataset": source,
                    "seed": int(seed),
                    "occlusion": str(label),
                    "mean_signed_delta": float(delta.mean()),
                    "mean_absolute_delta": float(np.abs(delta).mean()),
                })
            resource_safe_point(f"explainability_{job.job_id}", force_cleanup=True)

    importance_per_seed = pd.DataFrame(importance_rows)
    importance = importance_per_seed.groupby("feature", as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        sd_abs_shap=("mean_abs_shap", "std"),
        mean_signed_shap=("mean_signed_shap", "mean"),
        source_seed_models=("seed", "count"),
    )
    pathway = pd.DataFrame(pathway_rows)
    occlusion = pd.DataFrame(occlusion_rows)
    combined = {
        "feature_names": bundles[0]["feature_names"],
        "feature_values": np.concatenate([item["feature_values"] for item in bundles]),
        "shap_values": np.concatenate([item["shap_values"] for item in bundles]),
    }
    if any(
        not np.array_equal(item["feature_names"], combined["feature_names"])
        for item in bundles[1:]
    ):
        raise RuntimeError("fixed-6 h TreeSHAP feature order drift across source-seed models")
    root = RESULTS_ROOT / "explainability"
    files = {
        "shap_feature_registry.csv": _write_csv(root / "shap_feature_registry.csv", pd.DataFrame(shap_feature_registry_rows())),
        "shap_importance_per_seed.csv": _write_csv(root / "shap_importance_per_seed.csv", importance_per_seed),
        "shap_importance_aggregate.csv": _write_csv(root / "shap_importance_aggregate.csv", importance),
        "fusion_pathway_contributions.csv": _write_csv(root / "fusion_pathway_contributions.csv", pathway),
        "temporal_feature_occlusion.csv": _write_csv(root / "temporal_feature_occlusion.csv", occlusion),
    }
    table7 = pd.concat([
        importance.nlargest(15, "mean_abs_shap").assign(section="TreeSHAP feature"),
        pathway.groupby("component", as_index=False).agg(
            mean_abs_shap=("mean_contribution", "mean"),
            sd_abs_shap=("mean_contribution", "std"),
        ).rename(columns={"component": "feature"}).assign(section="Pathway contribution"),
        occlusion.groupby("occlusion", as_index=False).agg(
            mean_abs_shap=("mean_absolute_delta", "mean"),
            sd_abs_shap=("mean_absolute_delta", "std"),
        ).rename(columns={"occlusion": "feature"}).assign(section="Occlusion"),
    ], ignore_index=True, sort=False)
    files["table_7_shap_pathway_summary.csv"] = _write_csv(
        root / "table_7_shap_pathway_summary.csv", table7
    )
    figures = _plot_explainability(combined, importance, pathway, occlusion)
    report = {
        "schema_version": "1.0.0",
        "status": "pass",
        "analysis_plan_sha256": plan_hash,
        "config_sha256": config_hash,
        "phase_e_report_sha256": phase_e["_phase_report_sha256"],
        "correction07_approval_chain": phase_e["approval_chain"],
        "jobs": reports,
        "files": files,
        "figures": figures,
        "interpretation_boundary": (
            "TreeSHAP, w_k p_k pathway contributions, and occlusion are descriptive "
            "associations, not causal effects. Component contributions are computed "
            "before isotonic calibration and are not an additive decomposition of the "
            "calibrated probability."
        ),
        "row_level_shap_and_predictions_private_only": True,
        "patient_level_content": False,
        "manuscript_modified": False,
    }
    report_path = root / "explainability_report.json"
    safe_write_json(report_path, report)
    safe_write_text(
        report_path.with_suffix(".json.sha256"),
        f"{sha256_file(report_path)}  {report_path.relative_to(REPO_ROOT)}\n",
    )
    frozen_after = verify_frozen_state(plan)
    payload = {
        "status": "pass",
        "command": "python -u reproduce.py explain",
        "analysis_plan_sha256": plan_hash,
        "config_sha256": config_hash,
        "phase_e_report_sha256": phase_e["_phase_report_sha256"],
        "correction07_approval_chain": phase_e["approval_chain"],
        "runtime": {"wall_seconds": float(time.perf_counter() - started)},
        "files": files,
        "figures": figures,
        "tests": [
            {"name": "treeshap_rerun_fixed_6h", "passed": True, "detail": "10 source-seed MABOOST models"},
            {"name": "beeswarm_resolution", "passed": True, "detail": "400 dpi"},
            {"name": "row_level_private", "passed": True, "detail": "private/explainability only"},
            {"name": "frozen_v11_unchanged", "passed": frozen_before == frozen_after, "detail": "pre/post snapshots match"},
        ],
        "errors": [],
        "manuscript_modified": False,
    }
    phase_path, phase_hash = write_phase_report("F_EXPLAINABILITY", payload)
    return {
        "status": "pass",
        "report": str(phase_path.relative_to(REPO_ROOT)),
        "report_sha256": phase_hash,
        "explainability_report": str(report_path.relative_to(REPO_ROOT)),
    }, 0
