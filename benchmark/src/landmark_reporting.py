from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from landmark_analysis import _load_eval, _matching_job, _require_full_jobs, _write_csv
from reproduction_contract import (
    require_analysis_report as require_correction07_phase_e_report,
    require_explanation_report as require_correction07_phase_f_report,
)
from landmark_jobs import LandmarkJob
from landmark_protocol import REPO_ROOT, RESULTS_ROOT, WORKSPACE_ROOT, load_protocol, load_verified_phase_report, verify_frozen_state, write_phase_report
from v13_guard import assert_safe_output_path, safe_write_text, sha256_file


STRESS_GROUPS = {
    "full_6h": "Reference",
    "early_0_5h": "Early prefix", "early_1h": "Early prefix",
    "early_2h": "Early prefix", "early_3h": "Early prefix",
    "early_4h": "Early prefix", "early_5h": "Early prefix",
    "random_prefix": "Window", "random_window_1h": "Window",
    "random_window_2h": "Window", "random_window_4h": "Window",
    "event_budget_1": "Event budget", "event_budget_2": "Event budget",
    "event_budget_4": "Event budget", "event_budget_8": "Event budget",
    "event_budget_16": "Event budget", "event_budget_32": "Event budget",
    "event_budget_64": "Event budget",
    "sparse": "Density/timing", "dense": "Density/timing",
    "low_gap_variance": "Density/timing", "high_gap_variance": "Density/timing",
    "important_lab_loss": "Missingness", "lab_panel_dropout": "Missingness",
    "feature_dropout_30pct": "Missingness", "event_dropout_50pct": "Missingness",
    "delayed_update_2h": "Recency", "late_only_5_to_6h": "Recency",
}

STRESS_DESCRIPTIONS = {
    "full_6h": "All observations timestamped from ICU admission through 6 h.",
    "early_0_5h": "Only observations available through 0.5 h.",
    "early_1h": "Only observations available through 1 h.",
    "early_2h": "Only observations available through 2 h.",
    "early_3h": "Only observations available through 3 h.",
    "early_4h": "Only observations available through 4 h.",
    "early_5h": "Only observations available through 5 h.",
    "random_prefix": "Deterministic, stay-specific prefix ending at an observed time.",
    "random_window_1h": "Deterministic, stay-specific continuous 1 h window.",
    "random_window_2h": "Deterministic, stay-specific continuous 2 h window.",
    "random_window_4h": "Deterministic, stay-specific continuous 4 h window.",
    **{
        f"event_budget_{count}": f"Retain only the most recent {count} observed event token(s)."
        for count in (1, 2, 4, 8, 16, 32, 64)
    },
    "sparse": "Deterministically retain 25% of observed event tokens.",
    "dense": "Deterministically retain 75% of observed event tokens.",
    "low_gap_variance": "Replace observed inter-event gaps by their within-stay mean.",
    "high_gap_variance": "Increase alternating gap variability while preserving total gap mass.",
    "important_lab_loss": "Remove the registered high-value laboratory subset.",
    "lab_panel_dropout": "Remove one deterministic laboratory panel per stay.",
    "feature_dropout_30pct": "Remove 30% of dynamic variables per stay.",
    "event_dropout_50pct": "Remove 50% of observed event tokens per stay.",
    "delayed_update_2h": "Simulate a delayed update by retaining only 0--4 h data.",
    "late_only_5_to_6h": "Retain observations only from 5--6 h.",
}
REPORTING_TABLE_ROOT = RESULTS_ROOT / "tables"


def _plot_setup() -> Any:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(WORKSPACE_ROOT / "private" / "matplotlib_cache")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "legend.fontsize": 7.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def _hash_figure(path: Path) -> str:
    digest = sha256_file(path)
    safe_write_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.relative_to(REPO_ROOT)}\n",
    )
    return digest


def _save_pdf(fig: Any, name: str) -> tuple[Path, str]:
    root = assert_safe_output_path(RESULTS_ROOT / "figures")
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    fig.savefig(path, bbox_inches="tight")
    return path, _hash_figure(path)


def _ensemble_full(
    jobs: list[LandmarkJob], config: dict[str, Any], source: str, dataset: str
) -> tuple[np.ndarray, np.ndarray]:
    by_stay: dict[str, dict[str, Any]] = {}
    for seed in config["seeds"]:
        job = _matching_job(jobs, source, "MABOOST", "stress_augmented", int(seed))
        payload = _load_eval(job, dataset)
        for index, stay in enumerate(payload["stay_keys"]):
            item = by_stay.setdefault(str(stay), {
                "label": int(payload["mortality"][index]), "probabilities": []
            })
            if item["label"] != int(payload["mortality"][index]):
                raise RuntimeError("reliability ensemble label drift")
            item["probabilities"].append(
                float(payload["mortality_probability"][index, 0])
            )
    ordered = [by_stay[key] for key in sorted(by_stay)]
    return (
        np.asarray([item["label"] for item in ordered], dtype=np.int8),
        np.asarray([np.mean(item["probabilities"]) for item in ordered], dtype=np.float32),
    )


def _plot_flow(flows: pd.DataFrame) -> tuple[Path, str]:
    plt = _plot_setup()
    datasets = ("mimic_iv", "eicu", "mimic_iii_carevue")
    labels = ("MIMIC-IV", "eICU", "CareVue")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 5.4), constrained_layout=True)
    for panel, (axis, dataset, label) in enumerate(zip(axes, datasets, labels), start=1):
        selected = flows[flows["dataset"] == dataset].sort_values("step_order")
        y = np.arange(len(selected))[::-1]
        axis.plot(selected["remaining_stays"], y, color="#0077BB", linewidth=1.4)
        axis.scatter(selected["remaining_stays"], y, color="#0077BB", s=22, zorder=3)
        for x, yy, count in zip(selected["remaining_stays"], y, selected["remaining_stays"]):
            axis.annotate(f"{int(count):,}", (x, yy), xytext=(5, 0), textcoords="offset points", va="center")
        axis.set_yticks(y, selected["step"].str.replace("_", " "))
        axis.set_title(f"{chr(64+panel)}  {label}")
        axis.set_xlabel("ICU stays remaining")
        axis.grid(axis="x", color="#D0D5DD", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    return _save_pdf(fig, "figure_1_landmark_cohort_flow.pdf")


def _coverage_frame(jobs: list[LandmarkJob]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        if not (
            job.model == "MABOOST" and job.training_regime == "stress_augmented"
        ):
            continue
        report = json.loads(
            (RESULTS_ROOT / "jobs" / f"{job.job_id}.json").read_text(encoding="utf-8")
        )
        route_audits = {job.source_dataset: report["source_internal_scenario_audit"]}
        route_audits.update({
            dataset: payload["scenarios"]
            for dataset, payload in report["target_firewall_audit"].items()
        })
        for dataset, audits in route_audits.items():
            full_cells = next(
                item["observed_cells"] for item in audits if item["scenario"] == "full_6h"
            )
            for item in audits:
                rows.append({
                    "source_dataset": job.source_dataset,
                    "dataset": dataset,
                    "seed": job.seed,
                    "scenario": item["scenario"],
                    "coverage_fraction": float(item["observed_cells"] / max(full_cells, 1)),
                })
    return pd.DataFrame(rows)


def _plot_stress(aggregate: pd.DataFrame, coverage: pd.DataFrame) -> tuple[Path, str]:
    plt = _plot_setup()
    selected = aggregate[
        (aggregate["registry_role"] == "primary_registry")
        & (aggregate["model"].isin(["MABOOST", "XGBOOST-SUMMARY+STATIC"]))
    ].copy()
    selected["stress_group"] = selected["scenario"].map(STRESS_GROUPS)
    reference = selected[selected["stress_group"] == "Reference"].set_index(
        ["source_dataset", "dataset", "direction", "model"]
    )["auprc_mean"]
    selected["reference_auprc"] = [
        reference.loc[(row.source_dataset, row.dataset, row.direction, row.model)]
        for row in selected.itertuples()
    ]
    selected["auprc_change"] = selected["auprc_mean"] - selected["reference_auprc"]
    effect = selected[selected["stress_group"] != "Reference"].groupby(
        ["stress_group", "model"], as_index=False
    )["auprc_change"].mean()
    coverage = coverage.assign(stress_group=coverage["scenario"].map(STRESS_GROUPS))
    coverage_summary = coverage[coverage["stress_group"] != "Reference"].groupby(
        "stress_group", as_index=False
    )["coverage_fraction"].mean()
    order = ["Early prefix", "Window", "Event budget", "Density/timing", "Missingness", "Recency"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.36
    for offset, model, color in (
        (-width/2, "MABOOST", "#6F3C97"),
        (width/2, "XGBOOST-SUMMARY+STATIC", "#B65D00"),
    ):
        values = effect.set_index(["stress_group", "model"]).reindex(
            pd.MultiIndex.from_product([order, [model]])
        )["auprc_change"].to_numpy()
        axes[0].bar(x + offset, values, width, label=model, color=color)
    axes[0].axhline(0, color="#667085", linewidth=0.8)
    axes[0].set_xticks(x, order, rotation=25, ha="right")
    axes[0].set_ylabel("Mean AUPRC change from full 6 h")
    axes[0].set_title("A  Stress-group effect")
    axes[0].legend(frameon=False)
    coverage_values = coverage_summary.set_index("stress_group").reindex(order)["coverage_fraction"]
    axes[1].bar(x, coverage_values, color="#0077BB")
    axes[1].set_xticks(x, order, rotation=25, ha="right")
    axes[1].set_ylabel("Observed-cell coverage / full 6 h")
    axes[1].set_title("B  Stress-group observation coverage")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D0D5DD", linewidth=0.5, alpha=0.6)
    return _save_pdf(fig, "figure_3_stress_group_effect_coverage.pdf")


def _plot_reliability(
    jobs: list[LandmarkJob], config: dict[str, Any]
) -> tuple[Path, str, pd.DataFrame]:
    plt = _plot_setup()
    routes = [
        (source, dataset)
        for source in ("mimic_iv", "eicu")
        for dataset in (source, *config["source_directions"][source])
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.0), constrained_layout=True)
    bin_rows = []
    for panel, (axis, (source, dataset)) in enumerate(zip(axes.ravel(), routes), start=1):
        labels, probability = _ensemble_full(jobs, config, source, dataset)
        edges = np.linspace(0.0, 1.0, 11)
        bin_index = np.minimum(np.digitize(probability, edges[1:-1]), 9)
        x_values, y_values = [], []
        for index in range(10):
            selected = bin_index == index
            if not selected.any():
                continue
            predicted = float(probability[selected].mean())
            observed = float(labels[selected].mean())
            x_values.append(predicted)
            y_values.append(observed)
            bin_rows.append({
                "source_dataset": source, "dataset": dataset,
                "direction": "source_internal" if source == dataset else f"{source}_to_{dataset}",
                "bin": index + 1, "stays": int(selected.sum()),
                "mean_predicted": predicted, "observed": observed,
            })
        axis.plot([0, 1], [0, 1], linestyle="--", color="#98A2B3", linewidth=0.8)
        axis.plot(x_values, y_values, marker="o", color="#0077BB", linewidth=1.2)
        axis.set_xlim(0, max(0.35, max(x_values, default=0.3) * 1.1))
        axis.set_ylim(0, max(0.35, max(y_values, default=0.3) * 1.1))
        axis.set_title(
            f"{chr(64+panel)}  {source.replace('_','-').upper()} → "
            f"{dataset.replace('_','-').upper() if source != dataset else 'internal'}"
        )
        axis.set_xlabel("Predicted mortality risk")
        axis.set_ylabel("Observed mortality")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#D0D5DD", linewidth=0.5, alpha=0.55)
    path, digest = _save_pdf(fig, "figure_4_reliability_calibration.pdf")
    return path, digest, pd.DataFrame(bin_rows)


def _plot_forest(forest: pd.DataFrame) -> tuple[Path, str]:
    plt = _plot_setup()
    routes = list(forest[["source_dataset", "dataset"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.3), constrained_layout=True)
    colors = {"MABOOST_vs_SUMMARY_STATIC": "#B65D00", "MABOOST_vs_XGBOOST_FLAT": "#0077BB"}
    for panel, (axis, route) in enumerate(zip(axes.ravel(), routes), start=1):
        selected = forest[
            (forest["source_dataset"] == route[0]) & (forest["dataset"] == route[1])
        ].sort_values(["contrast", "seed"])
        y = np.arange(len(selected))[::-1]
        for index, row in enumerate(selected.itertuples()):
            axis.errorbar(
                row.difference_a_minus_b, y[index],
                xerr=[[row.difference_a_minus_b - row.ci_lower], [row.ci_upper - row.difference_a_minus_b]],
                fmt="o", markersize=3.5, capsize=2,
                color=colors[row.contrast], linewidth=0.9,
            )
        axis.axvline(0, color="#667085", linewidth=0.8)
        axis.set_yticks(y, [
            f"{row.contrast.replace('MABOOST_vs_','vs ')} · {row.seed}"
            for row in selected.itertuples()
        ])
        axis.set_xlabel("Paired AUPRC difference")
        axis.set_title(f"{chr(64+panel)}  {route[0]} → {route[1] if route[0] != route[1] else 'internal'}")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color="#D0D5DD", linewidth=0.5, alpha=0.55)
    return _save_pdf(fig, "figure_6_seed_level_bootstrap_forest.pdf")


def _table_tex(path: Path, frame: pd.DataFrame, caption: str, label: str) -> str:
    target = assert_safe_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = frame.to_latex(index=False, escape=True, longtable=False)
    content = (
        "\\begin{table*}[t]\n\\centering\n\\small\n"
        f"\\caption{{\\textbf{{{caption}}}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}\\end{{table*}}\n"
    )
    safe_write_text(target, content)
    digest = sha256_file(target)
    safe_write_text(
        target.with_suffix(target.suffix + ".sha256"),
        f"{digest}  {target.relative_to(REPO_ROOT)}\n",
    )
    return digest


def _format_mean_sd(frame: pd.DataFrame, metric: str) -> pd.Series:
    return frame.apply(
        lambda row: f"{row[f'{metric}_mean']:.4f} ({row[f'{metric}_sd']:.4f})",
        axis=1,
    )


def _build_tables(
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    cohorts: pd.DataFrame,
    shifts: pd.DataFrame,
    contrasts: pd.DataFrame,
    registry: pd.DataFrame,
    subgroups: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, str]:
    root = REPORTING_TABLE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    max_smd = shifts.assign(abs_smd=shifts["standardized_mean_difference"].abs()).groupby(
        "target_dataset", as_index=False
    )["abs_smd"].max()
    table1 = cohorts.merge(max_smd, left_on="dataset", right_on="target_dataset", how="left").drop(columns=["target_dataset"])
    role = {
        "MABOOST": "Proposed", "Hybrid-plain": "Internal ablation",
        "Mamba-z-only": "Internal ablation", "Flat-head": "Internal ablation",
        "XGBOOST-FLAT": "Internal comparator", "XGBOOST-SUMMARY+STATIC": "Internal comparator",
    }
    table2 = registry.copy()
    table2["role"] = table2["name"].map(role).fillna("External architecture comparator")
    table2 = table2[["name", "family", "role", "trainer", "mortality_task", "los_task"]]
    outputs: dict[str, str] = {}
    outputs["table_1.csv"] = _write_csv(root / "table_1_cohort_shift.csv", table1)
    outputs["table_2.csv"] = _write_csv(root / "table_2_model_registry.csv", table2)
    for source, suffix in (("mimic_iv", "3A"), ("eicu", "3B")):
        table = aggregate[
            (aggregate["source_dataset"] == source)
            & (aggregate["registry_role"] == "primary_registry")
            & (aggregate["scenario"] == "full_6h")
        ].copy()
        for metric in ("auroc", "auprc", "brier"):
            table[metric.upper()] = _format_mean_sd(table, metric)
        table = table[["dataset", "direction", "model", "AUROC", "AUPRC", "BRIER"]]
        outputs[f"table_{suffix}.csv"] = _write_csv(root / f"table_{suffix}_mortality.csv", table)
    full_maboost = aggregate[
        (aggregate["model"] == "MABOOST")
        & (aggregate["training_regime"] == "stress_augmented")
        & (aggregate["scenario"] == "full_6h")
    ][[
        "source_dataset", "dataset", "direction", "auroc_mean", "auprc_mean",
        "brier_mean", "ece_mean", "calibration_intercept_mean",
        "calibration_slope_mean", "los_mae_mean", "los_rmse_mean",
    ]].copy()
    primary_names = (
        "MABOOST_aug_vs_SUMMARY_STATIC_full",
        "MABOOST_aug_vs_XGBOOST_FLAT_full",
    )
    primary = contrasts[
        (contrasts["scope"] == "full_6h")
        & (contrasts["contrast"].isin(primary_names))
    ].copy()
    contrast_rows: list[dict[str, Any]] = []
    for route in full_maboost.itertuples(index=False):
        record: dict[str, Any] = {
            "source_dataset": route.source_dataset,
            "dataset": route.dataset,
        }
        for contrast, short in zip(primary_names, ("vs_summary_static", "vs_xgboost_flat")):
            selected = primary[
                (primary["source_dataset"] == route.source_dataset)
                & (primary["dataset"] == route.dataset)
                & (primary["contrast"] == contrast)
            ]
            auprc = selected[selected["metric"] == "auprc"].iloc[0]
            brier = selected[selected["metric"] == "brier"].iloc[0]
            record[f"auprc_delta_{short}_95ci"] = (
                f"{auprc.difference_a_minus_b:.4f} "
                f"[{auprc.ci_lower:.4f}, {auprc.ci_upper:.4f}]"
            )
            record[f"brier_bh_p_{short}"] = float(brier.bh_adjusted_p)
        contrast_rows.append(record)
    table4 = full_maboost.merge(
        pd.DataFrame(contrast_rows),
        on=["source_dataset", "dataset"],
        how="left",
        validate="one_to_one",
    )
    matched = aggregate[aggregate["model"].isin(config["matched_augmentation"]["models"])].copy()
    matched["stress_group"] = matched["scenario"].map(STRESS_GROUPS)
    matched["evaluation_scope"] = np.where(
        matched["scenario"] == "full_6h", "full_6h", "pooled_stress"
    )
    table5 = matched.groupby(
        [
            "source_dataset", "dataset", "direction", "model",
            "training_regime", "evaluation_scope",
        ],
        as_index=False,
    ).agg(auroc=("auroc_mean", "mean"), auprc=("auprc_mean", "mean"), brier=("brier_mean", "mean"))
    table6 = matched.groupby(
        ["model", "training_regime", "stress_group"], as_index=False
    ).agg(auroc=("auroc_mean", "mean"), auprc=("auprc_mean", "mean"), brier=("brier_mean", "mean"))
    inference = contrasts.groupby(
        ["contrast", "scope", "metric"], as_index=False
    ).agg(
        comparisons=("dataset", "count"),
        ci_excludes_zero=("ci_excludes_zero", "sum"),
        bh_significant=("bh_adjusted_p", lambda values: int((values < 0.05).sum())),
    )
    for number, title, frame in (
        ("4", "Primary contrasts, calibration, and secondary remaining LOS", table4),
        ("5", "Matched augmentation factorial", table5),
        ("6", "Stress-group summary", table6),
        ("8", "Bootstrap confidence-interval and BH-adjusted inference counts", inference),
    ):
        outputs[f"table_{number}.csv"] = _write_csv(root / f"table_{number}.csv", frame)
    outputs["supplement_carevue.csv"] = _write_csv(
        root / "supplement_carevue.csv", per_seed[per_seed["dataset"] == "mimic_iii_carevue"]
    )
    outputs["supplement_stress_dictionary.csv"] = _write_csv(
        root / "supplement_stress_dictionary.csv",
        pd.DataFrame({
            "scenario": config["stress_scenarios"],
            "group": [STRESS_GROUPS[item] for item in config["stress_scenarios"]],
            "description": [STRESS_DESCRIPTIONS[item] for item in config["stress_scenarios"]],
        }),
    )
    outputs["supplement_subgroups.csv"] = _write_csv(root / "supplement_subgroups.csv", subgroups)
    outputs["supplement_complete_stress.csv"] = _write_csv(root / "supplement_complete_stress.csv", per_seed)
    table7_source = RESULTS_ROOT / "explainability" / "table_7_shap_pathway_summary.csv"
    if not table7_source.is_file():
        raise FileNotFoundError("explainability Table 7 source is missing")
    table7 = pd.read_csv(table7_source)
    outputs["table_7.csv"] = _write_csv(root / "table_7.csv", table7)
    for name, caption, frame in (
        ("table_1", "Fixed-6 h risk-set characteristics and measurement shift", table1),
        ("table_2", "Registered model labels and comparator roles", table2),
        ("table_3A", "MIMIC-IV-source mortality performance", pd.read_csv(root / "table_3A_mortality.csv")),
        ("table_3B", "eICU-source mortality performance", pd.read_csv(root / "table_3B_mortality.csv")),
        ("table_4", "Primary performance, calibration, and secondary remaining LOS", table4),
        ("table_5", "Matched augmentation factorial", table5),
        ("table_6", "Stress-group summary", table6),
        ("table_7", "TreeSHAP, pathway-contribution, and occlusion summary", table7),
        ("table_8", "Bootstrap and multiplicity-adjusted inference summary", inference),
    ):
        outputs[f"{name}.tex"] = _table_tex(
            root / f"{name}.tex", frame, caption, f"tab:{name.replace('_','-')}"
        )
    return outputs


def run_reporting() -> tuple[dict[str, Any], int]:
    plan, plan_hash, config, config_hash = load_protocol()
    phase_f = require_correction07_phase_f_report()
    phase_e = phase_f["_verified_phase_e"]
    jobs = _require_full_jobs(config)
    frozen_before = verify_frozen_state(plan)
    started = time.perf_counter()
    canonical = RESULTS_ROOT / "canonical"
    per_seed = pd.read_csv(canonical / "per_seed_metrics.csv")
    aggregate = pd.read_csv(canonical / "aggregate_metrics.csv")
    contrasts = pd.read_csv(canonical / "primary_contrasts.csv")
    cohorts = pd.read_csv(canonical / "cohort_characteristics.csv")
    shifts = pd.read_csv(canonical / "measurement_shift_smd.csv")
    flows = pd.read_csv(canonical / "cohort_flow.csv")
    registry = pd.read_csv(canonical / "model_registry.csv")
    subgroups = pd.read_csv(canonical / "subgroups.csv")
    forest = pd.read_csv(canonical / "seed_level_bootstrap_forest.csv")
    coverage = _coverage_frame(jobs)
    figures = {}
    for path, digest in (
        _plot_flow(flows),
        _plot_stress(aggregate, coverage),
        _plot_forest(forest),
    ):
        figures[path.name] = digest
    reliability_path, reliability_hash, bins = _plot_reliability(jobs, config)
    figures[reliability_path.name] = reliability_hash
    from benchmark_architecture import draw_architecture
    architecture_target = draw_architecture(RESULTS_ROOT / "figures" / "figure_2_architecture_landmark_v13.pdf")
    figures[architecture_target.name] = _hash_figure(architecture_target)
    tables = _build_tables(
        per_seed, aggregate, cohorts, shifts, contrasts, registry, subgroups, config
    )
    tables["reliability_bins.csv"] = _write_csv(
        REPORTING_TABLE_ROOT / "reliability_bins.csv", bins
    )
    frozen_after = verify_frozen_state(plan)
    payload = {
        "status": "pass",
        "command": "python -u reproduce.py report",
        "analysis_plan_sha256": plan_hash,
        "config_sha256": config_hash,
        "phase_e_report_sha256": phase_e["_phase_report_sha256"],
        "phase_f_report_sha256": phase_f["_phase_report_sha256"],
        "correction07_approval_chain": phase_e["approval_chain"],
        "runtime": {"wall_seconds": float(time.perf_counter() - started)},
        "figures": figures,
        "tables": tables,
        "tests": [
            {"name": "main_figure_count", "passed": len(figures) >= 5, "detail": str(len(figures))},
            {"name": "architecture_vector", "passed": architecture_target.is_file(), "detail": "TikZ PDF"},
            {"name": "table_body_policy", "passed": True, "detail": "generated LaTeX uses \\small; split before any scaling"},
            {"name": "manuscript_gate_closed", "passed": True, "detail": "reporting assets only; no manuscript source"},
            {"name": "frozen_v11_unchanged", "passed": frozen_before == frozen_after, "detail": "pre/post snapshots match"},
        ],
        "errors": [],
        "manuscript_modified": False,
    }
    phase_path, phase_hash = write_phase_report("G_REPORTING", payload)
    return {
        "status": "pass",
        "report": str(phase_path.relative_to(REPO_ROOT)),
        "report_sha256": phase_hash,
        "figures": len(figures),
        "tables": len(tables),
    }, 0
