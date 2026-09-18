"""Verify published aggregate inputs and render the README figures (no raw data)."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/benchmark/correction09"
OUT = ROOT / "docs/figures"
ROUTES = [
    ("mimic_iv", "mimic_iv", "MIMIC-IV\ninternal"),
    ("mimic_iv", "eicu", "MIMIC-IV\nto eICU"),
    ("mimic_iv", "mimic_iii_carevue", "MIMIC-IV\nto CareVue"),
    ("eicu", "eicu", "eICU\ninternal"),
    ("eicu", "mimic_iv", "eICU\nto MIMIC-IV"),
    ("eicu", "mimic_iii_carevue", "eICU\nto CareVue"),
]


def verify():
    manifest = json.loads((DATA / "PROVENANCE.json").read_text())
    for record in manifest["artifacts"]:
        path = (DATA / record["file"]).resolve()
        if not path.is_relative_to(DATA.resolve()) or not path.is_file():
            raise RuntimeError(f"Missing/unsafe aggregate asset: {record['file']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Aggregate asset hash mismatch: {record['file']}")
    print(f"Verified {len(manifest['artifacts'])} aggregate assets from correction09.")
    return manifest


def save(fig, name):
    import matplotlib.pyplot as plt
    for suffix in ("png", "svg"):
        path = OUT / f"{name}.{suffix}"
        metadata = {"Date": None} if suffix == "svg" else None
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white", metadata=metadata)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def architecture():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(16, 7.8))
    ax.set(xlim=(0, 16), ylim=(0, 8)); ax.axis("off")
    ax.text(.3, 7.65, "MABOOST  |  fixed-6-hour prediction", fontsize=23, weight="bold", color="#16324f")
    ax.text(.3, 7.16, "A temporal–tabular model evaluated under observation stress and dataset shift", fontsize=13, color="#526474")

    def box(x, y, w, h, title, body, color="#eaf2f8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", facecolor=color, edgecolor="#b9cbd8", linewidth=1.2))
        ax.text(x+w/2, y+h-.32, title, ha="center", va="top", fontsize=12.2, weight="bold", color="#16324f")
        ax.text(x+w/2, y+(h-.42)/2, body, ha="center", va="center", fontsize=10.7, color="#334a5b", linespacing=1.6)

    def arrow(x, y, u, v):
        ax.annotate("", xy=(u, v), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", color="#7791a2", lw=1.5, mutation_scale=14))

    box(.3, 2.1, 3.0, 3.4, "FIXED LANDMARK", "Age ≥65\nAlive and in ICU at 6 h\n40 dynamic + 12 static features\nRecorded-time filtering\nwhere timestamps exist\nAt most 96 observed steps", "#edf5f3")
    box(3.95, 4.7, 3.5, 1.5, "TEMPORAL ENCODER", "Values + masks + time gaps\nMamba → 512-D representation")
    box(3.95, 2.75, 3.5, 1.5, "TABULAR FEATURES", "200 summaries + 12 static\n40 observedness rates (Flat-head)")
    box(3.95, .8, 3.5, 1.45, "SOURCE TRAINING", "Full view + 6 augmentation views\nSame views in matched ablations", "#fff5df")
    box(8.1, 4.8, 3.15, 1.4, "HYBRID", "512-D encoder + 212-D tabular\nXGBoost mortality / LOS heads")
    box(8.1, 2.85, 3.15, 1.4, "XGBOOST-FLAT", "200 summary features\nMortality / LOS heads")
    box(8.1, .9, 3.15, 1.4, "FLAT-HEAD", "200 summaries + 12 static\n+ 40 observedness rates")
    box(12.05, 2.25, 3.5, 3.0, "SOURCE-SELECTED FUSION", "Separate mortality / LOS weights\nMortality isotonic calibration\n\nPost-landmark ICU death risk\nRemaining ICU LOS from 6 h", "#e3f2ef")
    arrow(3.36, 4.8, 3.89, 5.4); arrow(3.36, 3.55, 3.89, 3.5)
    arrow(7.51, 5.45, 8.04, 5.45)
    arrow(7.51, 3.65, 8.04, 4.95); arrow(7.51, 3.5, 8.04, 3.5); arrow(7.51, 3.15, 8.04, 1.55)
    for y in (5.45, 3.55, 1.55): arrow(11.31, y, 11.99, 3.75)
    ax.text(.3, .22, "Source-only fitting and selection  •  Locked predictions before target-label evaluation  •  28 scenarios on the same 6 h risk set", fontsize=11, color="#526474")
    save(fig, "architecture")


def mortality_heatmap():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    a = pd.read_csv(DATA / "aggregate_metrics.csv")
    models = [("MABOOST", "stress_augmented"), ("MABOOST", "full_only"),
              ("Hybrid-plain", "stress_augmented"), ("Hybrid-plain", "full_only"),
              ("XGBOOST-SUMMARY+STATIC", "stress_augmented"), ("XGBOOST-SUMMARY+STATIC", "full_only"),
              ("XGBOOST-FLAT", "stress_augmented"), ("XGBOOST-FLAT", "full_only"),
              *[(m, "full_only") for m in ("Flat-head", "Mamba-z-only", "GRU-D", "LSTM", "Transformer", "TCN", "SAnD", "STraTS", "InterpNet", "DuETT")]]
    values = np.empty((len(models), len(ROUTES)))
    for i, (model, regime) in enumerate(models):
        for j, (source, target, _) in enumerate(ROUTES):
            row = a[(a.source_dataset == source) & (a.dataset == target) & (a.model == model) & (a.training_regime == regime) & (a.scenario == "full_6h")]
            if len(row) != 1 or int(row.iloc[0].seed_count) != 5:
                raise RuntimeError(f"Expected one five-seed metric: {source}/{target}/{model}/{regime}")
            values[i, j] = row.iloc[0].auprc_mean
    fig, ax = plt.subplots(figsize=(12, 10.2))
    im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=.5, aspect="auto")
    labels = [f"{m}  ·  {'aug' if r == 'stress_augmented' else 'full'}" for m, r in models]
    ax.set_yticks(range(len(models)), labels, fontsize=10)
    ax.set_xticks(range(len(ROUTES)), [r[2] for r in ROUTES], fontsize=10)
    ax.tick_params(length=0, pad=9)
    for (i, j), val in np.ndenumerate(values):
        best = val == values[:, j].max()
        ax.text(j, i, f"{val:.3f}", va="center", ha="center", fontsize=10.5, color="white" if val >= .30 else "#17334e", weight="bold" if best else "normal")
    ax.axhline(7.5, color="white", lw=2)
    ax.set_title("Mortality discrimination at the full 6 h view", fontsize=18, weight="bold", color="#16324f", loc="left", pad=22)
    for s in ax.spines.values(): s.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=.035, pad=.03); bar.set_label("AUPRC · mean across five seeds")
    fig.text(.12, .018, "aug: stress-augmented training; full: full-only training. Bold: largest displayed mean in each route.\nCareVue is a historical target. These descriptive seed means are not paired-ensemble effect estimates.", fontsize=10, color="#526474")
    fig.subplots_adjust(bottom=.13, left=.34)
    save(fig, "mortality_auprc")


def paired_effects():
    import matplotlib.pyplot as plt
    import pandas as pd
    p = pd.read_csv(DATA / "primary_contrasts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True, sharex=True)
    comparisons = [("MABOOST_aug_vs_SUMMARY_STATIC_full", "vs summary+static · full-only", "#087f8c", -.13),
                   ("MABOOST_aug_vs_SUMMARY_STATIC_aug", "vs summary+static · augmented", "#d17927", .13)]
    for ax, scope, title in zip(axes, ["full_6h", "pooled_stress"], ["Full 6 h view", "Pooled 27 stress views"]):
        ax.axvline(0, color="#687e8c", lw=1, ls="--")
        for i, (source, target, _) in enumerate(ROUTES):
            for contrast, label, color, offset in comparisons:
                row = p[(p.source_dataset == source) & (p.dataset == target) & (p.metric == "auprc") & (p.scope == scope) & (p.contrast == contrast)]
                if len(row) != 1: raise RuntimeError("Ambiguous paired contrast")
                d = row.iloc[0]; effect = d.difference_a_minus_b
                ax.errorbar(effect, i+offset, xerr=[[effect-d.ci_lower], [d.ci_upper-effect]], fmt="o", color=color, capsize=4, markersize=6, lw=1.8, label=label if i == 0 else None)
        ax.set_title(title, fontsize=14, weight="bold", loc="left", color="#16324f")
        ax.set_xlabel("Δ AUPRC (MABOOST augmented − comparator)", fontsize=10)
        ax.set_xlim(-.008, .056); ax.grid(axis="x", color="#e5ebef")
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(ROUTES)), [r[2].replace("\n", " ") for r in ROUTES], fontsize=11)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.56, .083), frameon=False)
    fig.suptitle("How much remains after matching stress augmentation?", fontsize=19, weight="bold", color="#16324f", x=.51, y=.98)
    fig.text(.06, .025, "Seed-ensemble predictions; 10,000 patient-clustered paired bootstrap draws; pointwise 95% CIs.\nPooled effects use stay × scenario predictions, not the arithmetic mean of scenario AUPRCs. No multiplicity-adjusted AUPRC test is claimed.", fontsize=10, color="#526474")
    fig.subplots_adjust(left=.20, right=.98, top=.88, bottom=.22, wspace=.12)
    save(fig, "paired_auprc_effects")


def los_and_fusion():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    a = pd.read_csv(DATA / "aggregate_metrics.csv")
    f = pd.read_csv(DATA / "fusion_pathway_contributions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for i, (source, target, _) in enumerate(ROUTES):
        rows = a[(a.source_dataset == source) & (a.dataset == target) & (a.scenario == "full_6h")]
        for model, regime, color, off in [("MABOOST", "stress_augmented", "#087f8c", -.12), ("XGBOOST-SUMMARY+STATIC", "full_only", "#d17927", .12)]:
            d = rows[(rows.model == model) & (rows.training_regime == regime)].iloc[0]
            ax.plot(d.los_mae_mean, i+off, "o", color=color, label=("MABOOST aug" if model == "MABOOST" else "Summary+static full") if i == 0 else None)
    ax.set_yticks(range(6), [r[2].replace("\n", " ") for r in ROUTES], fontsize=10)
    ax.invert_yaxis(); ax.set_xlabel("Remaining ICU LOS MAE (days) · lower is better")
    ax.set_title("LOS benefit is not consistent", loc="left", fontsize=14, weight="bold", color="#16324f")
    ax.grid(axis="x", color="#e5ebef"); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, frameon=False, fontsize=8)
    ax = axes[1]; seeds = [42, 137, 2025, 2718, 31415]
    keys = [(s, seed) for s in ("mimic_iv", "eicu") for seed in seeds]
    bottom = np.zeros(10)
    for component, color in [("Hybrid", "#087f8c"), ("XGBOOST-FLAT", "#d17927"), ("Flat-head", "#8e9cc5")]:
        vals = np.array([f[(f.source_dataset == s) & (f.seed == seed) & (f.component == component)].iloc[0].weight for s, seed in keys])
        ax.barh(range(10), vals, left=bottom, color=color, label=component, height=.68); bottom += vals
    np.testing.assert_allclose(bottom, 1)
    ax.set_yticks(range(10), [f"{'MIV' if s == 'mimic_iv' else 'eICU'} · {seed}" for s, seed in keys], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Source-selected mortality fusion weight")
    ax.set_title("The three paths need not all contribute", loc="left", fontsize=14, weight="bold", color="#16324f")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, fontsize=8, frameon=False)
    for ax in axes:
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.text(.08, .015, "Left: descriptive five-seed means, no uncertainty interval shown. Right: learned weights, not causal or feature-importance estimates.", fontsize=10, color="#526474")
    fig.subplots_adjust(left=.17, right=.98, top=.9, bottom=.2, wspace=.5)
    save(fig, "los_and_fusion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify()
    if args.verify_only: return
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "maboost-correction09", "font.size": 11})
    OUT.mkdir(parents=True, exist_ok=True)
    architecture(); mortality_heatmap(); paired_effects(); los_and_fusion()
    print(f"Rendered 4 figures (PNG + SVG) to {OUT.relative_to(ROOT)}")


if __name__ == "__main__": main()
