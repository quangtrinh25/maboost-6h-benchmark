# MABOOST: fixed-6-hour ICU benchmark

Code for reproducing a temporal–tabular benchmark for ICU mortality and remaining
length of stay in adults aged 65 and older. Every evaluation stay is alive and
still in the ICU at 6 hours; predictors use only the first 6 hours. Remaining LOS
starts at that same landmark.

This repository distributes **code, executable configuration, tests and this
README only**. It contains no clinical data, checkpoints, prediction files,
benchmark results, figures, internal documentation or manuscript. Running the
pipeline generates your own artifacts locally; generated directories are ignored
by Git. Dataset access must be obtained separately.

## Benchmark

- MIMIC-IV **3.1** and eICU **2.0** each act as a development source and an
  external target. The official **MIMIC-III CareVue subset 1.4** is a historical
  external target only. The full MIMIC-III release is not interchangeable with
  this subset because of overlap with MIMIC-IV.
- **40 temporal variables**, **12 static variables**, sequences of at most
  **96 steps**, and patient-grouped **60/20/20** source splits.
- Five seeds: **42, 137, 2025, 2718, 31415**.
- Fourteen model labels: MABOOST, Hybrid-plain, Mamba-z-only, Flat-head,
  XGBOOST-FLAT, XGBOOST-SUMMARY+STATIC, GRU-D, LSTM, Transformer, TCN, SAnD,
  STraTS, InterpNet and DuETT.
- The primary registry compares stress-augmented MABOOST with full-only
  comparators. A separate matched factorial evaluates both training regimes for
  MABOOST, Hybrid-plain and the two XGBoost baselines: **180 training jobs** total.
- **28 evaluation scenarios**: the full-6-hour reference plus 27 deterministic
  transformations of observation availability. Every scenario retains the same
  6-hour risk set. Pooled-stress inference excludes the reference scenario.
- **10,000 patient-clustered paired bootstrap replicates**; **10,000 Brier
  sign-permutation replicates**, with the configured BH families. AUROC, AUPRC
  and LOS confidence intervals are pointwise. Source-internal paired inference
  uses the union of seed-specific test stays and averages available held-out
  predictions; external stays use all five seeds.

Preprocessing, fitting, selection, fusion, calibration and thresholds use source
data only. Target labels remain separate until predictions and source choices
are locked. Untimestamped diagnosis-derived Elixhauser features are excluded.

## Environment

Use **Linux x86-64**, **Python 3.11**, an NVIDIA GPU with a CUDA-compatible driver,
the **CUDA 12.8 development toolkit** (`nvcc`) and a C/C++ build toolchain for
Mamba. Install `rustup` before running setup. The Rust crate pins **1.85.1** and
has no external crate dependencies. The full workload is substantial; preflight
checks CUDA, required input files and available disk space. Runtime monitoring
cleans up at 85% system RAM and stops at 95%.

```bash
git clone https://github.com/quangtrinh25/maboost-6h-benchmark.git
cd maboost-6h-benchmark
bash setup.sh
source .venv/bin/activate
```

`setup.sh` installs PyTorch 2.7.0 from the CUDA 12.8 wheel index, the pinned
scientific dependencies, Mamba-SSM 2.3.1 and the pinned Rust toolchain. It does
not download any datasets. An existing matching environment may be used instead.

## Local data layout

Obtain the governed resources directly:
[MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/),
[eICU 2.0](https://physionet.org/content/eicu-crd/2.0/), and
[MIMIC-III CareVue 1.4](https://physionet.org/content/mimic3-carevue/1.4/).
Place or symlink the following compressed CSV files into these directories
relative to the checkout. For MIMIC-IV, gather files from the `hosp` and `icu`
subdirectories into the single input directory below.

```text
data/raw/mimiciv/
  icustays.csv.gz       admissions.csv.gz     patients.csv.gz
  chartevents.csv.gz    labevents.csv.gz       outputevents.csv.gz
data/raw/eicu/
  patient.csv.gz        vitalPeriodic.csv.gz  vitalAperiodic.csv.gz
  nurseCharting.csv.gz  lab.csv.gz            respiratoryCharting.csv.gz
  intakeOutput.csv.gz
data/raw/mimic3-carevue/1.4/
  ICUSTAYS.csv.gz       ADMISSIONS.csv.gz     PATIENTS.csv.gz
  CHARTEVENTS.csv.gz    LABEVENTS.csv.gz      OUTPUTEVENTS.csv.gz
```

The CareVue loader accepts the release's lowercase header names inside those
files. Preserve the documented filenames; the loader probes them explicitly.
Do not substitute full MIMIC-III for the official CareVue non-overlap resource.

## Reproduce the complete benchmark

After setup and data placement, from the repository root:

```bash
source .venv/bin/activate
python -B reproduce.py all
```

For individual stages and visible checkpoints, the equivalent sequence is:

```bash
python -B reproduce.py init
python -B reproduce.py check
python -B reproduce.py build-kernel
python -B reproduce.py preflight
python -B reproduce.py etl
python -B reproduce.py train --resume
python -B reproduce.py analyze
python -B reproduce.py explain
python -B reproduce.py report
```

`init` records a **new local run identity** bound to code and configuration.
It does not copy or claim the original author's approvals or qualification.
`build-kernel` compiles Rust, runs its unit/contract tests and compares all five
metric kernels against Python on concrete synthetic clustered draws at one and
multiple threads. Analysis requires this local qualification and a verified
complete training matrix.

Training uses one process per job. `train --resume` verifies completed artifacts
before skipping them; incomplete attempts are archived locally. Job logs are in
`results/benchmark/logs/training/`. To inspect or run one registered job:

```bash
python -B reproduce.py matrix
python -B reproduce.py train --job mimic_iv__xgboost_flat__full_only__seed42
```

A single job does not unlock full analysis. Interrupted bootstrap work resumes
from hash-verified shards and saved RNG states when `analyze` is rerun. A completed
canonical analysis is verified and reused. Changed code/configuration invalidates
an initialized run: use a fresh checkout for a new experiment rather than mixing
artifacts. Maintain the same CPU affinity/thread count across kernel qualification
and analysis.

## Locally generated outputs

| Directory | Generated content |
| --- | --- |
| `benchmark/private/processed/` | Feature arrays and separately sealed labels |
| `benchmark/private/jobs/` | Model states, split/choice locks and evaluation arrays |
| `benchmark/private/` | Run identity, kernel qualification and bootstrap checkpoints |
| `benchmark/superseded/` | Archived incomplete attempts |
| `results/benchmark/canonical/` | Per-seed/aggregate metrics, paired contrasts, cohorts and subgroups |
| `results/benchmark/explainability/` | Aggregate SHAP, pathway and occlusion summaries |
| `results/benchmark/figures/`, `tables/` | Figures and tables generated by your run |
| `results/benchmark/phase_reports/` | Stage reports and SHA-256 bindings |

The architecture figure is generated from Python code in this release. Its layout
is a presentation adaptation; it does not affect numerical evaluation.

## Verification without clinical data

```bash
python -B -m unittest discover -s tests
python -B reproduce.py smoke
```

The scientific tests cover fixed-landmark eligibility/outcomes, static-feature
dimensions, all 28 stress views, event budgets, deterministic augmentation, and
seed-dependent ensemble alignment. Integrity tests reject changed or missing
code. The smoke command creates an isolated temporary checkout with generated
synthetic arrays, runs bounded XGBoost-flat and LSTM training, reloads their saved
models, evaluates all 28 scenarios over three routes, verifies prediction/label
locks and checks completed-job resume. Its artifacts cannot enter your full run.

These checks validate the portable execution paths, not full-cohort numerical
reproduction. The complete 180-job clinical workload has not been rerun for this
code-only release. Hardware, CUDA libraries and compiler differences may affect
numerical reproducibility. No performance results or manuscript claims are
published in this repository.

## Source layout

`reproduce.py` is the public CLI. `benchmark/src/` contains ETL, models/training
adapters, target-label separation, stress evaluation, checkpointed inference,
report generation and fresh-run integrity checks. `src/models/` and
`src/training/` contain the shared neural implementations. The corrected LOS,
SAnD, STraTS and seed-alignment implementations are retained. `benchmark/config/`
contains the executable recipe; `benchmark/rust_analysis_kernel/` contains the
Rust metric engine. `CODE_MANIFEST.json` binds the distributed source/configuration.

Upstream model and method references are retained in source comments. This is
research reproduction software; it does not establish clinical utility or an
independent implementation of every upstream baseline repository.
