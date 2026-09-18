# Reproducing the fixed-6-hour benchmark

[Project and results](README.md) · [Evidence assessment](EVALUATION.md)

This guide separates three activities: redraw published aggregate figures, run a
small synthetic demonstration, and reproduce the full benchmark from governed
clinical datasets. Synthetic outputs are never evidence of clinical performance.

## 1. Environment

The full benchmark uses Linux x86-64, Python 3.11, an NVIDIA GPU, a compatible
CUDA driver, the CUDA 12.8 development toolkit (`nvcc`), and a C/C++ toolchain.
Install `rustup` first. The analysis crate pins Rust 1.85.1 and has no external
crate dependencies.

```bash
git clone https://github.com/quangtrinh25/maboost-6h-benchmark.git
cd maboost-6h-benchmark
bash setup.sh
source .venv/bin/activate
python -B reproduce.py check
```

Setup installs PyTorch 2.7.0 using the CUDA 12.8 wheel index, Mamba-SSM 2.3.1, the
pinned scientific packages, and the Rust toolchain. It does not download clinical
data. An existing compatible environment may be used instead. Record the Git
commit, hardware, CUDA/compiler versions, and package versions with your run.

The author run's 180 successful training/evaluation jobs consumed about **109.72
wall hours cumulatively**. This excludes ETL, interrupted attempts, and downstream
stages, and is not a runtime promise for another machine. Preflight requires at
least 20 GB free disk; allow additional space for raw data, caches, and retry
archives. Guardian requests cleanup at 85% system RAM and stops at 95%. Keep other
memory-heavy workloads off the machine during training.

## 2. Redraw the README figures without clinical data

The committed inputs in `docs/benchmark/correction09/` are aggregate statistics.
After installing the plotting dependencies (NumPy, pandas, Matplotlib), run:

```bash
python -B tools/render_results.py --verify-only
python -B tools/render_results.py
```

The first command checks the published data hashes. The second generates four
figures as PNG and SVG in `docs/figures/`: architecture, all-model mortality AUPRC,
paired AUPRC effects, and LOS/fusion diagnostics. This reproduces the presentation
of saved results; it does not retrain or regenerate their underlying estimates.

## 3. Synthetic walkthrough: demonstrate execution without clinical data

From the configured environment:

```bash
python -B -m unittest discover -s tests -v
python -B reproduce.py matrix
python -B reproduce.py smoke
```

The matrix lists the 180 registered jobs without training them. The smoke creates
an isolated temporary checkout with explicitly synthetic arrays, trains bounded
XGBOOST-FLAT and LSTM jobs, reloads their saved models, evaluates all 28 scenarios
on three routes, checks prediction/label locks, and confirms that completed jobs
are skipped on resume. It exercises a small part of the execution path, not all
14 model labels or the full inference workload. It does not use your real
`data/raw/` inputs or populate production clinical completions.

The expected flow is:

```text
Synthetic fixtures
    → isolated smoke checkout and synthetic-only marker
    → local code/run registration
    → two bounded model jobs
    → 28 scenarios × three routes
    → saved model reload and locked-artifact checks
    → completed-job resume check
    → smoke report
```

This diagram describes the workflow; it is not a fabricated console transcript
or a claim that generated toy metrics equal the published results. Timestamp
regression tests separately exercise synthetic raw-table fixtures for all three
source formats and check availability filtering before aggregation/truncation.

## 4. Obtain and place the governed datasets

Obtain access directly to [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/),
[eICU 2.0](https://physionet.org/content/eicu-crd/2.0/), and the official
[MIMIC-III CareVue 1.4 subset](https://physionet.org/content/mimic3-carevue/1.4/).
The repository provides no data access credentials or dataset downloads.

Place or symlink these compressed CSV files relative to the checkout. For
MIMIC-IV, collect the files from `hosp` and `icu` into the input directory below.

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

The CareVue loader accepts the release's lowercase CSV headers inside the
specified filenames. Do not replace this resource with full MIMIC-III.
Keep independent recording/revision timestamp columns: correction09 needs them
and does not silently fall back to event time when a required field is missing.
Sources that genuinely have no such field follow the explicitly unverified
policy described in the README.

## 5. Full reproduction from a fresh checkout

```bash
source .venv/bin/activate
python -B -u reproduce.py all
```

The equivalent stage sequence makes the checkpoints explicit:

```bash
python -B reproduce.py init
python -B reproduce.py check
python -B reproduce.py build-kernel
python -B reproduce.py preflight
python -B -u reproduce.py etl
python -B -u reproduce.py train --resume
python -B -u reproduce.py analyze
python -B -u reproduce.py explain
python -B -u reproduce.py report
```

`init` binds a **new local run** to distributed code/configuration. The public
policy reconstructs recorded-time eligibility directly from raw data, hashes
inputs, and records timestamp category counts. It does not require the author's
private parent audit, old arrays, patient split files, or prior approval records.
Cohort and deterministic split construction follow the benchmark algorithm.

`build-kernel` compiles Rust, runs its tests, and checks all five metric kernels
against Python on identical synthetic clustered draws at one and multiple
threads. Keep CPU affinity/thread availability consistent between qualification
and analysis. Full analysis requires verified completion of all 180 jobs.

## 6. Interruption and resource-stop recovery

Resume **from the interrupted stage**, preserving completed upstream reports.
For a training interruption, including Guardian exit 86:

```bash
# First free sufficient RAM and make sure the previous runner has stopped.
python -B -u reproduce.py train --resume
python -B -u reproduce.py analyze
python -B -u reproduce.py explain
python -B -u reproduce.py report
```

Training verifies each completed artifact before skipping its job. An incomplete
attempt is archived before a fresh attempt. This is job-level recovery; a partial
encoder file does not provide an epoch-level restart. Do not delete live job
locks or relock code to force a resume. Logs are stored in
`results/benchmark/logs/training/<job-id>.log`; the final traceback may only be a
wrapper around the cause in that log.

To run one registered job:

```bash
python -B -u reproduce.py train --job mimic_iv__xgboost_flat__full_only__seed42
```

A single job does not unlock full analysis. If analysis is interrupted, rerun
`python -B -u reproduce.py analyze`; bootstrap shards and RNG states are verified
and reused. An already completed analysis is verified and skipped. After an
explanation or reporting interruption, restart that stage and any unfinished
successor. Avoid rerunning completed upstream stages solely to retry a later
stage: rewriting their reports can invalidate downstream provenance.

Use a fresh checkout/output tree after any recipe or locked-code change. The
old event-time code-only release is not the recipe for the correction09 results.
Do not run `tools/lock_code.py` on an initialized run.

## 7. Outputs and interpretation

| Location | Contents |
| --- | --- |
| `benchmark/private/processed/` | Feature arrays, separately sealed labels, raw-input provenance and ETL audits |
| `benchmark/private/jobs/` | Models, source choices, completion locks and private predictions |
| `benchmark/private/` | Run registration, source splits, kernel qualification and bootstrap state |
| `benchmark/superseded/` | Archived incomplete attempts |
| `results/benchmark/canonical/` | Seed metrics, aggregates, paired contrasts, cohorts and subgroups |
| `results/benchmark/explainability/` | Aggregate pathway, SHAP and occlusion summaries |
| `results/benchmark/figures/`, `tables/` | Generated figures and reporting tables |
| `results/benchmark/phase_reports/` | Stage status and SHA-256 provenance |

Keep these generated clinical outputs local. The committed publication assets
are an explicit, reviewed aggregate subset and are separate from a new run's
outputs. The result renderer reads the published snapshot by default.

Read [EVALUATION.md](EVALUATION.md) before comparing results: seed means, pooled
ensemble contrasts, and scenario-macro summaries estimate different quantities.
Only Brier receives permutation/BH tests. Cohort counts, raw hashes and policy
are useful first diagnostics if a fresh run differs. Hardware, CUDA libraries,
compiler versions, and numerical behavior can affect exact values; a full
independent replication of the portable checkout is not claimed.
