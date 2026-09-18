# Correction09 evaluation report

Assessment date: **18 September 2026**. The completed author run contains **180/180 verified jobs**, with analysis, explanation and reporting artifacts checked against their locked hashes. This is a technical evidence assessment for the planned CMPB manuscript; it is not journal peer review.

[README](README.md) · [Reproduction guide](REPRODUCING.md) · [Aggregate data and provenance](docs/benchmark/correction09/)

All CSV line references count the header as line 1. Numerical results come from the corrected run, not the earlier manuscript.

## Study scope

MABOOST was evaluated for death during the same ICU stay after a fixed six-hour landmark, with remaining ICU length of stay as a secondary outcome. Eligible patients were at least 65 years old and alive and still in the ICU at the landmark. The corrected benchmark comprises 180 training/evaluation jobs, five seeds, 14 model labels, and 28 evaluation scenarios: one six-hour reference plus 27 information-restriction stress views. MIMIC-IV and eICU each serve as a development source and a transfer target; CareVue is a historical target only. The six evaluation routes therefore do not constitute six independent datasets.

Correction09 is a **post-audit methodological correction**, not a prospectively prespecified new study. Recorded-time eligibility is enforced where supported: event time and independent timestamp must both fall within the permitted six-hour window, and observations enter at the independent time. CareVue laboratories and eICU vital tables retain unverified event-time bins; static-field availability remains unverified. eICU laboratory revision time is a conservative final-version cutoff, not verified first-result availability. Neither universal prospective availability nor prediction at earlier clinical landmarks is established. Early prefixes are restricted-input tests on the same six-hour survivor risk set.

The analysis reports 50,833 MIMIC-IV stays (35,841 patients; 4,508 deaths; 8.8683%), 94,702 eICU stays (70,158 patients; 5,980 deaths; 6.3145%), and 10,803 CareVue stays (8,337 patients; 1,497 deaths; 13.8573%). These are full eligible cohorts, not the source test set sizes for an individual seed. Sources: [cohort_characteristics.csv](docs/benchmark/correction09/cohort_characteristics.csv), lines 2–4; [cohort_flow.csv](docs/benchmark/correction09/cohort_flow.csv); [author-run provenance](docs/benchmark/correction09/PROVENANCE.json) and the [public recipe](benchmark/analysis_plan_v13_landmark_amendment_02.json).

## Six-hour reference results

Values below are arithmetic means across five seed-specific evaluations of **stress-augmented MABOOST**. “±” is sample SD across seeds, not a confidence interval. AUPRC is computed as average precision. Remaining LOS is measured in days after the landmark. Transfer targets use their entire eligible cohort; source-internal evaluation uses each seed’s patient-level holdout.

| Route | AUROC mean ± SD | AUPRC mean ± SD | Brier mean | ECE mean | LOS MAE mean | Aggregate CSV line |
|---|---:|---:|---:|---:|---:|---:|
| MIMIC-IV internal | 0.861112 ± 0.004317 | 0.442039 ± 0.008765 | 0.062124 | 0.007426 | 2.257303 | 2753 |
| MIMIC-IV → eICU | 0.835985 ± 0.003224 | 0.345437 ± 0.007972 | 0.050356 | 0.033089 | 2.122563 | 1745 |
| MIMIC-IV → CareVue | 0.813919 ± 0.001736 | 0.450484 ± 0.002796 | 0.098256 | 0.041546 | 3.632573 | 2249 |
| eICU internal | 0.865144 ± 0.004461 | 0.393231 ± 0.015747 | 0.047009 | 0.004770 | 1.760369 | 233 |
| eICU → MIMIC-IV | 0.819999 ± 0.003329 | 0.379526 ± 0.006137 | 0.066802 | 0.015977 | 2.480079 | 1241 |
| eICU → CareVue | 0.795839 ± 0.001853 | 0.431464 ± 0.004890 | 0.103088 | 0.059750 | 3.414729 | 737 |

Source: [aggregate_metrics.csv](docs/benchmark/correction09/aggregate_metrics.csv), filters `model=MABOOST`, `training_regime=stress_augmented`, `scenario=full_6h`. Mortality prevalence differs substantially across targets, so absolute AUPRC should not be used to rank which cohort is inherently easier.

## Statistical interpretation

The primary registered contrasts compare augmented MABOOST against full-only XGBOOST-SUMMARY+STATIC and XGBOOST-FLAT. For both comparators, the paired ensemble AUPRC difference is positive with a pointwise 95% bootstrap interval excluding zero in all six routes under both `full_6h` and `pooled_stress`. Brier differences favor MABOOST in all corresponding contrasts, with BH-adjusted permutation p-values below 0.05. This supports a benchmark-specific statement about these comparators and regimes. It does not isolate architecture from augmentation.

Representative exact contrast values against **full-only XGBOOST-SUMMARY+STATIC** follow. Deltas are MABOOST minus comparator; higher AUPRC is better, lower Brier is better.

| Route | Scope | ΔAUPRC [pointwise 95% CI] | ΔBrier [pointwise 95% CI] | Brier BH-adjusted p | Contrast CSV lines (AP, Brier) |
|---|---|---:|---:|---:|---:|
| MIMIC-IV internal | full_6h | +0.009401 [+0.003034, +0.015933] | -0.000665 [-0.001029, -0.000301] | 0.00229977 | 3, 4 |
| MIMIC-IV internal | pooled_stress | +0.032873 [+0.028894, +0.036927] | -0.002226 [-0.002438, -0.002014] | 0.00009999 | 38, 39 |
| MIMIC-IV → eICU | full_6h | +0.016860 [+0.012435, +0.021190] | -0.000997 [-0.001171, -0.000826] | 0.00029997 | 73, 74 |
| MIMIC-IV → eICU | pooled_stress | +0.033838 [+0.031116, +0.036586] | -0.002155 [-0.002262, -0.002045] | 0.00009999 | 108, 109 |
| MIMIC-IV → CareVue | full_6h | +0.007743 [+0.000964, +0.014594] | -0.000795 [-0.001332, -0.000267] | 0.00229977 | 143, 144 |
| MIMIC-IV → CareVue | pooled_stress | +0.024543 [+0.019550, +0.029540] | -0.001956 [-0.002322, -0.001583] | 0.00009999 | 178, 179 |
| eICU internal | full_6h | +0.017533 [+0.012230, +0.023021] | -0.000795 [-0.001003, -0.000579] | 0.00009999 | 213, 214 |
| eICU internal | pooled_stress | +0.039440 [+0.036281, +0.042659] | -0.001770 [-0.001894, -0.001647] | 0.00009999 | 248, 249 |
| eICU → MIMIC-IV | full_6h | +0.034322 [+0.029379, +0.039289] | -0.001546 [-0.001793, -0.001293] | 0.00009999 | 283, 284 |
| eICU → MIMIC-IV | pooled_stress | +0.046947 [+0.043470, +0.050391] | -0.002524 [-0.002688, -0.002358] | 0.00009999 | 318, 319 |
| eICU → CareVue | full_6h | +0.013018 [+0.006259, +0.019751] | -0.001114 [-0.001608, -0.000619] | 0.00009999 | 353, 354 |
| eICU → CareVue | pooled_stress | +0.026033 [+0.021257, +0.030752] | -0.003094 [-0.003491, -0.002713] | 0.00009999 | 388, 389 |

These intervals concern **seed-ensembled predictions**, not differences between the seed means in the preceding table. Predictions are averaged within each stay/scenario over the available held-out seeds. All five seeds cover each external-target stay; internal analysis combines the union of seed-specific holdouts and has minimum coverage one. Internal unions contain 34,097 stays/23,994 patients for MIMIC-IV and 63,429 stays/47,015 patients for eICU. The code samples patient clusters for paired bootstrap inference, keeping repeated stays and stress views within the patient cluster. Seeds are not independent statistical samples. The 10,000-draw bootstrap CIs are pointwise; they are not multiplicity-adjusted simultaneous intervals.

`full_6h` uses one reference prediction per stay. **Two distinct summaries are labeled `pooled_stress` in the outputs:** [table_5.csv](docs/benchmark/correction09/table_5.csv) averages the 27 scenario-level seed-mean metrics, whereas [primary_contrasts.csv](docs/benchmark/correction09/primary_contrasts.csv) computes each metric after concatenating the 27 stay/scenario prediction sets and ensembling across seeds. Pooled AUROC/AUPRC therefore need not equal a macro average of scenario metrics. Use “mean across 27 stress scenarios” for table 5 and “pooled stress contrast” for the paired inference. Do not attach a pooled CI to a difference between scenario-macro averages. Pooled rows repeat patients; the 2,556,954 MIMIC-IV→eICU stress rows still represent 70,158 patient clusters, not that many independent cases.

Only **Brier** receives paired sign-permutation tests and BH adjustment. The implementation groups BH families by `(evidence_family, source_dataset, scope)`: six primary-model tests, twelve matched-augmentation tests, or three matched-architecture tests per such group. The `bh_significant=0` entries for AUROC, AUPRC and LOS in table 8 mean no BH test was computed; they do not mean all those comparisons failed a significance test. Table 8 also counts intervals excluding zero in either direction, so an exclusion count is not a win count.

A further statistical qualification is required: the Brier bootstrap point estimate is the stay/scenario-weighted mean loss difference, while the sign-permutation test first averages losses within each patient and then averages patient means. For patients with unequal numbers of stays, these are related but differently weighted estimands. Report the Brier permutation result as a separate patient-level test; avoid claiming it is the exact multiplicity-adjusted test of the displayed stay-weighted CI. Sources: `benchmark/src/landmark_analysis.py`, `correction07_ensemble.py`, `correction07_analysis.py`, and `landmark_reporting.py` under correction09.

## Matched factorial evidence and direction-specific exceptions

Four architectures were evaluated in both full-only and stress-augmented regimes with matched splits and registered views: XGBOOST-FLAT, XGBOOST-SUMMARY+STATIC, Hybrid-plain, and MABOOST. The other model labels are evaluated in full-only training. Comparisons with those models therefore do not hold augmentation exposure constant. The matched factorial permits within-architecture augmentation comparisons and augmented-MABOOST versus augmented-SUMMARY+STATIC contrasts; it does not establish a causal contribution for every component.

For MABOOST itself, augmentation raises the **scenario-macro AUPRC** from 0.228141 to 0.264122 for MIMIC-IV→eICU and from 0.237661 to 0.286559 for eICU→MIMIC-IV. The corresponding paired pooled-stress AUPRC deltas are +0.036728 [0.033906, 0.039588] and +0.049979 [0.046366, 0.053593]. Positive pooled-stress AUPRC intervals occur in all six routes. At `full_6h`, however, the augmentation-versus-full-only AUPRC interval includes zero for MIMIC-IV internal (+0.002694 [−0.003515, 0.008568]) and MIMIC-IV→CareVue (+0.004626 [−0.002333, 0.011472]). MIMIC-IV internal Brier also has an interval spanning zero and BH-adjusted p=0.777322. “Augmentation improves every metric in every setting” would be false.

With augmentation held constant, the AUPRC advantage over XGBOOST-SUMMARY+STATIC is smaller. The exact paired contrasts are:

| Route | Full-six-hour ΔAUPRC [95% CI] | Pooled-stress ΔAUPRC [95% CI] | Contrast CSV lines |
|---|---:|---:|---:|
| MIMIC-IV internal | +0.009285 [+0.004678, +0.013929] | +0.007766 [+0.005482, +0.010064] | 33, 68 |
| MIMIC-IV → eICU | +0.011499 [+0.008363, +0.014655] | +0.009067 [+0.007183, +0.010956] | 103, 138 |
| MIMIC-IV → CareVue | +0.010062 [+0.005645, +0.014586] | +0.008922 [+0.006091, +0.011767] | 173, 208 |
| eICU internal | +0.008735 [+0.004726, +0.012850] | +0.010555 [+0.008634, +0.012477] | 243, 278 |
| eICU → MIMIC-IV | -0.000967 [-0.003904, +0.001973] | +0.004153 [+0.002530, +0.005710] | 313, 348 |
| eICU → CareVue | +0.006220 [+0.001257, +0.011232] | +0.003815 [+0.000880, +0.006703] | 383, 418 |

The reverse direction is the clearest counterexample to universal architecture superiority. For eICU→MIMIC-IV at `full_6h`, augmented MABOOST versus augmented SUMMARY+STATIC has ΔAUROC −0.003590 [−0.004625, −0.002497] and ΔAUPRC −0.000967 [−0.003904, 0.001973], although Brier favors MABOOST. The seed-mean AUPRC difference has the opposite sign (0.379526 versus 0.375926), which is possible because averaging predictions changes the estimand. Both values are legitimate only with their proper labels.

Moreover, augmented XGBOOST-FLAT has higher seed-mean AUROC/AUPRC in eICU→MIMIC-IV at `full_6h` (0.828003/0.386526 versus MABOOST 0.819999/0.379526), and higher scenario-macro AUROC/AUPRC with lower Brier under stress (0.762012/0.293926/0.071960 versus 0.756343/0.286559/0.072141). No direct registered paired contrast against **augmented XGBOOST-FLAT** is provided, so these are descriptive comparisons, not newly tested superiority claims. MIMIC-IV→eICU full-six-hour AUROC versus full-only SUMMARY+STATIC also has a CI spanning zero: +0.000550 [−0.001197, 0.002299].

## Calibration, CareVue and the secondary LOS endpoint

Source-validation isotonic calibration does not guarantee external calibration. The main-model ECE rises from 0.007426 on MIMIC-IV internal evaluation to 0.033089 on eICU and 0.041546 on CareVue; eICU-source ECE is 0.004770 internally, 0.015977 on MIMIC-IV and 0.059750 on CareVue. Corresponding CareVue calibration intercept/slope means are 0.263728/0.865482 from MIMIC-IV and 0.456663/0.849429 from eICU. These results show residual calibration shift despite improved Brier against selected baselines. A lower Brier score alone is not proof of clinically adequate calibration.

CareVue is restricted to the historical CareVue cohort, not full MIMIC-III. It is useful as an additional historical transfer test, with 1,497 deaths and substantial prevalence and measurement differences. It is not a separate training direction and does not validate universal portability. Unverified laboratory timing remains a direct limitation; the two source→CareVue tests share the same target patients and are not independent replications.

LOS results are mixed and should be retained as a secondary, qualified finding. Against full-only SUMMARY+STATIC at `full_6h`, the paired LOS-MAE delta is +0.085406 days [0.082286, 0.088528] for MIMIC-IV→eICU, +0.047500 [0.044288, 0.050745] for eICU→MIMIC-IV, and +0.102168 [0.090506, 0.113521] for MIMIC-IV→CareVue: all favor the comparator. eICU→CareVue is inconclusive at −0.004687 [−0.011473, 0.002037]. Internal LOS-MAE contrasts favor MABOOST. LOS RMSE and MAE do not always favor the same model; there is no consistent claim of joint mortality-and-LOS superiority.

## Component evidence and interpretability boundaries

The mortality fusion allows weights to lie on the simplex boundary. In the five eICU-source augmented runs, XGBOOST-FLAT receives weight zero for seeds 42, 137, 2025 and 31415; its weight is 0.30 only for seed 2718. Thus four eICU-source runs effectively use two active mortality pathways. Mean eICU-source weights are approximately 0.51 Hybrid, 0.06 XGBOOST-FLAT and 0.43 Flat-head; MIMIC-IV-source means are 0.51, 0.19 and 0.30, with all three positive for all five seeds. This is partial branch collapse/selection, not collapse of the full predictor. The result contradicts a claim that every run derives complementary benefit from all three branches. The file concerns mortality; do not transfer these weights to the LOS fusion without separate evidence. Source: [fusion_pathway_contributions.csv](docs/benchmark/correction09/fusion_pathway_contributions.csv), lines 2–31.

Standalone Mamba-z-only has full-six-hour mean AUROC between 0.502136 and 0.524759 across the six routes. This is a material limitation of the implemented representation-only ablation. Together with the strong Flat-head baseline, it prevents attributing the overall advantage to powerful standalone temporal embeddings or proving the encoder is essential. The matched comparison against SUMMARY+STATIC also changes multiple ingredients; it is not a clean encoder-only intervention. These results diagnose the evaluated implementation, not the Mamba model family generally.

TreeSHAP is computed for the **Flat-head mortality direct-clinical pathway**, using 252 input features on 512 sampled source-test stays per source/seed model. The highest mean absolute SHAP features are mean respiratory rate (0.197767), age (0.147768), last GCS total (0.145383), last FiO2 (0.138516), and last heart rate (0.129955), in the component model’s log-odds scale. These are associations in that pathway and those samples, not causal predictors or a global explanation of the final calibrated model.

The pathway values are `weight × raw component probability`, computed **before isotonic calibration**; they do not add up to an attribution decomposition of calibrated probability. Occlusion reports changes in model predictions, not measured clinical effects or changes in predictive accuracy. The implementation masks values while retaining the original tau array; it is best described as fixed-timing input masking, not a full re-simulation of the acquisition process. Temporal intervals include both endpoints, so boundaries at 2 h and 4 h can participate in adjacent occlusions. These restrictions belong in the explainability caption or linked methods text.

## Safe headline and claims to avoid

Suggested headline: “In a corrected retrospective benchmark of older ICU patients at a six-hour landmark, stress-augmented MABOOST improved mortality average precision and Brier relative to the registered full-only tabular comparators across both cross-dataset directions and the historical CareVue target. Matched-regime comparisons show a smaller additional benefit, with reverse-transfer exceptions, residual calibration shift, and mixed remaining-LOS results.”

Avoid claims of state-of-the-art performance, clinical readiness, universal temporal availability, earlier clinical prediction, causal interpretability, unique architecture-driven robustness, or statistically significant superiority across every model/metric. No target-site prospective validation, decision-utility experiment, or exhaustive equal-budget optimization of competing model families is established by these artifacts. Do not describe the correction as prespecified. Do not describe 27 repeated stress views as 27 independent clinical cohorts.

## Verification and release boundaries

All 180 completed job inventories were verified, together with the run identity,
the analysis/explanation/reporting predecessor chain, the 10 canonical CSVs,
explanation outputs, 5 reporting figures, and 23 reporting table assets. The
explanation figure belongs to phase F and is separate from the 5 phase-G figures.
The four README illustrations are redrawn from reviewed aggregates; their source
script is [tools/render_results.py](tools/render_results.py).

The completed reporting manifest uses logical names for four tables (for example,
`table_1.csv` refers to `table_1_cohort_shift.csv`). Actual bytes and sidecars match
the registered aliases. The older generic verifier failed to resolve these names;
the portable release corrects that verification behavior without changing metrics.

The computational run emitted 15,120 per-seed metric rows and 420 paired contrasts.
Successful training/evaluation jobs consumed **109.72 cumulative wall hours**;
this excludes ETL, interrupted attempts, and downstream analysis/reporting.
No new clinical training was launched to prepare this release.

[PROVENANCE.json](docs/benchmark/correction09/PROVENANCE.json) binds each published
aggregate file to the author output. The portable raw-data recipe creates its own
identity and does not require the author's private audit, arrays, or split files.
The scientific timestamp selection, cohort construction and deterministic split
algorithms are retained; historical comparisons to author-only artifacts are not
represented as validations performed by a new user. The complete clinical run
has not been independently repeated on this public checkout.

These results are ready to support manuscript drafting with the qualifications
above. A submission still requires a corrected manuscript, author/institutional
metadata and ethics statements, citation checks, and review of the clinical and
statistical interpretation. This repository does not imply that those manuscript
steps or journal acceptance are complete.
