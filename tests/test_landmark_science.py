from pathlib import Path
import sys, tempfile, unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark/src'))
from landmark_protocol import load_protocol
from landmark_jobs import build_job_matrix, training_config_for_job
from v13_etl import _EventAccumulator, _scan_mimic_events, _select_mimic_event_sample, build_mimic_cohort, build_eicu_cohort, build_static_feature_matrix
from v13_analysis import temporal_stress_view
from v13_training import remaining_los_for_rows, matched_augmentation_rows
from v13_model_registry import build_flat_design, MODEL_NAMES
from v13_feature_registry import DYNAMIC_FEATURES, STATIC_FEATURES
def _gzip_csv(frame,path):frame.to_csv(path,index=False,compression='gzip')
class LandmarkScientificTests(unittest.TestCase):
    def test_mimic_fixed_landmark_risk_set_and_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _gzip_csv(pd.DataFrame({
                "stay_id": [1, 2, 3, 4], "subject_id": [11, 12, 13, 14],
                "hadm_id": [21, 22, 23, 24],
                "intime": ["2020-01-01 00:00:00"] * 4,
                "outtime": ["2020-01-01 12:00:00", "2020-01-01 04:00:00", "2020-01-01 12:00:00", "2020-01-01 18:00:00"],
                "first_careunit": ["MICU", "MICU", "SICU", "CICU"],
            }), root / "icustays.csv.gz")
            _gzip_csv(pd.DataFrame({
                "hadm_id": [21, 22, 23, 24],
                "deathtime": ["2020-01-01 08:00:00", None, "2020-01-01 03:00:00", None],
                "hospital_expire_flag": [1, 0, 1, 0],
                "admission_type": ["EMERGENCY"] * 4,
                "insurance": ["Medicare"] * 4,
            }), root / "admissions.csv.gz")
            _gzip_csv(pd.DataFrame({
                "subject_id": [11, 12, 13, 14], "gender": ["M", "F", "M", "F"],
                "anchor_age": [70, 71, 72, 73], "anchor_year": [2020] * 4,
            }), root / "patients.csv.gz")
            result = build_mimic_cohort(
                root, dataset="mimic_iv", minimum_age=65,
                exclude_uncertain_positive=True,
            )
            self.assertEqual(result.frame["stay_key"].tolist(), [1, 4])
            self.assertEqual(result.frame["mortality_label"].tolist(), [1, 0])
            np.testing.assert_allclose(result.frame["los_days"], [0.25, 0.5])

    def test_eicu_landmark_and_static_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _gzip_csv(pd.DataFrame({
                "patientunitstayid": [1, 2, 3],
                "patienthealthsystemstayid": [10, 20, 30],
                "uniquepid": ["p1", "p2", "p3"], "age": [70, 80, 90],
                "gender": ["Male", "Female", "Male"],
                "unittype": ["MICU", "SICU", "CICU"],
                "unitstaytype": ["admit"] * 3,
                "unitdischargestatus": ["Expired", "Alive", "Alive"],
                "unitdischargeoffset": [720, 360, 1800],
            }), root / "patient.csv.gz")
            cohort = build_eicu_cohort(root, minimum_age=65)
            self.assertEqual(cohort.frame["stay_key"].tolist(), [1, 3])
            np.testing.assert_allclose(cohort.frame["los_days"], [0.25, 1.0])
            matrix, audit = build_static_feature_matrix(
                cohort.frame, root / "no_diagnosis", dataset="eicu", max_rows=0
            )
            self.assertEqual(matrix.shape, (2, 12))
            self.assertFalse(audit["diagnosis_tables_opened"])

    def test_fixed_landmark_los_does_not_move(self) -> None:
        arrays = {
            "los": np.asarray([1.5, 2.0], dtype=np.float32),
            "mask": np.asarray([[[1], [0]], [[1], [1]]], dtype=np.uint8),
            "elapsed": np.asarray([[1800, 0], [3600, 18000]], dtype=np.float32),
        }
        labels, anchors = remaining_los_for_rows(arrays, np.asarray([0, 1]))
        np.testing.assert_allclose(labels, [1.5, 2.0])
        np.testing.assert_allclose(anchors, [0.25, 0.25])

    def test_28_stress_views_preserve_risk_set(self) -> None:
        _, _, config, _ = load_protocol()
        rng = np.random.default_rng(42)
        values = rng.normal(size=(5, 96, 40)).astype(np.float32)
        mask = (rng.random(size=values.shape) > 0.7).astype(np.uint8)
        values[mask == 0] = np.nan
        elapsed = np.tile(np.linspace(0, 21600, 96, dtype=np.float32), (5, 1))
        tau = np.full((5, 96), 21600 / 95, dtype=np.float32)
        self.assertEqual(len(config["stress_scenarios"]), 28)
        for scenario in config["stress_scenarios"]:
            view, audit = temporal_stress_view(
                values, mask, tau, elapsed, scenario=scenario, seed=42,
                source_row_ids=np.arange(5),
            )
            self.assertEqual(len(view["values"]), 5)
            self.assertEqual(view["source_row_indices"].tolist(), list(range(5)))
            self.assertTrue(audit["one_row_per_included_stay"])

    def test_event_budgets_count_observed_variable_tokens(self) -> None:
        values = np.arange(1, 13, dtype=np.float32).reshape(1, 3, 4)
        mask = np.ones_like(values, dtype=np.uint8)
        elapsed = np.asarray([[0.0, 60.0, 120.0]], dtype=np.float32)
        tau = np.asarray([[1.0, 60.0, 60.0]], dtype=np.float32)
        for budget in (1, 2, 4, 8):
            view, _ = temporal_stress_view(
                values,
                mask,
                tau,
                elapsed,
                scenario=f"event_budget_{budget}",
                seed=42,
            )
            self.assertEqual(int(view["mask"].sum()), budget)

    def test_sequence_audit_accepts_registered_96_steps(self) -> None:
        accumulator = _EventAccumulator([1], sequence_length=96)
        accumulator.add(1, 30.0, "heart_rate", 80.0)
        self.assertTrue(accumulator.summary()["contract_pass"])
        arrays, audit = accumulator.materialize_arrays()
        self.assertEqual(arrays["values"].shape, (1, 96, 40))
        self.assertTrue(audit["contract_pass"])

    def test_official_carevue_lowercase_event_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = pd.DataFrame({
                "stay_key": [1],
                "hadm_key": [2],
                "intime": [pd.Timestamp("2130-01-01 00:00:00")],
            })
            _gzip_csv(pd.DataFrame({
                "icustay_id": [1], "charttime": ["2130-01-01 01:00:00"],
                "itemid": [211], "valuenum": [80.0], "error": [0],
            }), root / "CHARTEVENTS.csv.gz")
            _gzip_csv(pd.DataFrame({
                "hadm_id": [2], "charttime": ["2130-01-01 02:00:00"],
                "itemid": [50912], "valuenum": [1.1],
            }), root / "LABEVENTS.csv.gz")
            _gzip_csv(pd.DataFrame({
                "icustay_id": [1], "charttime": ["2130-01-01 03:00:00"],
                "itemid": [40055], "value": [125.0],
            }), root / "OUTPUTEVENTS.csv.gz")
            sampled, sample_audit = _select_mimic_event_sample(
                cohort,
                root,
                dataset="mimic_iii_carevue",
                max_stays=1,
                max_rows=100,
            )
            self.assertEqual(sampled["stay_key"].tolist(), [1])
            self.assertEqual(sample_audit["selection_used_labels"], 0)
            audit, selected = _scan_mimic_events(
                cohort,
                root,
                dataset="mimic_iii_carevue",
                max_rows=0,
                sequence_length=96,
            )
            self.assertEqual(selected, [1])
            self.assertEqual(audit["tables"]["chartevents"]["accepted_observations"], 1)
            self.assertEqual(audit["tables"]["labevents"]["accepted_observations"], 1)
            self.assertEqual(audit["tables"]["outputevents"]["accepted_observations"], 1)
            self.assertTrue(audit["contract_pass"])

    def test_matched_augmentation_row_sampling_is_architecture_independent(self) -> None:
        rows = np.arange(1000, dtype=np.int64)
        first = matched_augmentation_rows(
            rows, seed=42, scenario="early_2h", maximum_rows=123
        )
        second = matched_augmentation_rows(
            rows, seed=42, scenario="early_2h", maximum_rows=123
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 123)

    def test_dimensions_and_job_matrix(self) -> None:
        _, _, config, _ = load_protocol()
        values = np.full((3, 96, len(DYNAMIC_FEATURES)), np.nan, dtype=np.float32)
        mask = np.zeros_like(values, dtype=np.uint8)
        static = np.zeros((3, len(STATIC_FEATURES)), dtype=np.float32)
        self.assertEqual(build_flat_design("XGBOOST-FLAT", values, mask, static)[0].shape[1], 200)
        self.assertEqual(build_flat_design("XGBOOST-SUMMARY+STATIC", values, mask, static)[0].shape[1], 212)
        self.assertEqual(build_flat_design("Flat-head", values, mask, static)[0].shape[1], 252)
        self.assertEqual(len(build_job_matrix(config)), 180)
        smoke = build_job_matrix(config, seeds=[42])
        self.assertEqual(len(smoke), 36)
        self.assertEqual(sum(job.registry_role == "primary_registry" for job in smoke), 2 * len(MODEL_NAMES))
        for job in smoke:
            enabled = training_config_for_job(config, job)["temporal_robust_stage2"]["enabled"]
            self.assertEqual(enabled, job.training_regime == "stress_augmented")
