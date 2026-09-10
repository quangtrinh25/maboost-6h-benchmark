from pathlib import Path
import sys, unittest, importlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark/src'))
class AnalysisEngineCorrection07DrawTests(unittest.TestCase):
    """Catch RNG-stream changes caused by batching or integer-dtype choices."""

    def _api(self):
        try:
            return importlib.import_module("correction07_bootstrap")
        except ModuleNotFoundError:
            self.fail("production correction07_bootstrap API is missing")

    def test_draw_batches_match_legacy_rng_stream(self) -> None:
        api = self._api()
        legacy_rng = np.random.default_rng(137)
        expected = np.stack(
            [legacy_rng.integers(0, 7, size=7) for _ in range(11)]
        )
        for batch_size in (1, 2, 4, 11):
            rng = np.random.default_rng(137)
            actual = np.concatenate(
                [
                    api.generate_draw_batch(rng, min(batch_size, 11 - start), 7)
                    for start in range(0, 11, batch_size)
                ]
            )
            np.testing.assert_array_equal(actual, expected)

    def test_array_hash_binds_dtype_shape_and_bytes(self) -> None:
        api = self._api()
        base = np.asarray([1, 2, 3], dtype=np.uint32)
        self.assertEqual(api.sha256_array(base), api.sha256_array(base.copy()))
        self.assertNotEqual(api.sha256_array(base), api.sha256_array(base.astype(np.uint64)))
        self.assertNotEqual(api.sha256_array(base), api.sha256_array(base.reshape(3, 1)))


class AnalysisEngineCorrection07EnsembleTests(unittest.TestCase):
    """Catch row-order, seed-alignment, or float32-averaging drift."""

    def _api(self):
        try:
            return importlib.import_module("correction07_ensemble")
        except ModuleNotFoundError:
            self.fail("production correction07_ensemble API is missing")

    @staticmethod
    def _payload(seed_index: int, *, model_offset: float) -> dict[str, np.ndarray]:
        mortality_probability = (
            np.arange(84, dtype=np.float32).reshape(3, 28) / 100.0
            + np.float32(seed_index / 1000.0 + model_offset)
        )
        los_prediction = (
            np.arange(84, dtype=np.float32).reshape(3, 28) / 10.0
            + np.float32(seed_index / 100.0 + model_offset)
        )
        return {
            "mortality": np.array([0, 1, 0], dtype=np.int8),
            "los": np.array([1.5, 4.0, 2.25], dtype=np.float32),
            "stay_keys": np.array(["s2", "s10", "a"]),
            "group_keys": np.array(["p2", "p10", "p2"]),
            "mortality_probability": mortality_probability,
            "los_prediction": los_prediction,
        }

    def _payloads(self) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
        left = [self._payload(seed, model_offset=0.0) for seed in range(5)]
        right = [self._payload(seed, model_offset=0.05) for seed in range(5)]
        order = np.array([2, 0, 1])
        for collection in (left, right):
            for seed in (1, 3):
                collection[seed] = {
                    name: value[order] for name, value in collection[seed].items()
                }
        return left, right

    @staticmethod
    def _legacy(payloads_a, payloads_b, scope):
        accumulator: dict[str, dict[str, object]] = {}
        scenarios = [0] if scope == "full_6h" else list(range(1, 28))
        for a, b in zip(payloads_a, payloads_b, strict=True):
            for row in range(len(a["mortality"])):
                for scenario in scenarios:
                    key = f"{a['stay_keys'][row]}|{scenario}"
                    entry = accumulator.setdefault(
                        key,
                        {
                            "mortality": int(a["mortality"][row]),
                            "los": float(a["los"][row]),
                            "group": str(a["group_keys"][row]),
                            "a": [],
                            "b": [],
                            "a_los": [],
                            "b_los": [],
                        },
                    )
                    entry["a"].append(float(a["mortality_probability"][row, scenario]))
                    entry["b"].append(float(b["mortality_probability"][row, scenario]))
                    entry["a_los"].append(float(a["los_prediction"][row, scenario]))
                    entry["b_los"].append(float(b["los_prediction"][row, scenario]))
        ordered = [accumulator[key] for key in sorted(accumulator)]
        return {
            "mortality": np.asarray([item["mortality"] for item in ordered], dtype=np.int8),
            "los": np.asarray([item["los"] for item in ordered], dtype=np.float32),
            "group": np.asarray([item["group"] for item in ordered]),
            "a": np.asarray([np.mean(item["a"]) for item in ordered], dtype=np.float32),
            "b": np.asarray([np.mean(item["b"]) for item in ordered], dtype=np.float32),
            "a_los": np.asarray(
                [np.mean(item["a_los"]) for item in ordered], dtype=np.float32
            ),
            "b_los": np.asarray(
                [np.mean(item["b_los"]) for item in ordered], dtype=np.float32
            ),
        }

    def test_vectorized_ensemble_exactly_matches_legacy_order_and_values(self) -> None:
        api = self._api()
        payloads_a, payloads_b = self._payloads()
        for scope in ("full_6h", "pooled_stress"):
            legacy = self._legacy(payloads_a, payloads_b, scope)
            actual = api.prepare_seed_ensemble(payloads_a, payloads_b, scope=scope)
            np.testing.assert_array_equal(actual.mortality, legacy["mortality"])
            np.testing.assert_array_equal(actual.los, legacy["los"])
            np.testing.assert_array_equal(actual.predictions_a, legacy["a"])
            np.testing.assert_array_equal(actual.predictions_b, legacy["b"])
            np.testing.assert_array_equal(actual.los_predictions_a, legacy["a_los"])
            np.testing.assert_array_equal(actual.los_predictions_b, legacy["b_los"])
            self.assertEqual(actual.minimum_seed_coverage, 5)
            self.assertEqual(actual.cluster_count, 2)

    def test_ensemble_rejects_identity_or_label_drift(self) -> None:
        api = self._api()
        for field, replacement in (
            ("mortality", np.array([1, 1, 0], dtype=np.int8)),
            ("los", np.array([8.0, 4.0, 2.25], dtype=np.float32)),
            ("group_keys", np.array(["other", "p10", "p2"])),
        ):
            payloads_a, payloads_b = self._payloads()
            payloads_b[4][field] = replacement
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "alignment failure"):
                    api.prepare_seed_ensemble(payloads_a, payloads_b, scope="full_6h")

    def test_seed_specific_holdouts_match_legacy_union_and_available_seed_mean(self):
        left, right = self._payloads()
        # Each seed has its own source holdout, paired across the two models.
        for seed, rows in enumerate(([0], [0, 1], [1, 2], [1], [0, 2])):
            for collection in (left, right):
                collection[seed] = {k: v[rows] for k, v in collection[seed].items()}
        for scope in ("full_6h", "pooled_stress"):
            expected = self._legacy(left, right, scope)
            actual = self._api().prepare_seed_ensemble(left, right, scope=scope)
            for field, legacy_field in (
                ("mortality", "mortality"), ("los", "los"),
                ("predictions_a", "a"), ("predictions_b", "b"),
                ("los_predictions_a", "a_los"), ("los_predictions_b", "b_los"),
            ):
                np.testing.assert_array_equal(getattr(actual, field), expected[legacy_field])
            self.assertEqual(actual.minimum_seed_coverage, 1)
            self.assertEqual(actual.cluster_count, 2)

    def test_seed_specific_holdouts_still_require_paired_model_membership(self):
        left, right = self._payloads()
        right[2] = {k: v[:2] for k, v in right[2].items()}
        with self.assertRaisesRegex(RuntimeError, "alignment failure"):
            self._api().prepare_seed_ensemble(left, right, scope="full_6h")

    def test_ensemble_rejects_wrong_seed_count_duplicate_keys_and_shape(self) -> None:
        api = self._api()
        payloads_a, payloads_b = self._payloads()
        with self.assertRaisesRegex(ValueError, "exactly five"):
            api.prepare_seed_ensemble(payloads_a[:4], payloads_b[:4], scope="full_6h")
        duplicate_a, duplicate_b = self._payloads()
        duplicate_a[0]["stay_keys"][1] = duplicate_a[0]["stay_keys"][0]
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            api.prepare_seed_ensemble(duplicate_a, duplicate_b, scope="full_6h")
        bad_a, bad_b = self._payloads()
        bad_a[2]["mortality_probability"] = bad_a[2]["mortality_probability"][:, :27]
        with self.assertRaisesRegex(ValueError, "28 scenario"):
            api.prepare_seed_ensemble(bad_a, bad_b, scope="pooled_stress")
