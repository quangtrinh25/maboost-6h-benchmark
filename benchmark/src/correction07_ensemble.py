"""Correction-07 vectorized five-seed ensemble preparation.

The legacy implementation accumulated one Python dictionary entry per
stay/scenario. This preserves its lexical ordering and float32 outputs while
keeping only dense numeric arrays at full cohort scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Mapping, Sequence

import numpy as np


EXPECTED_SEEDS = 5
EXPECTED_SCENARIOS = 28
_REQUIRED_FIELDS = frozenset(
    {
        "mortality", "los", "stay_keys", "group_keys",
        "mortality_probability", "los_prediction",
    }
)

# Hash-keyed caches retain integer layouts only, never labels/predictions.
_LAYOUT_CACHE: dict[tuple[str, tuple[int, ...]], tuple[np.ndarray, np.ndarray, str]] = {}
_ROW_ALIGNMENT_CACHE: dict[tuple[str, str], np.ndarray] = {}


@dataclass(frozen=True)
class PreparedEnsemble:
    mortality: np.ndarray
    los: np.ndarray
    cluster_codes: np.ndarray
    predictions_a: np.ndarray
    predictions_b: np.ndarray
    los_predictions_a: np.ndarray
    los_predictions_b: np.ndarray
    minimum_seed_coverage: int
    cluster_count: int
    alignment_sha256: str


def _hash_strings(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _hash_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(struct.pack("<Q", contiguous.ndim))
    for dimension in contiguous.shape:
        digest.update(struct.pack("<Q", dimension))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _validate_payload(payload: Mapping[str, np.ndarray], *, name: str) -> np.ndarray:
    missing = _REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    stay_keys = np.asarray(payload["stay_keys"])
    if stay_keys.ndim != 1:
        raise ValueError(f"{name} stay_keys must be one-dimensional")
    row_count = len(stay_keys)
    if len({str(key) for key in stay_keys}) != row_count:
        raise RuntimeError(f"{name} contains duplicate stay keys")
    if any("|" in str(key) for key in stay_keys):
        raise RuntimeError(f"{name} stay key contains reserved delimiter")
    for field in ("mortality", "los", "group_keys"):
        values = np.asarray(payload[field])
        if values.ndim != 1 or len(values) != row_count:
            raise ValueError(f"{name} {field} row shape mismatch")
    for field in ("mortality_probability", "los_prediction"):
        values = np.asarray(payload[field])
        if values.shape != (row_count, EXPECTED_SCENARIOS):
            raise ValueError(f"{name} {field} must have exactly 28 scenario columns")
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name} {field} contains non-finite values")
    return stay_keys


def _layout(
    stay_keys: np.ndarray, scenarios: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, str]:
    sequence_hash = _hash_strings(stay_keys)
    cache_key = (sequence_hash, scenarios)
    cached = _LAYOUT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    # Sorting stay + '|' exactly matches grouping from legacy stay|scenario
    # strings, including IDs where one is a prefix of another.
    row_order = np.asarray(
        sorted(range(len(stay_keys)), key=lambda index: f"{stay_keys[index]}|"),
        dtype=np.int64,
    )
    scenario_order = np.asarray(
        sorted(scenarios, key=lambda scenario: str(scenario)), dtype=np.int64
    )
    result = (row_order, scenario_order, _hash_strings(stay_keys[row_order]))
    _LAYOUT_CACHE[cache_key] = result
    return result


def _alignment(
    reference_keys: np.ndarray, candidate_keys: np.ndarray, *, reference_hash: str
) -> np.ndarray:
    candidate_hash = _hash_strings(candidate_keys)
    cache_key = (reference_hash, candidate_hash)
    cached = _ROW_ALIGNMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    candidate_rows = {str(key): index for index, key in enumerate(candidate_keys)}
    try:
        mapping = np.fromiter(
            (candidate_rows[str(key)] for key in reference_keys),
            dtype=np.int64,
            count=len(reference_keys),
        )
    except KeyError as error:
        raise RuntimeError("stay-key alignment failure") from error
    if len(candidate_rows) != len(reference_keys):
        raise RuntimeError("stay-key alignment failure")
    _ROW_ALIGNMENT_CACHE[cache_key] = mapping
    return mapping


def _assert_identity(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    mapping: np.ndarray,
    *,
    name: str,
) -> None:
    if not np.array_equal(
        np.asarray(reference["mortality"]), np.asarray(candidate["mortality"])[mapping]
    ):
        raise RuntimeError(f"{name} mortality alignment failure")
    if not np.allclose(
        np.asarray(reference["los"], dtype=np.float64),
        np.asarray(candidate["los"], dtype=np.float64)[mapping],
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(f"{name} LOS alignment failure")
    if not np.array_equal(
        np.asarray(reference["group_keys"]).astype(str),
        np.asarray(candidate["group_keys"])[mapping].astype(str),
    ):
        raise RuntimeError(f"{name} group alignment failure")


def _accumulate_predictions(
    payloads: Sequence[Mapping[str, np.ndarray]],
    *,
    reference: Mapping[str, np.ndarray],
    reference_keys: np.ndarray,
    reference_hash: str,
    row_order: np.ndarray,
    scenario_order: np.ndarray,
    coverage: np.ndarray,
    field: str,
    collection_name: str,
) -> np.ndarray:
    accumulator = np.zeros((len(row_order), len(scenario_order)), dtype=np.float64)
    for seed_index, payload in enumerate(payloads):
        # The union is lexically sorted; source holdouts may differ by seed.
        mapping = np.searchsorted(reference_keys, np.asarray(payload["stay_keys"]).astype(str))
        _assert_identity(
            payload, reference, mapping, name=f"{collection_name}[{seed_index}]"
        )
        selected = np.asarray(payload[field])[:, scenario_order]
        accumulator[mapping] += selected.astype(np.float64, copy=False)
    accumulator /= coverage[:, None]
    return accumulator[row_order].reshape(-1).astype(np.float32)


def prepare_seed_ensemble(
    payloads_a: Sequence[Mapping[str, np.ndarray]],
    payloads_b: Sequence[Mapping[str, np.ndarray]],
    *,
    scope: str,
) -> PreparedEnsemble:
    """Align and average exactly five seeds for one registered contrast."""
    if len(payloads_a) != EXPECTED_SEEDS or len(payloads_b) != EXPECTED_SEEDS:
        raise ValueError("correction-07 ensemble requires exactly five seeds per model")
    if scope == "full_6h":
        scenarios = (0,)
    elif scope == "pooled_stress":
        scenarios = tuple(range(1, EXPECTED_SCENARIOS))
    else:
        raise ValueError(f"unsupported analysis scope: {scope}")

    all_payloads = list(payloads_a) + list(payloads_b)
    all_keys = [
        _validate_payload(payload, name=f"payload[{index}]")
        for index, payload in enumerate(all_payloads)
    ]
    # Pair models within each seed before pooling seed-specific holdouts.
    # A union across seeds must never hide missing predictions in one model.
    for seed_index, (a, b) in enumerate(zip(payloads_a, payloads_b, strict=True)):
        mapping = _alignment(all_keys[seed_index], all_keys[EXPECTED_SEEDS + seed_index],
                             reference_hash=_hash_strings(all_keys[seed_index]))
        _assert_identity(a, b, mapping, name=f"paired_models[{seed_index}]")
    reference_keys, first_rows, inverse = np.unique(
        np.concatenate([keys.astype(str) for keys in all_keys[:EXPECTED_SEEDS]]),
        return_index=True, return_inverse=True,
    )
    if not len(reference_keys):
        raise ValueError("correction-07 ensemble has no evaluation stays")
    reference = {
        field: np.concatenate([np.asarray(payload[field]) for payload in payloads_a])[first_rows]
        for field in ("mortality", "los", "group_keys")
    }
    coverage = np.bincount(inverse, minlength=len(reference_keys))
    reference_hash = _hash_strings(reference_keys)
    row_order, scenario_order, sorted_key_hash = _layout(reference_keys, scenarios)

    common = dict(
        reference=reference,
        reference_keys=reference_keys,
        reference_hash=reference_hash,
        row_order=row_order,
        scenario_order=scenario_order,
        coverage=coverage,
    )
    predictions_a = _accumulate_predictions(
        payloads_a, field="mortality_probability", collection_name="model_a", **common
    )
    predictions_b = _accumulate_predictions(
        payloads_b, field="mortality_probability", collection_name="model_b", **common
    )
    los_predictions_a = _accumulate_predictions(
        payloads_a, field="los_prediction", collection_name="model_a", **common
    )
    los_predictions_b = _accumulate_predictions(
        payloads_b, field="los_prediction", collection_name="model_b", **common
    )

    base_mortality = np.asarray(reference["mortality"])[row_order].astype(
        np.int8, copy=False
    )
    base_los = np.asarray(reference["los"])[row_order].astype(np.float32, copy=False)
    base_groups = np.asarray(reference["group_keys"])[row_order].astype(str)
    repeat_count = len(scenario_order)
    mortality = np.repeat(base_mortality, repeat_count)
    los = np.repeat(base_los, repeat_count)
    groups = np.repeat(base_groups, repeat_count)
    unique_groups, cluster_codes = np.unique(groups, return_inverse=True)
    cluster_codes = cluster_codes.astype(np.uint32, copy=False)

    identity_hashes = (
        sorted_key_hash,
        _hash_array(base_mortality),
        _hash_array(base_los),
        _hash_strings(base_groups),
    )
    alignment_sha256 = hashlib.sha256(
        "|".join(identity_hashes).encode("ascii")
    ).hexdigest()
    return PreparedEnsemble(
        mortality=mortality,
        los=los,
        cluster_codes=cluster_codes,
        predictions_a=predictions_a,
        predictions_b=predictions_b,
        los_predictions_a=los_predictions_a,
        los_predictions_b=los_predictions_b,
        minimum_seed_coverage=int(coverage.min()),
        cluster_count=int(len(unique_groups)),
        alignment_sha256=alignment_sha256,
    )
