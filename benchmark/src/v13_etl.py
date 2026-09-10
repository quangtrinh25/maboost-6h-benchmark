from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from v13_feature_registry import (
    DYNAMIC_FEATURES,
    EICU_APERIODIC_COLUMNS,
    EICU_LAB_LABELS,
    EICU_NURSE_LABELS,
    EICU_PERIODIC_COLUMNS,
    EICU_RESP_LABELS,
    ELIXHAUSER_FEATURES,
    ELIXHAUSER_ICD9_PREFIXES,
    ELIXHAUSER_ICD10_PREFIXES,
    MIMIC_CHART_ITEMS,
    MIMIC_LAB_ITEMS,
    MIMIC_URINE_OUTPUT_ITEMS,
    PLAUSIBLE_RANGES,
    STATIC_FEATURES,
)


LANDMARK_HOURS = 6.0
LANDMARK_MINUTES = int(LANDMARK_HOURS * 60)


@dataclass
class CohortResult:
    dataset: str
    frame: pd.DataFrame
    flow: list[dict[str, Any]]
    metadata: dict[str, Any]


def _lower_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={column: column.lower() for column in frame.columns})


def _flow_start(dataset: str, count: int) -> list[dict[str, Any]]:
    return [{
        "dataset": dataset,
        "step_order": 0,
        "step": "raw_stays",
        "input_stays": int(count),
        "excluded_stays": 0,
        "remaining_stays": int(count),
    }]


def _exclude(
    frame: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    reason: str,
    flow: list[dict[str, Any]],
) -> pd.DataFrame:
    normalized = pd.Series(mask, index=frame.index).fillna(True).astype(bool)
    before = len(frame)
    excluded = int(normalized.sum())
    out = frame.loc[~normalized].copy()
    flow.append({
        "dataset": flow[0]["dataset"],
        "step_order": len(flow),
        "step": reason,
        "input_stays": int(before),
        "excluded_stays": excluded,
        "remaining_stays": int(len(out)),
    })
    return out


def _finalize_cohort(
    dataset: str,
    frame: pd.DataFrame,
    flow: list[dict[str, Any]],
    *,
    group_field: str,
    age_correction_count: int = 0,
    uncertain_positive_count: int = 0,
) -> CohortResult:
    frame = frame.sort_values("stay_key", kind="stable").reset_index(drop=True)
    short_count = int((frame["total_los_days"] < LANDMARK_HOURS / 24.0).sum())
    metadata = {
        "eligible_stays": int(len(frame)),
        "unique_groups": int(frame["group_key"].nunique()),
        "group_field": group_field,
        "group_fallback_used": False,
        "mortality_positive_stays": int(frame["mortality_label"].sum()),
        "mortality_prevalence": float(frame["mortality_label"].mean()) if len(frame) else None,
        "mean_age_years": float(frame["age"].mean()) if len(frame) else None,
        "median_age_years": float(frame["age"].median()) if len(frame) else None,
        "mean_los_days": float(frame["los_days"].mean()) if len(frame) else None,
        "median_los_days": float(frame["los_days"].median()) if len(frame) else None,
        "fixed_landmark_hours": LANDMARK_HOURS,
        "stays_not_reaching_landmark_after_final_filter": short_count,
        "remaining_los_is_measured_from_landmark": True,
        "age_deidentification_corrections": int(age_correction_count),
        "uncertain_hospital_deaths_without_icu_death_time_excluded": int(uncertain_positive_count),
    }
    required = {
        "stay_key", "group_key", "hadm_key", "age", "gender", "admission_type",
        "insurance", "careunit", "mortality_label", "los_days",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{dataset} cohort contract is missing columns: {missing}")
    if frame["stay_key"].duplicated().any():
        raise RuntimeError(f"{dataset} cohort has duplicate stay keys")
    if frame["group_key"].isna().any():
        raise RuntimeError(f"{dataset} cohort has missing grouped-split keys")
    return CohortResult(dataset=dataset, frame=frame, flow=flow, metadata=metadata)


def build_mimic_cohort(
    raw_root: Path,
    *,
    dataset: str,
    minimum_age: int,
    exclude_uncertain_positive: bool,
) -> CohortResult:
    if dataset not in {"mimic_iv", "mimic_iii_carevue"}:
        raise ValueError(f"unsupported MIMIC dataset: {dataset}")
    is_v4 = dataset == "mimic_iv"
    names = {
        "icu": "icustays.csv.gz" if is_v4 else "ICUSTAYS.csv.gz",
        "admissions": "admissions.csv.gz" if is_v4 else "ADMISSIONS.csv.gz",
        "patients": "patients.csv.gz" if is_v4 else "PATIENTS.csv.gz",
    }
    icu = _lower_columns(pd.read_csv(raw_root / names["icu"], low_memory=False))
    admissions = _lower_columns(pd.read_csv(raw_root / names["admissions"], low_memory=False))
    patients = _lower_columns(pd.read_csv(raw_root / names["patients"], low_memory=False))
    if "stay_id" not in icu.columns and "icustay_id" in icu.columns:
        icu = icu.rename(columns={"icustay_id": "stay_id"})

    flow = _flow_start(dataset, len(icu))
    key_columns = ["stay_id", "subject_id", "hadm_id", "intime", "outtime"]
    icu = _exclude(icu, icu[key_columns].isna().any(axis=1), "missing_stay_group_or_time", flow)
    icu = _exclude(icu, icu["stay_id"].duplicated(keep="first"), "duplicate_stay_key", flow)
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu = _exclude(
        icu,
        icu["intime"].isna() | icu["outtime"].isna() | (icu["outtime"] <= icu["intime"]),
        "invalid_icu_window",
        flow,
    )

    admission_columns = [
        "hadm_id", "deathtime", "hospital_expire_flag", "admission_type", "insurance",
    ]
    admissions = admissions[admission_columns].drop_duplicates("hadm_id", keep="first")
    admissions["deathtime"] = pd.to_datetime(admissions["deathtime"], errors="coerce")

    age_correction_count = 0
    if is_v4:
        patient_columns = ["subject_id", "gender", "anchor_age", "anchor_year"]
        patients = patients[patient_columns].drop_duplicates("subject_id", keep="first")
        frame = icu.merge(patients, on="subject_id", how="left", validate="many_to_one")
        year_delta = frame["intime"].dt.year - pd.to_numeric(frame["anchor_year"], errors="coerce")
        frame["age"] = pd.to_numeric(frame["anchor_age"], errors="coerce") + year_delta
    else:
        patient_columns = ["subject_id", "gender", "dob", "dod_hosp"]
        patients = patients[patient_columns].drop_duplicates("subject_id", keep="first")
        patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")
        patients["dod_hosp"] = pd.to_datetime(patients["dod_hosp"], errors="coerce")
        frame = icu.merge(patients, on="subject_id", how="left", validate="many_to_one")
        # MIMIC-III shifts some DOB values by roughly 300 years. Subtracting the
        # timestamps directly can overflow pandas' nanosecond timedelta range,
        # so calculate completed calendar years before applying the registered
        # de-identification clamp.
        raw_age = frame["intime"].dt.year - frame["dob"].dt.year
        birthday_not_reached = (
            (frame["intime"].dt.month < frame["dob"].dt.month)
            | (
                (frame["intime"].dt.month == frame["dob"].dt.month)
                & (frame["intime"].dt.day < frame["dob"].dt.day)
            )
        )
        raw_age = (raw_age - birthday_not_reached.astype("Int64")).astype("Float64")
        correction = raw_age > 150.0
        age_correction_count = int(correction.fillna(False).sum())
        frame["age"] = raw_age.mask(correction, 91.4)

    frame = frame.merge(admissions, on="hadm_id", how="left", validate="many_to_one")
    frame = _exclude(frame, frame["hospital_expire_flag"].isna(), "missing_admission_record", flow)
    frame = _exclude(frame, frame["age"].isna(), "missing_age", flow)
    frame = _exclude(frame, frame["age"] < minimum_age, "age_below_minimum", flow)

    death_time = frame["deathtime"].copy()
    if not is_v4:
        death_time = death_time.fillna(frame["dod_hosp"])
    hospital_expire = pd.to_numeric(frame["hospital_expire_flag"], errors="coerce").fillna(0).astype(int)
    uncertain = (hospital_expire == 1) & death_time.isna()
    uncertain_positive_count = int(uncertain.sum())
    if exclude_uncertain_positive:
        frame = _exclude(frame, uncertain, "uncertain_positive_without_death_time", flow)
        death_time = death_time.loc[frame.index]

    frame["landmark_time"] = frame["intime"] + pd.Timedelta(hours=LANDMARK_HOURS)
    frame = _exclude(
        frame,
        frame["outtime"] <= frame["landmark_time"],
        "not_in_icu_strictly_after_6h_landmark",
        flow,
    )
    death_time = death_time.loc[frame.index]
    frame = _exclude(
        frame,
        death_time.notna() & (death_time <= frame["landmark_time"]),
        "death_at_or_before_6h_landmark",
        flow,
    )
    death_time = death_time.loc[frame.index]
    frame["mortality_label"] = (
        death_time.notna()
        & (death_time > frame["landmark_time"])
        & (death_time <= frame["outtime"])
    ).astype(np.int8)
    frame["total_los_days"] = (
        (frame["outtime"] - frame["intime"]).dt.total_seconds() / 86400.0
    )
    frame["los_days"] = (
        (frame["outtime"] - frame["landmark_time"]).dt.total_seconds() / 86400.0
    )
    frame = _exclude(
        frame,
        frame["los_days"].isna() | ~np.isfinite(frame["los_days"]) | (frame["los_days"] <= 0),
        "invalid_los",
        flow,
    )

    frame["stay_key"] = pd.to_numeric(frame["stay_id"], errors="raise").astype(np.int64)
    frame["group_key"] = pd.to_numeric(frame["subject_id"], errors="raise").astype(np.int64)
    frame["hadm_key"] = pd.to_numeric(frame["hadm_id"], errors="raise").astype(np.int64)
    frame["gender"] = frame["gender"].fillna("").astype(str)
    frame["admission_type"] = frame["admission_type"].fillna("").astype(str)
    frame["insurance"] = frame["insurance"].fillna("").astype(str)
    frame["careunit"] = frame.get("first_careunit", pd.Series("", index=frame.index)).fillna("").astype(str)

    return _finalize_cohort(
        dataset,
        frame,
        flow,
        group_field="subject_id",
        age_correction_count=age_correction_count,
        uncertain_positive_count=uncertain_positive_count,
    )


def build_eicu_cohort(
    raw_root: Path,
    *,
    minimum_age: int,
) -> CohortResult:
    usecols = [
        "patientunitstayid", "patienthealthsystemstayid", "uniquepid", "age", "gender",
        "unittype", "unitstaytype", "unitdischargestatus", "unitdischargeoffset",
    ]
    frame = pd.read_csv(raw_root / "patient.csv.gz", usecols=usecols, low_memory=False)
    flow = _flow_start("eicu", len(frame))
    frame = _exclude(
        frame,
        frame[["patientunitstayid", "patienthealthsystemstayid", "uniquepid"]].isna().any(axis=1),
        "missing_stay_or_uniquepid",
        flow,
    )
    frame = _exclude(
        frame,
        frame["patientunitstayid"].duplicated(keep="first"),
        "duplicate_stay_key",
        flow,
    )
    age_text = frame["age"].astype(str).str.strip().replace({"> 89": "90", ">89": "90"})
    frame["age"] = pd.to_numeric(age_text, errors="coerce")
    frame = _exclude(frame, frame["age"].isna(), "missing_age", flow)
    frame = _exclude(frame, frame["age"] < minimum_age, "age_below_minimum", flow)
    frame = _exclude(frame, frame["unitdischargestatus"].isna(), "missing_unit_discharge_status", flow)
    discharge_minutes = pd.to_numeric(frame["unitdischargeoffset"], errors="coerce")
    frame["total_los_days"] = discharge_minutes / 1440.0
    frame = _exclude(
        frame,
        discharge_minutes.isna() | (discharge_minutes <= LANDMARK_MINUTES),
        "not_in_icu_strictly_after_6h_landmark",
        flow,
    )
    discharge_minutes = pd.to_numeric(frame["unitdischargeoffset"], errors="coerce")
    frame["los_days"] = (discharge_minutes - LANDMARK_MINUTES) / 1440.0
    frame = _exclude(
        frame,
        frame["los_days"].isna() | ~np.isfinite(frame["los_days"]) | (frame["los_days"] <= 0),
        "invalid_los",
        flow,
    )
    frame["mortality_label"] = (
        frame["unitdischargestatus"].astype(str).str.strip().str.lower() == "expired"
    ).astype(np.int8)
    frame["stay_key"] = pd.to_numeric(frame["patientunitstayid"], errors="raise").astype(np.int64)
    frame["group_key"] = frame["uniquepid"].astype(str)
    frame["hadm_key"] = pd.to_numeric(
        frame["patienthealthsystemstayid"], errors="raise"
    ).astype(np.int64)
    frame["admission_type"] = frame["unitstaytype"].fillna("").astype(str)
    frame["insurance"] = ""
    frame["careunit"] = frame["unittype"].fillna("").astype(str)
    frame["gender"] = frame["gender"].fillna("").astype(str)
    frame["intime"] = pd.NaT
    frame["outtime"] = pd.NaT
    return _finalize_cohort(
        "eicu",
        frame,
        flow,
        group_field="uniquepid",
    )


def label_blind_group_holdout(
    stay_groups: dict[Any, Any],
    *,
    fraction: float,
    seed: int,
) -> list[Any]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout fraction must be between zero and one")
    groups = sorted(set(stay_groups.values()), key=lambda value: str(value))
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}|{group}".encode("utf-8")).hexdigest(),
    )
    n_holdout = max(1, int(round(len(ranked) * fraction)))
    selected = set(ranked[:n_holdout])
    return sorted(
        [stay for stay, group in stay_groups.items() if group in selected],
        key=lambda stay: hashlib.sha256(f"{seed}|target|{stay}".encode("utf-8")).hexdigest(),
    )


def _csv_chunks(
    path: Path,
    *,
    usecols: list[str],
    max_rows: int,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    kwargs: dict[str, Any] = {
        "usecols": usecols,
        "chunksize": chunksize,
        "low_memory": False,
    }
    if max_rows > 0:
        kwargs["nrows"] = max_rows
    yield from pd.read_csv(path, **kwargs)


def _reverse_item_map(mapping: dict[str, tuple[int, ...]]) -> dict[int, str]:
    return {item: feature for feature, items in mapping.items() for item in items}


def _numeric_value(raw: Any) -> float | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(raw).replace(",", ""))
        if not match:
            return None
        value = float(match.group(0))
    return value if math.isfinite(value) else None


class _EventAccumulator:
    def __init__(
        self,
        stay_ids: Iterable[int],
        *,
        sequence_length: int,
        retain_only_last: bool = False,
    ) -> None:
        self.sequence_length = int(sequence_length)
        self.stay_ids = [int(value) for value in stay_ids]
        self.retain_only_last = bool(retain_only_last)
        self.buckets: dict[int, dict[int, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
        )
        self.predecessor_minutes: dict[int, int] = {}
        self.discarded_early_observations = 0
        self.rejected = Counter()

    def _minute_bucket(
        self,
        stay: int,
        minute: int,
    ) -> dict[str, list[float]] | None:
        stay = int(stay)
        stay_buckets = self.buckets[stay]
        if (
            self.retain_only_last
            and minute not in stay_buckets
            and len(stay_buckets) >= self.sequence_length
        ):
            earliest_retained = min(stay_buckets)
            if minute < earliest_retained:
                self.predecessor_minutes[stay] = max(
                    minute, self.predecessor_minutes.get(stay, minute)
                )
                return None
            self.predecessor_minutes[stay] = max(
                earliest_retained,
                self.predecessor_minutes.get(stay, earliest_retained),
            )
            del stay_buckets[earliest_retained]
        return stay_buckets[minute]

    def add(self, stay: int, offset_minutes: float, feature: str, raw_value: Any) -> bool:
        if feature not in DYNAMIC_FEATURES:
            return False
        offset = _numeric_value(offset_minutes)
        value = _numeric_value(raw_value)
        if offset is None or value is None or offset < 0.0 or offset > LANDMARK_MINUTES:
            self.rejected[f"{feature}:invalid_time_or_value"] += 1
            return False
        if feature == "fio2" and 0.0 < value <= 1.0:
            value *= 100.0
        lower, upper = PLAUSIBLE_RANGES[feature]
        if value < lower or value > upper:
            self.rejected[f"{feature}:out_of_range"] += 1
            return False
        minute = int(math.floor(offset))
        minute_bucket = self._minute_bucket(int(stay), minute)
        if minute_bucket is None:
            self.discarded_early_observations += 1
            return True
        total_count = minute_bucket[feature]
        total_count[0] += float(value)
        total_count[1] += 1.0
        return True

    def add_valid_aggregate(
        self,
        stay: int,
        minute: int,
        feature: str,
        total: float,
        count: int,
    ) -> None:
        """Add a prevalidated minute-feature aggregate from a vectorized scan."""
        minute_bucket = self._minute_bucket(int(stay), int(minute))
        if minute_bucket is None:
            self.discarded_early_observations += int(count)
            return
        total_count = minute_bucket[feature]
        total_count[0] += float(total)
        total_count[1] += float(count)

    @staticmethod
    def _derive_gcs(stay_buckets: dict[int, dict[str, list[float]]]) -> None:
        for values in stay_buckets.values():
            if "gcs_total" not in values and all(
                component in values for component in ("gcs_eye", "gcs_motor", "gcs_verbal")
            ):
                gcs = sum(values[component][0] / values[component][1] for component in (
                    "gcs_eye", "gcs_motor", "gcs_verbal"
                ))
                values["gcs_total"] = [gcs, 1.0]

    def materialize_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Materialize the fixed sequence contract without exposing identifiers."""
        n_stays = len(self.stay_ids)
        n_features = len(DYNAMIC_FEATURES)
        feature_index = {feature: idx for idx, feature in enumerate(DYNAMIC_FEATURES)}
        values = np.full(
            (n_stays, self.sequence_length, n_features), np.nan, dtype=np.float32
        )
        mask = np.zeros((n_stays, self.sequence_length, n_features), dtype=np.uint8)
        tau = np.full((n_stays, self.sequence_length), 60.0, dtype=np.float32)
        elapsed = np.zeros((n_stays, self.sequence_length), dtype=np.float32)
        feature_stays = Counter()
        stays_with_observations = 0
        final_observations = 0

        for output_row, stay in enumerate(self.stay_ids):
            stay_buckets = self.buckets.get(stay, {})
            self._derive_gcs(stay_buckets)
            minutes = sorted(stay_buckets)[-self.sequence_length:]
            if not minutes:
                continue
            start = self.sequence_length - len(minutes)
            elapsed_real = np.asarray(minutes, dtype=np.float32) * 60.0
            predecessor = self.predecessor_minutes.get(stay)
            tau_real = np.empty(len(minutes), dtype=np.float32)
            if predecessor is None:
                tau_real[0] = float(np.clip(elapsed_real[0], 1.0, 86400.0))
            else:
                tau_real[0] = float(max((minutes[0] - predecessor) * 60.0, 1.0))
            if len(minutes) > 1:
                tau_real[1:] = np.maximum(np.diff(elapsed_real), 1.0)
            tau[output_row, start:] = tau_real
            elapsed[output_row, start:] = elapsed_real
            for time_index, minute in enumerate(minutes, start=start):
                for feature, (total, count) in stay_buckets[minute].items():
                    column = feature_index[feature]
                    values[output_row, time_index, column] = float(total / count)
                    mask[output_row, time_index, column] = 1
            observed = mask[output_row].sum(axis=0)
            if int(observed.sum()) > 0:
                stays_with_observations += 1
            final_observations += int(observed.sum())
            for column, feature in enumerate(DYNAMIC_FEATURES):
                if observed[column] > 0:
                    feature_stays[feature] += 1

        arrays = {"values": values, "tau": tau, "mask": mask, "elapsed": elapsed}
        audit = {
            "sequence_shape": [self.sequence_length, n_features],
            "stays": n_stays,
            "stays_with_observations": int(stays_with_observations),
            "zero_event_stays": int(n_stays - stays_with_observations),
            "final_observed_cells": int(final_observations),
            "feature_stay_coverage": {
                feature: int(feature_stays[feature]) for feature in DYNAMIC_FEATURES
            },
            "rejected_values": dict(sorted(self.rejected.items())),
            "retain_only_last": self.retain_only_last,
            "discarded_valid_early_observations": int(self.discarded_early_observations),
            "contract_pass": (
                values.shape == (n_stays, self.sequence_length, n_features)
                and mask.shape == values.shape
                and np.isfinite(tau).all()
                and np.isfinite(elapsed).all()
                and np.array_equal(np.isnan(values), mask == 0)
            ),
        }
        return arrays, audit

    def summary(self) -> dict[str, Any]:
        feature_index = {feature: idx for idx, feature in enumerate(DYNAMIC_FEATURES)}
        feature_stays = Counter()
        sequence_stays = 0
        total_observations = 0
        maximum_unpadded_steps = 0
        for stay in self.stay_ids:
            stay_buckets = self.buckets.get(stay, {})
            self._derive_gcs(stay_buckets)
            minutes = sorted(stay_buckets)
            maximum_unpadded_steps = max(maximum_unpadded_steps, len(minutes))
            minutes = minutes[-self.sequence_length:]
            values = np.full(
                (self.sequence_length, len(DYNAMIC_FEATURES)), np.nan, dtype=np.float32
            )
            mask = np.zeros_like(values, dtype=np.float32)
            start = self.sequence_length - len(minutes)
            for row_index, minute in enumerate(minutes, start=start):
                for feature, (total, count) in stay_buckets[minute].items():
                    column = feature_index[feature]
                    values[row_index, column] = float(total / count)
                    mask[row_index, column] = 1.0
            observed = mask.sum(axis=0)
            if int(observed.sum()) > 0:
                sequence_stays += 1
            total_observations += int(observed.sum())
            for index, feature in enumerate(DYNAMIC_FEATURES):
                if observed[index] > 0:
                    feature_stays[feature] += 1
            if values.shape != (self.sequence_length, len(DYNAMIC_FEATURES)):
                raise AssertionError(
                    "bounded sequence shape violates the configured temporal contract"
                )
        return {
            "selected_stays": len(self.stay_ids),
            "sequence_stays_with_observations": int(sequence_stays),
            "zero_event_stays": int(len(self.stay_ids) - sequence_stays),
            "sequence_shape": [self.sequence_length, len(DYNAMIC_FEATURES)],
            "maximum_unpadded_steps": int(maximum_unpadded_steps),
            "total_accepted_observations": int(total_observations),
            "feature_stay_coverage": {
                feature: int(feature_stays[feature]) for feature in DYNAMIC_FEATURES
            },
            "rejected_values": dict(sorted(self.rejected.items())),
            "retain_only_last": self.retain_only_last,
            "discarded_valid_early_observations": int(self.discarded_early_observations),
            "contract_pass": (
                len(DYNAMIC_FEATURES) == 40 and self.sequence_length > 0
            ),
        }


def _mimic_offset_minutes(
    timestamp: Any,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> float | None:
    parsed = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(parsed):
        return None
    intime, end = window
    if parsed < intime or parsed > end:
        return None
    return float((parsed - intime).total_seconds() / 60.0)


def _accumulate_grouped_events(
    accumulator: _EventAccumulator,
    frame: pd.DataFrame,
    *,
    stay_column: str,
    offset_column: str,
    feature_column: str,
    value_column: str,
) -> int:
    """Validate, aggregate, and add raw observations without scalar timestamp parsing."""
    if frame.empty:
        return 0
    work = frame[[stay_column, offset_column, feature_column, value_column]].copy()
    work[offset_column] = pd.to_numeric(work[offset_column], errors="coerce")
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    invalid = (
        work[offset_column].isna()
        | work[value_column].isna()
        | ~np.isfinite(work[offset_column])
        | ~np.isfinite(work[value_column])
        | (work[offset_column] < 0.0)
        | (work[offset_column] > LANDMARK_MINUTES)
    )
    for feature, count in work.loc[invalid, feature_column].value_counts().items():
        accumulator.rejected[f"{feature}:invalid_time_or_value"] += int(count)
    work = work.loc[~invalid].copy()
    if work.empty:
        return 0
    fio2 = (work[feature_column] == "fio2") & (work[value_column] > 0.0) & (
        work[value_column] <= 1.0
    )
    work.loc[fio2, value_column] *= 100.0
    lower = work[feature_column].map({name: bounds[0] for name, bounds in PLAUSIBLE_RANGES.items()})
    upper = work[feature_column].map({name: bounds[1] for name, bounds in PLAUSIBLE_RANGES.items()})
    out_of_range = (work[value_column] < lower) | (work[value_column] > upper)
    for feature, count in work.loc[out_of_range, feature_column].value_counts().items():
        accumulator.rejected[f"{feature}:out_of_range"] += int(count)
    work = work.loc[~out_of_range].copy()
    accepted = len(work)
    if not accepted:
        return 0
    work["_minute"] = np.floor(work[offset_column]).astype(np.int32)
    grouped = (
        work.groupby([stay_column, "_minute", feature_column], sort=False, observed=True)[
            value_column
        ]
        .agg(["sum", "count"])
        .reset_index()
    )
    for stay, minute, feature, total, count in grouped.itertuples(index=False, name=None):
        accumulator.add_valid_aggregate(
            int(stay), int(minute), str(feature), float(total), int(count)
        )
    return int(accepted)


def _select_mimic_event_sample(
    cohort: pd.DataFrame,
    raw_root: Path,
    *,
    dataset: str,
    max_stays: int,
    max_rows: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Choose a label-blind bounded sample represented in each event source."""
    is_v4 = dataset == "mimic_iv"
    eligible_stays = set(int(value) for value in cohort["stay_key"])
    eligible_hadm = set(int(value) for value in cohort["hadm_key"])
    hadm_to_stays: dict[int, list[int]] = defaultdict(list)
    for row in cohort.itertuples(index=False):
        hadm_to_stays[int(row.hadm_key)].append(int(row.stay_key))
    chart_items = set(_reverse_item_map(MIMIC_CHART_ITEMS))
    lab_items = set(_reverse_item_map(MIMIC_LAB_ITEMS))

    chart_path = raw_root / ("chartevents.csv.gz" if is_v4 else "CHARTEVENTS.csv.gz")
    chart_stay = "stay_id" if is_v4 else "icustay_id"
    chart_item = "itemid"
    chart_candidates: list[int] = []
    for chunk in _csv_chunks(
        chart_path, usecols=[chart_stay, chart_item], max_rows=max_rows
    ):
        rows = chunk[chunk[chart_stay].isin(eligible_stays) & chunk[chart_item].isin(chart_items)]
        chart_candidates.extend(int(value) for value in rows[chart_stay].dropna().unique())

    lab_path = raw_root / ("labevents.csv.gz" if is_v4 else "LABEVENTS.csv.gz")
    lab_hadm = "hadm_id"
    lab_item = "itemid"
    lab_candidates: list[int] = []
    for chunk in _csv_chunks(lab_path, usecols=[lab_hadm, lab_item], max_rows=max_rows):
        rows = chunk[chunk[lab_hadm].isin(eligible_hadm) & chunk[lab_item].isin(lab_items)]
        for hadm in rows[lab_hadm].dropna().unique():
            lab_candidates.extend(hadm_to_stays[int(hadm)])

    output_path = raw_root / ("outputevents.csv.gz" if is_v4 else "OUTPUTEVENTS.csv.gz")
    output_stay = "stay_id" if is_v4 else "icustay_id"
    output_item = "itemid"
    output_candidates: list[int] = []
    for chunk in _csv_chunks(
        output_path, usecols=[output_stay, output_item], max_rows=max_rows
    ):
        rows = chunk[
            chunk[output_stay].isin(eligible_stays)
            & chunk[output_item].isin(MIMIC_URINE_OUTPUT_ITEMS)
        ]
        output_candidates.extend(int(value) for value in rows[output_stay].dropna().unique())

    sources = {
        "chartevents": list(dict.fromkeys(chart_candidates)),
        "labevents": list(dict.fromkeys(lab_candidates)),
        "outputevents": list(dict.fromkeys(output_candidates)),
    }
    selected: list[int] = []
    quota = max(1, max_stays // len(sources))
    for values in sources.values():
        for stay in values[:quota]:
            if stay not in selected:
                selected.append(stay)
    for values in sources.values():
        for stay in values:
            if len(selected) >= max_stays:
                break
            if stay not in selected:
                selected.append(stay)
    if not selected:
        selected = [int(value) for value in cohort.head(max_stays)["stay_key"]]
    selected_set = set(selected[:max_stays])
    sample = cohort[cohort["stay_key"].isin(selected_set)].copy()
    order = {stay: index for index, stay in enumerate(selected[:max_stays])}
    sample["_smoke_order"] = sample["stay_key"].map(order)
    sample = sample.sort_values("_smoke_order", kind="stable").drop(columns="_smoke_order")
    return sample, {
        "chartevents_candidate_stays": len(set(chart_candidates)),
        "labevents_candidate_stays": len(set(lab_candidates)),
        "outputevents_candidate_stays": len(set(output_candidates)),
        "selection_used_labels": 0,
    }


def _scan_mimic_events(
    cohort: pd.DataFrame,
    raw_root: Path,
    *,
    dataset: str,
    max_rows: int,
    sequence_length: int,
    accumulator: _EventAccumulator | None = None,
) -> tuple[dict[str, Any], list[int]]:
    selected = cohort.sort_values("stay_key", kind="stable").copy()
    selected_ids = [int(value) for value in selected["stay_key"]]
    if accumulator is None:
        accumulator = _EventAccumulator(selected_ids, sequence_length=sequence_length)
    elif accumulator.stay_ids != selected_ids:
        raise ValueError("event accumulator stay order differs from the selected cohort")
    windows = {
        int(row.stay_key): (
            pd.Timestamp(row.intime),
            pd.Timestamp(row.intime) + pd.Timedelta(hours=LANDMARK_HOURS),
        )
        for row in selected.itertuples(index=False)
    }
    hadm_windows: dict[int, list[int]] = defaultdict(list)
    for row in selected.itertuples(index=False):
        hadm_windows[int(row.hadm_key)].append(int(row.stay_key))
    window_frame = pd.DataFrame({
        "_stay": list(windows),
        "_intime": [windows[stay][0] for stay in windows],
        "_window_end": [windows[stay][1] for stay in windows],
    })
    stay_intime = window_frame.set_index("_stay")["_intime"]
    stay_window_end = window_frame.set_index("_stay")["_window_end"]
    lab_window_frame = selected[["hadm_key", "stay_key"]].copy()
    lab_window_frame["_intime"] = lab_window_frame["stay_key"].map(stay_intime)
    lab_window_frame["_window_end"] = lab_window_frame["stay_key"].map(stay_window_end)
    lab_window_frame = lab_window_frame.rename(
        columns={"hadm_key": "_hadm", "stay_key": "_stay"}
    )
    stay_set = set(selected_ids)
    stats: dict[str, Any] = {}
    is_v4 = dataset == "mimic_iv"
    chart_path = raw_root / ("chartevents.csv.gz" if is_v4 else "CHARTEVENTS.csv.gz")
    lab_path = raw_root / ("labevents.csv.gz" if is_v4 else "LABEVENTS.csv.gz")
    output_path = raw_root / ("outputevents.csv.gz" if is_v4 else "OUTPUTEVENTS.csv.gz")
    # The official CareVue v1.4 archive keeps the historical upper-case file
    # names but stores lower-case CSV headers.  Treat header casing as part of
    # the dataset schema instead of inferring it from the file name.
    chart_stay = "stay_id" if is_v4 else "icustay_id"
    chart_time = "charttime"
    chart_item = "itemid"
    chart_value = "valuenum"
    chart_flag = "warning" if is_v4 else "error"
    chart_map = _reverse_item_map(MIMIC_CHART_ITEMS)
    lab_map = _reverse_item_map(MIMIC_LAB_ITEMS)

    scanned = candidates = accepted = 0
    for chunk in _csv_chunks(
        chart_path,
        usecols=[chart_stay, chart_time, chart_item, chart_value, chart_flag],
        max_rows=max_rows,
    ):
        scanned += len(chunk)
        chunk = chunk[
            chunk[chart_stay].isin(stay_set)
            & chunk[chart_item].isin(chart_map)
            & (pd.to_numeric(chunk[chart_flag], errors="coerce").fillna(0) == 0)
        ].copy()
        candidates += len(chunk)
        if not chunk.empty:
            parsed = pd.to_datetime(
                chunk[chart_time], format="%Y-%m-%d %H:%M:%S", errors="coerce"
            )
            mapped_intime = chunk[chart_stay].map(stay_intime)
            chunk["_offset"] = (parsed - mapped_intime).dt.total_seconds() / 60.0
            within_end = parsed <= chunk[chart_stay].map(stay_window_end)
            chunk.loc[~within_end, "_offset"] = np.nan
            chunk["_feature"] = chunk[chart_item].map(chart_map)
            accepted += _accumulate_grouped_events(
                accumulator,
                chunk,
                stay_column=chart_stay,
                offset_column="_offset",
                feature_column="_feature",
                value_column=chart_value,
            )
    stats["chartevents"] = {
        "input_rows_scanned": scanned, "candidate_rows": candidates, "accepted_observations": accepted,
    }

    lab_hadm = "hadm_id"
    lab_time = "charttime"
    lab_item = "itemid"
    lab_value = "valuenum"
    scanned = candidates = accepted = 0
    for chunk in _csv_chunks(
        lab_path,
        usecols=[lab_hadm, lab_time, lab_item, lab_value],
        max_rows=max_rows,
    ):
        scanned += len(chunk)
        chunk = chunk[
            chunk[lab_hadm].isin(hadm_windows) & chunk[lab_item].isin(lab_map)
        ].copy()
        candidates += len(chunk)
        if not chunk.empty:
            chunk = chunk.rename(columns={lab_hadm: "_hadm"}).merge(
                lab_window_frame,
                on="_hadm",
                how="inner",
                validate="many_to_many",
            )
            parsed = pd.to_datetime(
                chunk[lab_time], format="%Y-%m-%d %H:%M:%S", errors="coerce"
            )
            chunk["_offset"] = (parsed - chunk["_intime"]).dt.total_seconds() / 60.0
            chunk.loc[parsed > chunk["_window_end"], "_offset"] = np.nan
            chunk["_feature"] = chunk[lab_item].map(lab_map)
            accepted += _accumulate_grouped_events(
                accumulator,
                chunk,
                stay_column="_stay",
                offset_column="_offset",
                feature_column="_feature",
                value_column=lab_value,
            )
    stats["labevents"] = {
        "input_rows_scanned": scanned, "candidate_rows": candidates, "accepted_observations": accepted,
    }

    output_stay = "stay_id" if is_v4 else "icustay_id"
    output_time = "charttime"
    output_item = "itemid"
    output_value = "value"
    scanned = candidates = accepted = 0
    for chunk in _csv_chunks(
        output_path,
        usecols=[output_stay, output_time, output_item, output_value],
        max_rows=max_rows,
    ):
        scanned += len(chunk)
        chunk = chunk[
            chunk[output_stay].isin(stay_set)
            & chunk[output_item].isin(MIMIC_URINE_OUTPUT_ITEMS)
        ].copy()
        candidates += len(chunk)
        if not chunk.empty:
            parsed = pd.to_datetime(
                chunk[output_time], format="%Y-%m-%d %H:%M:%S", errors="coerce"
            )
            mapped_intime = chunk[output_stay].map(stay_intime)
            chunk["_offset"] = (parsed - mapped_intime).dt.total_seconds() / 60.0
            within_end = parsed <= chunk[output_stay].map(stay_window_end)
            chunk.loc[~within_end, "_offset"] = np.nan
            chunk["_feature"] = "urine_output"
            accepted += _accumulate_grouped_events(
                accumulator,
                chunk,
                stay_column=output_stay,
                offset_column="_offset",
                feature_column="_feature",
                value_column=output_value,
            )
    stats["outputevents"] = {
        "input_rows_scanned": scanned, "candidate_rows": candidates, "accepted_observations": accepted,
    }
    summary = accumulator.summary()
    summary["tables"] = stats
    return summary, selected_ids


def _urine_output_label(label: Any) -> bool:
    text = str(label).strip().lower()
    include = (
        "urine", "foley", "urethral", "urinary catheter", "straight catheter",
        "nephrostomy", "urostomy", "suprapubic", "external urethral device",
    )
    exclude = ("urine count", "incontinent voids")
    return any(token in text for token in include) and not any(token in text for token in exclude)


def _scan_eicu_events(
    cohort: pd.DataFrame,
    raw_root: Path,
    *,
    max_rows: int,
    sequence_length: int,
) -> tuple[dict[str, Any], list[int]]:
    selected = cohort.sort_values("stay_key", kind="stable").copy()
    selected_ids = [int(value) for value in selected["stay_key"]]
    stay_set = set(selected_ids)
    accumulator = _EventAccumulator(selected_ids, sequence_length=sequence_length)
    stats: dict[str, Any] = {}

    for table_name, filename, offset_column, mapping in (
        ("vitalPeriodic", "vitalPeriodic.csv.gz", "observationoffset", EICU_PERIODIC_COLUMNS),
        ("vitalAperiodic", "vitalAperiodic.csv.gz", "observationoffset", EICU_APERIODIC_COLUMNS),
    ):
        usecols = ["patientunitstayid", offset_column, *mapping]
        scanned = candidates = accepted = 0
        for chunk in _csv_chunks(raw_root / filename, usecols=usecols, max_rows=max_rows):
            scanned += len(chunk)
            chunk = chunk[chunk["patientunitstayid"].isin(stay_set)]
            candidates += len(chunk)
            for row in chunk.itertuples(index=False):
                stay = int(row.patientunitstayid)
                offset = getattr(row, offset_column)
                for source, feature in mapping.items():
                    if accumulator.add(stay, offset, feature, getattr(row, source)):
                        accepted += 1
        stats[table_name] = {
            "input_rows_scanned": scanned, "candidate_rows": candidates, "accepted_observations": accepted,
        }

    scanned = candidates = accepted = 0
    nurse_columns = [
        "patientunitstayid", "nursingchartoffset", "nursingchartcelltypevalname", "nursingchartvalue",
    ]
    for chunk in _csv_chunks(raw_root / "nurseCharting.csv.gz", usecols=nurse_columns, max_rows=max_rows):
        scanned += len(chunk)
        chunk = chunk[chunk["patientunitstayid"].isin(stay_set)]
        labels = chunk["nursingchartcelltypevalname"].astype(str).str.strip().str.lower()
        chunk = chunk.assign(_feature=labels.map(EICU_NURSE_LABELS)).dropna(subset=["_feature"])
        candidates += len(chunk)
        for stay, offset, _, value, feature in chunk.itertuples(index=False, name=None):
            if accumulator.add(int(stay), offset, feature, value):
                accepted += 1
    stats["nurseCharting"] = {
        "input_rows_scanned": scanned, "candidate_rows": candidates, "accepted_observations": accepted,
    }

    scanned = candidates = accepted = 0
    lab_columns = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    for chunk in _csv_chunks(raw_root / "lab.csv.gz", usecols=lab_columns, max_rows=max_rows):
        scanned += len(chunk)
        chunk = chunk[chunk["patientunitstayid"].isin(stay_set)]
        labels = chunk["labname"].astype(str).str.strip().str.lower()
        chunk = chunk.assign(_feature=labels.map(EICU_LAB_LABELS)).dropna(subset=["_feature"])
        candidates += len(chunk)
        for stay, offset, _, value, feature in chunk.itertuples(index=False, name=None):
            if accumulator.add(int(stay), offset, feature, value):
                accepted += 1
    stats["lab"] = {
        "input_rows_scanned": scanned, "candidate_rows": candidates, "accepted_observations": accepted,
    }

    scanned = candidates = accepted = 0
    respiratory_columns = [
        "patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue",
    ]
    for chunk in _csv_chunks(
        raw_root / "respiratoryCharting.csv.gz", usecols=respiratory_columns, max_rows=max_rows
    ):
        scanned += len(chunk)
        chunk = chunk[chunk["patientunitstayid"].isin(stay_set)]
        labels = chunk["respchartvaluelabel"].astype(str).str.strip().str.lower()
        chunk = chunk.assign(_feature=labels.map(EICU_RESP_LABELS)).dropna(subset=["_feature"])
        candidates += len(chunk)
        for stay, offset, _, value, feature in chunk.itertuples(index=False, name=None):
            if accumulator.add(int(stay), offset, feature, value):
                accepted += 1
    stats["respiratoryCharting"] = {
        "input_rows_scanned": scanned, "candidate_rows": candidates, "accepted_observations": accepted,
    }

    scanned = candidates = accepted = 0
    intake_columns = [
        "patientunitstayid", "intakeoutputoffset", "celllabel", "cellvaluenumeric",
    ]
    for chunk in _csv_chunks(raw_root / "intakeOutput.csv.gz", usecols=intake_columns, max_rows=max_rows):
        scanned += len(chunk)
        chunk = chunk[chunk["patientunitstayid"].isin(stay_set)]
        chunk = chunk[chunk["celllabel"].map(_urine_output_label)]
        candidates += len(chunk)
        for stay, offset, _, value in chunk.itertuples(index=False, name=None):
            if accumulator.add(int(stay), offset, "urine_output", value):
                accepted += 1
    stats["intakeOutput"] = {
        "input_rows_scanned": scanned, "candidate_rows": candidates, "accepted_observations": accepted,
    }
    summary = accumulator.summary()
    summary["tables"] = stats
    return summary, selected_ids


def _normalize_icd_codes(raw: Any) -> set[str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return set()
    values = set()
    for part in re.split(r"[,;|/ ]+", str(raw).upper()):
        code = re.sub(r"[^A-Z0-9]", "", part)
        if code and code[0].isalnum():
            values.add(code)
    return values


def elixhauser_flags(
    icd9_codes: set[str],
    icd10_codes: set[str],
) -> dict[str, int]:
    """Return independent Quan Elixhauser flags from normalized code prefixes."""
    normalized9 = {
        code for raw in icd9_codes for code in _normalize_icd_codes(raw)
    }
    normalized10 = {
        code for raw in icd10_codes for code in _normalize_icd_codes(raw)
    }
    flags: dict[str, int] = {}
    for condition in ELIXHAUSER_FEATURES:
        match9 = any(
            code.startswith(prefix)
            for code in normalized9
            for prefix in ELIXHAUSER_ICD9_PREFIXES[condition]
        )
        match10 = any(
            code.startswith(prefix)
            for code in normalized10
            for prefix in ELIXHAUSER_ICD10_PREFIXES[condition]
        )
        flags[condition] = int(match9 or match10)
    return flags


def _unit_flags(unit: str) -> set[str]:
    text = str(unit).strip().lower()
    if any(token in text for token in ("cticu", "csicu", "cardiothoracic", "cardiac surgery")):
        return {"csru"}
    if "nicu" in text or "neonatal" in text:
        return {"nicu"}
    if any(token in text for token in ("cicu", "ccu", "coronary", "cardiac")):
        return {"cicu"}
    if "sicu" in text or "surgical" in text:
        return {"sicu"}
    if any(token in text for token in ("micu", "medical", "med-surg", "med/surg")):
        return {"micu"}
    return {"micu"} if "icu" in text else set()


def build_static_feature_matrix(
    cohort: pd.DataFrame,
    raw_root: Path,
    *,
    dataset: str,
    max_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the time-safe admission/context feature contract.

    The former v13 implementation scanned all diagnoses attached to the same
    hospitalization to construct Elixhauser flags.  MIMIC does not provide a
    reliable timestamp proving that those codes were known at the six-hour
    landmark, so the corrected primary contract intentionally never opens a
    diagnosis table.
    """
    expected = (
        "age", "gender_male", "emergency_admit", "elective_admit",
        "insurance_medicare", "insurance_medicaid", "insurance_private",
        "micu", "sicu", "cicu", "nicu", "csru",
    )
    if tuple(STATIC_FEATURES) != expected:
        raise RuntimeError("fixed-landmark static feature registry drift")
    index = {name: idx for idx, name in enumerate(STATIC_FEATURES)}
    vectors: list[np.ndarray] = []
    for row in cohort.itertuples(index=False):
        vector = np.zeros(len(STATIC_FEATURES), dtype=np.float32)
        vector[index["age"]] = float(row.age)
        vector[index["gender_male"]] = float(
            str(row.gender).strip().upper().startswith("M")
        )
        admission = str(row.admission_type).upper()
        insurance = str(row.insurance).upper()
        vector[index["emergency_admit"]] = float("EMERG" in admission)
        vector[index["elective_admit"]] = float("ELECTIVE" in admission)
        vector[index["insurance_medicare"]] = float("MEDICARE" in insurance)
        vector[index["insurance_medicaid"]] = float("MEDICAID" in insurance)
        vector[index["insurance_private"]] = float("PRIVATE" in insurance)
        for unit in _unit_flags(row.careunit):
            vector[index[unit]] = 1.0
        vectors.append(vector)
    matrix = (
        np.vstack(vectors)
        if vectors
        else np.empty((0, len(STATIC_FEATURES)), dtype=np.float32)
    )
    audit = {
        "static_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "all_finite": bool(np.isfinite(matrix).all()),
        "availability_class": "admission_time_or_context",
        "diagnosis_tables_opened": False,
        "elixhauser_features_excluded": 30,
        "excluded_reason": (
            "same-hospitalization diagnoses are not reliably timestamped before "
            "the fixed six-hour landmark across all datasets"
        ),
        "contract_pass": (
            matrix.shape[1] == 12 and bool(np.isfinite(matrix).all())
        ),
    }
    return matrix, audit


def _static_audit(
    cohort: pd.DataFrame,
    raw_root: Path,
    *,
    dataset: str,
    max_rows: int,
) -> dict[str, Any]:
    _, audit = build_static_feature_matrix(
        cohort,
        raw_root,
        dataset=dataset,
        max_rows=max_rows,
    )
    return audit


def bounded_etl_audit(
    cohort: CohortResult,
    raw_root: Path,
    *,
    max_stays: int,
    max_rows_per_table: int,
    sequence_length: int,
) -> dict[str, Any]:
    if cohort.dataset == "eicu":
        selected = cohort.frame.head(max_stays).copy()
        sampling = {
            "strategy": "first eligible stays; eICU tables are stay-key ordered in bounded scan",
            "selection_used_labels": 0,
        }
        dynamic, selected_ids = _scan_eicu_events(
            selected,
            raw_root,
            max_rows=max_rows_per_table,
            sequence_length=sequence_length,
        )
    else:
        selected, candidate_counts = _select_mimic_event_sample(
            cohort.frame,
            raw_root,
            dataset=cohort.dataset,
            max_stays=max_stays,
            max_rows=max_rows_per_table,
        )
        sampling = {
            "strategy": "balanced event-presence sample from bounded chart/lab/output scans",
            **candidate_counts,
        }
        dynamic, selected_ids = _scan_mimic_events(
            selected,
            raw_root,
            dataset=cohort.dataset,
            max_rows=max_rows_per_table,
            sequence_length=sequence_length,
        )
    static = _static_audit(
        selected[selected["stay_key"].isin(selected_ids)],
        raw_root,
        dataset=cohort.dataset,
        max_rows=max_rows_per_table,
    )
    return {
        "dataset": cohort.dataset,
        "mode": "bounded_in_memory_smoke",
        "max_stays": int(max_stays),
        "max_input_rows_per_table": int(max_rows_per_table),
        "dynamic": dynamic,
        "static": static,
        "sampling": sampling,
        "patient_level_artifact_written": False,
        "normalization_fit_performed": False,
        "target_values_accessed_for_dynamic_or_static_mapping": False,
        "contract_pass": bool(dynamic["contract_pass"] and static["contract_pass"]),
    }
