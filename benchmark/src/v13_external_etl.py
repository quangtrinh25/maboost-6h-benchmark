from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from v13_etl import (
    _EventAccumulator,
    _accumulate_grouped_events,
    _csv_chunks,
    _urine_output_label,
)
from v13_feature_registry import (
    EICU_APERIODIC_COLUMNS,
    EICU_LAB_LABELS,
    EICU_NURSE_LABELS,
    EICU_PERIODIC_COLUMNS,
    EICU_RESP_LABELS,
)


def _accumulate_wide_table(
    accumulator: _EventAccumulator,
    raw_root: Path,
    *,
    filename: str,
    offset_column: str,
    mapping: dict[str, str],
    stay_set: set[int],
    max_rows: int,
    chunksize: int,
) -> dict[str, int]:
    scanned = selected_rows = candidate_observations = accepted = 0
    columns = ["patientunitstayid", offset_column, *mapping]
    for chunk in _csv_chunks(
        raw_root / filename,
        usecols=columns,
        max_rows=max_rows,
        chunksize=chunksize,
    ):
        scanned += len(chunk)
        chunk = chunk[chunk["patientunitstayid"].isin(stay_set)]
        selected_rows += len(chunk)
        if chunk.empty:
            continue
        for source_column, feature in mapping.items():
            observed = chunk[source_column].notna()
            candidate_observations += int(observed.sum())
            if not observed.any():
                continue
            work = chunk.loc[
                observed, ["patientunitstayid", offset_column, source_column]
            ].copy()
            work["_feature"] = feature
            accepted += _accumulate_grouped_events(
                accumulator,
                work,
                stay_column="patientunitstayid",
                offset_column=offset_column,
                feature_column="_feature",
                value_column=source_column,
            )
    return {
        "input_rows_scanned": int(scanned),
        "selected_cohort_rows": int(selected_rows),
        "candidate_observations": int(candidate_observations),
        "accepted_observations": int(accepted),
    }


def _accumulate_label_table(
    accumulator: _EventAccumulator,
    raw_root: Path,
    *,
    filename: str,
    offset_column: str,
    label_column: str,
    value_column: str,
    mapping: dict[str, str],
    stay_set: set[int],
    max_rows: int,
    chunksize: int,
) -> dict[str, int]:
    scanned = selected_rows = candidates = accepted = 0
    columns = ["patientunitstayid", offset_column, label_column, value_column]
    for chunk in _csv_chunks(
        raw_root / filename,
        usecols=columns,
        max_rows=max_rows,
        chunksize=chunksize,
    ):
        scanned += len(chunk)
        chunk = chunk[chunk["patientunitstayid"].isin(stay_set)].copy()
        selected_rows += len(chunk)
        if chunk.empty:
            continue
        labels = chunk[label_column].astype(str).str.strip().str.lower()
        chunk["_feature"] = labels.map(mapping)
        chunk = chunk.dropna(subset=["_feature"])
        candidates += len(chunk)
        accepted += _accumulate_grouped_events(
            accumulator,
            chunk,
            stay_column="patientunitstayid",
            offset_column=offset_column,
            feature_column="_feature",
            value_column=value_column,
        )
    return {
        "input_rows_scanned": int(scanned),
        "selected_cohort_rows": int(selected_rows),
        "candidate_observations": int(candidates),
        "accepted_observations": int(accepted),
    }


def _accumulate_urine_output(
    accumulator: _EventAccumulator,
    raw_root: Path,
    *,
    stay_set: set[int],
    max_rows: int,
    chunksize: int,
) -> dict[str, int]:
    scanned = selected_rows = candidates = accepted = 0
    columns = [
        "patientunitstayid",
        "intakeoutputoffset",
        "celllabel",
        "cellvaluenumeric",
    ]
    for chunk in _csv_chunks(
        raw_root / "intakeOutput.csv.gz",
        usecols=columns,
        max_rows=max_rows,
        chunksize=chunksize,
    ):
        scanned += len(chunk)
        chunk = chunk[chunk["patientunitstayid"].isin(stay_set)].copy()
        selected_rows += len(chunk)
        if chunk.empty:
            continue
        chunk = chunk[chunk["celllabel"].map(_urine_output_label)].copy()
        chunk["_feature"] = "urine_output"
        candidates += len(chunk)
        accepted += _accumulate_grouped_events(
            accumulator,
            chunk,
            stay_column="patientunitstayid",
            offset_column="intakeoutputoffset",
            feature_column="_feature",
            value_column="cellvaluenumeric",
        )
    return {
        "input_rows_scanned": int(scanned),
        "selected_cohort_rows": int(selected_rows),
        "candidate_observations": int(candidates),
        "accepted_observations": int(accepted),
    }


def scan_eicu_events_vectorized(
    cohort: pd.DataFrame,
    raw_root: Path,
    *,
    max_rows: int,
    sequence_length: int,
    chunksize: int = 250_000,
    accumulator: _EventAccumulator | None = None,
) -> tuple[dict[str, Any], _EventAccumulator]:
    """Scan complete eICU tables with vectorized validation/minute aggregation."""
    selected = cohort.sort_values("stay_key", kind="stable").copy()
    selected_ids = [int(value) for value in selected["stay_key"]]
    if accumulator is None:
        accumulator = _EventAccumulator(
            selected_ids,
            sequence_length=sequence_length,
            retain_only_last=True,
        )
    elif accumulator.stay_ids != selected_ids:
        raise ValueError("event accumulator stay order differs from the eICU cohort")
    stay_set = set(selected_ids)

    tables: dict[str, dict[str, int]] = {}
    tables["vitalPeriodic"] = _accumulate_wide_table(
        accumulator,
        raw_root,
        filename="vitalPeriodic.csv.gz",
        offset_column="observationoffset",
        mapping=EICU_PERIODIC_COLUMNS,
        stay_set=stay_set,
        max_rows=max_rows,
        chunksize=chunksize,
    )
    tables["vitalAperiodic"] = _accumulate_wide_table(
        accumulator,
        raw_root,
        filename="vitalAperiodic.csv.gz",
        offset_column="observationoffset",
        mapping=EICU_APERIODIC_COLUMNS,
        stay_set=stay_set,
        max_rows=max_rows,
        chunksize=chunksize,
    )
    tables["nurseCharting"] = _accumulate_label_table(
        accumulator,
        raw_root,
        filename="nurseCharting.csv.gz",
        offset_column="nursingchartoffset",
        label_column="nursingchartcelltypevalname",
        value_column="nursingchartvalue",
        mapping=EICU_NURSE_LABELS,
        stay_set=stay_set,
        max_rows=max_rows,
        chunksize=chunksize,
    )
    tables["lab"] = _accumulate_label_table(
        accumulator,
        raw_root,
        filename="lab.csv.gz",
        offset_column="labresultoffset",
        label_column="labname",
        value_column="labresult",
        mapping=EICU_LAB_LABELS,
        stay_set=stay_set,
        max_rows=max_rows,
        chunksize=chunksize,
    )
    tables["respiratoryCharting"] = _accumulate_label_table(
        accumulator,
        raw_root,
        filename="respiratoryCharting.csv.gz",
        offset_column="respchartoffset",
        label_column="respchartvaluelabel",
        value_column="respchartvalue",
        mapping=EICU_RESP_LABELS,
        stay_set=stay_set,
        max_rows=max_rows,
        chunksize=chunksize,
    )
    tables["intakeOutput"] = _accumulate_urine_output(
        accumulator,
        raw_root,
        stay_set=stay_set,
        max_rows=max_rows,
        chunksize=chunksize,
    )

    return {
        "selected_stays": int(len(selected_ids)),
        "sequence_length": int(sequence_length),
        "complete_raw_table_scan": max_rows == 0,
        "scanner": "vectorized_chunked_minute_aggregation",
        "chunksize": int(chunksize),
        "tables": tables,
        "apacheApsVar": {
            "file_present": (raw_root / "apacheApsVar.csv.gz").is_file(),
            "injected_into_dynamic_or_static_features": False,
            "reason": (
                "APACHE worst-value timing is not a raw timestamped event; exclusion "
                "prevents temporal look-ahead leakage"
            ),
        },
    }, accumulator
