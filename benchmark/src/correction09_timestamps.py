"""Correction09 raw timestamp parser, independent of historical audit artifacts.

The table definitions and parsing/filtering functions are carried over from the
completed correction09 audit. This module does not open saved cohorts or reports.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip

import numpy as np
import pandas as pd

from v13_etl import _reverse_item_map, _urine_output_label
from v13_feature_registry import (
    PLAUSIBLE_RANGES, MIMIC_CHART_ITEMS, MIMIC_LAB_ITEMS,
    MIMIC_URINE_OUTPUT_ITEMS, EICU_PERIODIC_COLUMNS, EICU_APERIODIC_COLUMNS,
    EICU_NURSE_LABELS, EICU_LAB_LABELS, EICU_RESP_LABELS,
)

CATEGORIES = ("within_6h", "after_6h", "missing_or_invalid", "before_event", "no_independent_timestamp")


@dataclass(frozen=True)
class TableSpec:
    name: str
    filename: str
    stay_column: str
    event_column: str
    availability_column: str | None
    evidence: str
    mapping: dict
    label_column: str = ""
    value_column: str = ""
    flag_column: str = ""
    kind: str = "label"
    absolute_time: bool = False

    def columns(self):
        columns = [self.stay_column, self.event_column]
        columns += list(self.mapping) if self.kind == "wide" else [self.label_column, self.value_column]
        columns += [self.availability_column, self.flag_column]
        return list(dict.fromkeys(c for c in columns if c))


def table_specs(dataset):
    if dataset in ("mimic_iv", "mimic_iii_carevue"):
        iv = dataset == "mimic_iv"
        stay = "stay_id" if iv else "icustay_id"
        def filename(name):
            return (name if iv else name.upper()) + ".csv.gz"
        return [
            TableSpec("labevents", filename("labevents"), "hadm_id", "charttime",
                      "storetime" if iv else None, "lab_result_availability" if iv else "unverifiable",
                      _reverse_item_map(MIMIC_LAB_ITEMS), "itemid", "valuenum", kind="lab_join", absolute_time=True),
            TableSpec("chartevents", filename("chartevents"), stay, "charttime", "storetime",
                      "recording_time_proxy", _reverse_item_map(MIMIC_CHART_ITEMS), "itemid", "valuenum",
                      "warning" if iv else "error", kind="item", absolute_time=True),
            TableSpec("outputevents", filename("outputevents"), stay, "charttime", "storetime",
                      "recording_time_proxy", {i: "urine_output" for i in MIMIC_URINE_OUTPUT_ITEMS},
                      "itemid", "value", kind="item", absolute_time=True),
        ]
    if dataset != "eicu":
        raise ValueError("Unknown dataset")
    return [
        TableSpec("lab", "lab.csv.gz", "patientunitstayid", "labresultoffset", "labresultrevisedoffset",
                  "revision_time_only", EICU_LAB_LABELS, "labname", "labresult"),
        TableSpec("nurseCharting", "nurseCharting.csv.gz", "patientunitstayid", "nursingchartoffset",
                  "nursingchartentryoffset", "entry_time_proxy", EICU_NURSE_LABELS,
                  "nursingchartcelltypevalname", "nursingchartvalue"),
        TableSpec("respiratoryCharting", "respiratoryCharting.csv.gz", "patientunitstayid", "respchartoffset",
                  "respchartentryoffset", "entry_time_proxy", EICU_RESP_LABELS, "respchartvaluelabel", "respchartvalue"),
        TableSpec("intakeOutput", "intakeOutput.csv.gz", "patientunitstayid", "intakeoutputoffset",
                  "intakeoutputentryoffset", "entry_time_proxy", {}, "celllabel", "cellvaluenumeric", kind="urine"),
        TableSpec("vitalPeriodic", "vitalPeriodic.csv.gz", "patientunitstayid", "observationoffset", None,
                  "unverifiable", EICU_PERIODIC_COLUMNS, kind="wide"),
        TableSpec("vitalAperiodic", "vitalAperiodic.csv.gz", "patientunitstayid", "observationoffset", None,
                  "unverifiable", EICU_APERIODIC_COLUMNS, kind="wide"),
    ]


def event_frames(chunk, spec, context):
    if spec.kind == "lab_join":
        keep = chunk[spec.stay_column].isin(context.cohort["hadm_id"]) & chunk[spec.label_column].isin(spec.mapping)
        work = chunk.loc[keep].merge(context.cohort, on="hadm_id", how="inner", validate="many_to_many")
        work["row"] = work["stay"].map(context.rows)
    else:
        work = chunk.loc[chunk[spec.stay_column].isin(context.rows.index)].copy()
        work["row"] = work[spec.stay_column].map(context.rows)
        if spec.flag_column:
            work = work.loc[pd.to_numeric(work[spec.flag_column], errors="coerce").fillna(0) == 0].copy()
        if spec.absolute_time:
            work["intime"] = work[spec.stay_column].map(context.intimes)
    if work.empty:
        return
    if spec.absolute_time:
        for column, target in [(spec.event_column, "offset"), (spec.availability_column, "availability")]:
            if column:
                parsed = pd.to_datetime(work[column], format="%Y-%m-%d %H:%M:%S", errors="coerce")
                work[target] = (parsed - work["intime"]).dt.total_seconds() / 60
            else:
                work[target] = np.nan
    else:
        work["offset"] = pd.to_numeric(work[spec.event_column], errors="coerce")
        work["availability"] = (pd.to_numeric(work[spec.availability_column], errors="coerce")
                                if spec.availability_column else np.nan)
    base = ["row", "offset", "availability"]
    if spec.kind == "wide":
        for column, feature in spec.mapping.items():
            out = work.loc[work[column].notna(), base + [column]].rename(columns={column: "value"}).copy()
            out["feature"] = feature
            yield out
    else:
        if spec.kind == "urine":
            work = work.loc[work[spec.label_column].map(_urine_output_label)].copy()
            work["feature"] = "urine_output"
        else:
            labels = work[spec.label_column]
            if spec.kind == "label":
                labels = labels.astype(str).str.strip().str.lower()
            work["feature"] = labels.map(spec.mapping)
        yield work.loc[work["feature"].notna(), base + ["feature", spec.value_column]].rename(
            columns={spec.value_column: "value"})


def accepted_events(frame):
    work = frame.copy()
    for name in ("offset", "value"):
        work[name] = pd.to_numeric(work[name], errors="coerce")
    valid = np.isfinite(work["offset"]) & np.isfinite(work["value"]) & work["offset"].between(0, 360)
    work = work.loc[valid].copy()
    fio2 = work["feature"].eq("fio2") & work["value"].gt(0) & work["value"].le(1)
    work["value"] = work["value"].astype(float)
    work.loc[fio2, "value"] *= 100
    lower = work["feature"].map({k: v[0] for k, v in PLAUSIBLE_RANGES.items()})
    upper = work["feature"].map({k: v[1] for k, v in PLAUSIBLE_RANGES.items()})
    return work.loc[work["value"].ge(lower) & work["value"].le(upper)].copy()


def classify_times(event, available, has_timestamp):
    if not has_timestamp:
        return np.full(len(event), 4, dtype=np.int8)
    out = np.full(len(event), 2, dtype=np.int8)
    finite = np.isfinite(available)
    out[finite & (available >= event) & (available <= 360)] = 0
    out[finite & (available > 360)] = 1
    out[finite & (available < event)] = 3
    return out


def raw_schema(path, spec):
    with gzip.open(path, "rt", newline="") as handle:
        columns = next(csv.reader(handle))
    missing = sorted(set(spec.columns()) - set(columns))
    if missing:
        raise RuntimeError(f"{spec.name}: required audit columns missing: {missing}")
    return {"table": spec.name, "file": spec.filename, "compressed_bytes": path.stat().st_size,
            "event_time": spec.event_column, "independent_time": spec.availability_column,
            "evidence": spec.evidence, "columns_verified": spec.columns()}


def stat_signature(path):
    info = path.stat()
    return [info.st_size, info.st_mtime_ns, info.st_ino]
