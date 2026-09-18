"""Post-audit correction: recorded-time eligibility, with unverified sources disclosed."""
from collections import Counter
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from correction09_timestamps import (
    CATEGORIES, accepted_events, classify_times, event_frames, raw_schema, stat_signature, table_specs,
)
from v13_etl import _EventAccumulator, _accumulate_grouped_events
from v13_guard import sha256_file


def correct_events(frame, has_timestamp):
    codes=classify_times(frame['offset'].to_numpy(float), frame['availability'].to_numpy(float), has_timestamp)
    counts={name:int(np.sum(codes==i)) for i,name in enumerate(CATEGORIES)}
    keep=(codes==0) if has_timestamp else (codes==4)
    corrected=frame.loc[keep].copy()
    if has_timestamp:
        # Ceiling prevents a fractional recording time from entering an earlier
        # integer-minute prefix. Window filtering happened before quantization.
        corrected['offset']=np.ceil(corrected['availability'].to_numpy(float))
        if not corrected.empty and not (
            corrected['availability'].le(corrected['offset']).all()
            and corrected['offset'].le(360).all()
        ):
            raise RuntimeError('Availability timeline invariant failure')
    return corrected,counts


def add_events(accumulator, frame, stay_keys):
    if frame.empty:
        return 0
    work=frame.copy()
    work['stay']=stay_keys[work['row'].to_numpy(dtype=np.int64)]
    return _accumulate_grouped_events(accumulator,work,stay_column='stay',offset_column='offset',
                                      feature_column='feature',value_column='value')


class ScanContext:
    def __init__(self, cohort):
        self.keys=cohort['stay_key'].to_numpy(np.int64)
        if len(np.unique(self.keys))!=len(self.keys):
            raise RuntimeError('Duplicate cohort stay keys')
        self.rows=pd.Series(np.arange(len(self.keys)),index=self.keys)
        self.cohort=cohort[['stay_key','hadm_key','intime']].rename(columns={'stay_key':'stay','hadm_key':'hadm_id'})
        self.intimes=self.cohort.set_index('stay')['intime']


def scan_tables(cohort,raw_root,dataset,*,sequence_length=96,max_rows=0,chunksize=100000,
                specs=None):
    if max_rows < 0:
        raise ValueError("Preview rows cannot be negative")
    selected=cohort.sort_values('stay_key',kind='stable')
    context=ScanContext(selected)
    acc=_EventAccumulator(context.keys,sequence_length=sequence_length,retain_only_last=True)
    tables={}
    for spec in (table_specs(dataset) if specs is None else specs):
        path=Path(raw_root)/spec.filename
        raw_schema(path,spec)
        before=stat_signature(path)
        # Bind each fresh full scan to the reproducer's own raw table.
        raw_hash = sha256_file(path) if max_rows == 0 else None
        started=last=time.monotonic()
        rows=accepted=kept=0
        categories=Counter({name:0 for name in CATEGORIES})
        print(f'CORRECTION09 SCAN {dataset}/{spec.name} rows={max_rows or "FULL"}',flush=True)
        with pd.read_csv(path,usecols=spec.columns(),chunksize=chunksize,nrows=max_rows or None,low_memory=False) as reader:
            for chunk in reader:
                rows+=len(chunk)
                for frame in event_frames(chunk,spec,context):
                    frame=accepted_events(frame)
                    accepted+=len(frame)
                    corrected,counts=correct_events(frame,spec.availability_column is not None)
                    categories.update(counts)
                    added=add_events(acc,corrected,context.keys)
                    if added!=len(corrected):
                        raise RuntimeError('Validated correction observations rejected by accumulator')
                    kept+=added
                if time.monotonic()-last>=15:
                    print(f'  {dataset}/{spec.name}: scanned={rows:,} accepted={accepted:,} kept={kept:,}',flush=True)
                    last=time.monotonic()
        if stat_signature(path)!=before:
            raise RuntimeError('Raw table changed during correction scan')
        tables[spec.name]={'input_rows_scanned':rows,'accepted_before_correction':accepted,
                          'accepted_after_correction':kept,'excluded_observations':accepted-kept,
                          'timestamp_categories':dict(categories),'evidence':spec.evidence,
                          'independent_timestamp':spec.availability_column,
                          'raw_sha256':raw_hash,
                          'wall_seconds':time.monotonic()-started}
        print(f'DONE {dataset}/{spec.name}: kept={kept:,} excluded={accepted-kept:,} evidence={spec.evidence}',flush=True)
    return acc,{'correction':'09','complete_raw_table_scan':max_rows==0,'tables':tables,
                'unverified_sources_retained':True,'universal_availability_claim':False,
                'timeline':'ceil(independent minute) when present; original event minute otherwise',
                'selection':'filter before aggregation/GCS/truncation',
                'eicu_lab_policy':'conservative final-version revision cutoff; first availability not proven'}


CORRECTION_MODE = 'fresh_public_raw_data'
CORRECTION_RECORD = Path(__file__).resolve().parents[1] / 'config/availability_correction09.json'


def load_correction_record(config):
    """Require the public policy lock; historical private artifacts are not inputs."""
    if config.get('availability_correction_sha256') != sha256_file(CORRECTION_RECORD):
        raise RuntimeError('Correction registration mismatch')
    record = json.loads(CORRECTION_RECORD.read_text())
    if (record.get('execution_mode') != CORRECTION_MODE
            or record.get('correction_id') != '09_recorded_availability'):
        raise RuntimeError('Unsupported availability correction mode')
    return record


def scan_corrected(cohort, raw_root, dataset, config, *, max_rows=0):
    record = load_correction_record(config)
    accumulator, summary = scan_tables(
        cohort, raw_root, dataset, sequence_length=config['sequence_length'],
        max_rows=max_rows,
    )
    summary.update({
        'execution_mode': record['execution_mode'],
        'historical_parent_artifacts_required': False,
        'historical_parent_comparison_performed': False,
        'cohort_and_splits': 'rebuilt from governed raw data using the unchanged deterministic algorithms',
        'static_availability': 'unverified; original static feature construction retained',
    })
    return summary, accumulator
