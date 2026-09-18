from pathlib import Path
import sys
import tempfile
import unittest
import importlib
import json
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark/src'))


class AvailabilityCorrectionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.c = importlib.import_module('correction09_availability')
        except ModuleNotFoundError:
            self.fail('Availability correction has not been implemented')

    def frame(self, offsets, available, values=None, features=None):
        n=len(offsets)
        return pd.DataFrame({'row':[0]*n,'offset':offsets,'availability':available,
                             'value':values if values is not None else [80]*n,
                             'feature':features if features is not None else ['heart_rate']*n})

    def test_known_times_drop_late_missing_and_before_event(self):
        work=self.frame([10,10,10,10,360],[20,361,np.nan,9,360])
        filtered,counts=self.c.correct_events(work,True)
        self.assertEqual(filtered['offset'].tolist(),[20,360])
        self.assertEqual(counts,{'within_6h':2,'after_6h':1,'missing_or_invalid':1,'before_event':1,'no_independent_timestamp':0})

    def test_fractional_recording_time_never_moves_before_minute_cutoff(self):
        out,_=self.c.correct_events(self.frame([10,120,359.5],[120.01,120,359.99]),True)
        self.assertEqual(out['offset'].tolist(),[121,120,360])

    def test_unverified_tables_keep_original_event_bins_and_report_unknown(self):
        out,counts=self.c.correct_events(self.frame([10.5,360],[np.nan,np.nan]),False)
        self.assertEqual(out['offset'].tolist(),[10.5,360])
        self.assertEqual(counts['no_independent_timestamp'],2)

    def test_filter_precedes_mean_and_gcs_derivation(self):
        from v13_etl import _EventAccumulator
        from v13_feature_registry import DYNAMIC_FEATURES
        acc=_EventAccumulator([101],sequence_length=96,retain_only_last=True)
        frame=self.frame([10]*5,[20,400,20,20,400],[60,180,4,6,5],
                         ['heart_rate','heart_rate','gcs_eye','gcs_motor','gcs_verbal'])
        corrected,_=self.c.correct_events(frame,True)
        self.c.add_events(acc,corrected,np.array([101]))
        arrays,_=acc.materialize_arrays()
        self.assertEqual(arrays['values'][0,-1,0],60)
        self.assertEqual(arrays['elapsed'][0,-1],1200)
        self.assertEqual(arrays['mask'][0,-1,DYNAMIC_FEATURES.index('gcs_total')],0)

    def test_dropped_late_minute_does_not_evict_earlier_valid_minute(self):
        from v13_etl import _EventAccumulator
        acc=_EventAccumulator([101],sequence_length=2,retain_only_last=True)
        corrected,_=self.c.correct_events(self.frame([10,20,30],[10,20,400]),True)
        self.c.add_events(acc,corrected,np.array([101]))
        arrays,_=acc.materialize_arrays()
        self.assertEqual(arrays['elapsed'][0].tolist(),[600,1200])
        self.assertEqual(arrays['values'][0,:,0].tolist(),[80,80])

    def test_scan_real_csv_shape_and_version_rejection_before_aggregation(self):
        from correction09_timestamps import table_specs
        cohort=pd.DataFrame({'stay_key':[101], 'hadm_key':[20], 'intime':pd.to_datetime(['2100-01-01'])})
        spec=next(s for s in table_specs('mimic_iv') if s.name=='labevents')
        raw=pd.DataFrame({'hadm_id':[20,20],'itemid':[50912,50912],
                          'charttime':['2100-01-01 00:10:00']*2,
                          'storetime':['2100-01-01 00:20:00','2100-01-01 07:00:00'],
                          'valuenum':[1,9]})
        with tempfile.TemporaryDirectory() as tmp:
            raw.to_csv(Path(tmp)/spec.filename,index=False,compression='gzip')
            acc,audit=self.c.scan_tables(cohort,Path(tmp),'mimic_iv',sequence_length=96,
                                         max_rows=0,chunksize=1,specs=[spec])
            arrays,_=acc.materialize_arrays()
            self.assertEqual(arrays['values'][0,-1,20],1.)
            self.assertEqual(audit['tables']['labevents']['accepted_before_correction'],2)
            self.assertEqual(audit['tables']['labevents']['accepted_after_correction'],1)

    def test_bounded_etl_cannot_write_production_dataset(self):
        from landmark_etl import _write_dataset
        from v13_guard import WORKSPACE_ROOT
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            config={'availability_correction_sha256':'test','datasets':{'mimic_iv':{'private_root':str(Path(tmp)/'data')}}}
            with self.assertRaisesRegex(RuntimeError,'complete correction09'):
                _write_dataset('mimic_iv',{}, {'availability_correction':{'complete_raw_table_scan':False}},'config','plan',config)


    def test_all_twelve_table_parsers_apply_timestamp_policy(self):
        from correction09_timestamps import table_specs
        from v13_feature_registry import PLAUSIBLE_RANGES
        cohort = pd.DataFrame({'stay_key': [101], 'hadm_key': [20],
                               'intime': pd.to_datetime(['2100-01-01'])})
        available = [20, 361, np.nan, 9, 360, np.inf]
        offsets = [10, 10, 10, 10, 360, 10]
        for dataset in ('mimic_iv', 'eicu', 'mimic_iii_carevue'):
            for spec in table_specs(dataset):
                with self.subTest(dataset=dataset, table=spec.name), tempfile.TemporaryDirectory() as tmp:
                    raw = {column: [None]*6 for column in spec.columns()}
                    raw[spec.stay_column] = [20 if spec.kind == 'lab_join' else 101]*6
                    def timestamps(minutes):
                        return [(pd.Timestamp('2100-01-01')+pd.Timedelta(minutes=value)).strftime('%Y-%m-%d %H:%M:%S')
                                if np.isfinite(value) else None for value in minutes]
                    raw[spec.event_column] = timestamps(offsets) if spec.absolute_time else offsets
                    if spec.availability_column:
                        raw[spec.availability_column] = timestamps(available) if spec.absolute_time else available
                    if spec.flag_column:
                        raw[spec.flag_column] = [0]*6
                    if spec.kind == 'urine':
                        feature, label = 'urine_output', 'Urine'
                    else:
                        label, feature = next(iter(spec.mapping.items()))
                    low, high = PLAUSIBLE_RANGES[feature]
                    value = (low + high) / 2
                    if spec.kind == 'wide':
                        raw[label] = [value]*6
                    else:
                        raw[spec.label_column] = [label]*6
                        raw[spec.value_column] = [value]*6
                    pd.DataFrame(raw).to_csv(Path(tmp)/spec.filename, index=False, compression='gzip')
                    _, audit = self.c.scan_tables(cohort, Path(tmp), dataset, specs=[spec], chunksize=2)
                    report = audit['tables'][spec.name]
                    self.assertEqual(report['accepted_before_correction'], 6)
                    self.assertEqual(report['accepted_after_correction'], 2 if spec.availability_column else 6)
                    self.assertEqual(report['timestamp_categories']['no_independent_timestamp'], 0 if spec.availability_column else 6)

    def test_policy_is_bound_and_identifies_a_fresh_public_run(self):
        from landmark_protocol import load_protocol
        plan, _, config, _ = load_protocol()
        record = self.c.load_correction_record(config)
        self.assertEqual(record['execution_mode'], 'fresh_public_raw_data')
        self.assertEqual(plan['analysis_id'], 'maboost_6h_correction09_public_reproduction')
        self.assertEqual(plan['availability_correction'], record)
        self.assertFalse(record['historical_parent_artifacts_required'])
        self.assertFalse(any(key.startswith('parent_') for key in record))
        with self.assertRaisesRegex(RuntimeError, 'registration mismatch'):
            self.c.load_correction_record({**config, 'availability_correction_sha256': 'wrong'})

    def test_fresh_raw_etl_all_datasets_without_parent_artifacts(self):
        import landmark_etl
        from landmark_protocol import load_protocol
        from correction09_timestamps import table_specs
        from v13_etl import build_static_feature_matrix
        from v13_feature_registry import DYNAMIC_FEATURES
        config = load_protocol()[2]
        for dataset in config['datasets']:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory(dir=landmark_etl.WORKSPACE_ROOT) as tmp:
                root = Path(tmp)
                raw = root / config['datasets'][dataset]['raw_root']
                raw.mkdir(parents=True)
                self._cohort_fixture(raw, dataset)
                specs = table_specs(dataset)
                for spec in specs:
                    # Real headers for all 12 source tables, including timestamp
                    # columns on otherwise empty tables.
                    pd.DataFrame(columns=spec.columns()).to_csv(raw/spec.filename, index=False, compression='gzip')
                spec = next(s for s in specs if s.name in {'lab', 'labevents'})
                if dataset == 'eicu':
                    data = {'patientunitstayid': [1, 1], 'labname': ['creatinine']*2,
                            'labresultoffset': [10, 10], 'labresultrevisedoffset': [20, 400],
                            'labresult': [1, 9]}
                else:
                    data = {'hadm_id': [21, 21], 'itemid': [50912]*2,
                            'charttime': ['2020-01-01 00:10:00']*2, 'valuenum': [1, 9]}
                    if dataset == 'mimic_iv':
                        data['storetime'] = ['2020-01-01 00:20:00', '2020-01-01 07:00:00']
                pd.DataFrame(data).to_csv(raw/spec.filename, index=False, compression='gzip')
                with patch.object(landmark_etl, 'REPO_ROOT', root):
                    cohort, arrays, audit = landmark_etl._build_dataset(dataset, config)
                    manifest = landmark_etl._write_dataset(dataset, arrays, audit, 'synthetic-config', 'synthetic-plan', config)
                self.assertEqual(arrays['stay_keys'].tolist(), [1, 3])
                self.assertEqual(arrays['mortality'].tolist(), [1, 0])
                np.testing.assert_allclose(arrays['los'], [0.25, 1.0])
                self.assertEqual(arrays['mask'][1].sum(), 0)  # Missing inputs never remove a stay.
                expected_static, _ = build_static_feature_matrix(cohort.frame, raw, dataset=dataset, max_rows=0)
                np.testing.assert_array_equal(arrays['static'], expected_static)
                availability = audit['availability_correction']
                self.assertEqual(availability['execution_mode'], 'fresh_public_raw_data')
                self.assertFalse(availability['historical_parent_comparison_performed'])
                self.assertTrue(availability['complete_raw_table_scan'])
                self.assertEqual(len(availability['tables']), len(specs))
                for table in specs:
                    self.assertEqual(availability['tables'][table.name]['raw_sha256'], self.c.sha256_file(raw/table.filename))
                    self.assertEqual(manifest['dynamic_raw_table_sha256'][table.name], self.c.sha256_file(raw/table.filename))
                creatinine = DYNAMIC_FEATURES.index('creatinine')
                if dataset == 'mimic_iii_carevue':
                    self.assertEqual(arrays['values'][0,-1,creatinine], 5)
                    self.assertEqual(arrays['elapsed'][0,-1], 600)
                else:
                    self.assertEqual(arrays['values'][0,-1,creatinine], 1)
                    self.assertEqual(arrays['elapsed'][0,-1], 1200)
                self.assertEqual(audit['static']['availability_class'], 'per_field_availability_unverified')

    @staticmethod
    def _cohort_fixture(raw, dataset):
        def write(name, data):
            pd.DataFrame(data).to_csv(raw/name, index=False, compression='gzip')
        if dataset == 'eicu':
            write('patient.csv.gz', {'patientunitstayid': [3, 2, 1],
                  'patienthealthsystemstayid': [23, 22, 21], 'uniquepid': ['p3', 'p2', 'p1'],
                  'age': [70]*3, 'gender': ['Male']*3, 'unittype': ['MICU']*3,
                  'unitstaytype': ['admit']*3, 'unitdischargestatus': ['Alive', 'Alive', 'Expired'],
                  'unitdischargeoffset': [1800, 360, 720]})
            return
        iv = dataset == 'mimic_iv'
        def filename(name):
            return (name if iv else name.upper()) + '.csv.gz'
        write(filename('icustays'), {'stay_id' if iv else 'icustay_id': [3, 2, 1],
              'subject_id': [13, 12, 11], 'hadm_id': [23, 22, 21],
              'intime': ['2020-01-01 00:00:00']*3,
              'outtime': ['2020-01-02 06:00:00', '2020-01-01 06:00:00', '2020-01-01 12:00:00'],
              'first_careunit': ['MICU']*3, 'dbsource': ['carevue']*3})
        write(filename('admissions'), {'hadm_id': [21, 22, 23],
              'deathtime': ['2020-01-01 08:00:00', None, None],
              'hospital_expire_flag': [1, 0, 0], 'admission_type': ['EMERGENCY']*3,
              'insurance': ['Medicare']*3})
        patients = {'subject_id': [11, 12, 13], 'gender': ['M']*3}
        patients.update({'anchor_age': [70]*3, 'anchor_year': [2020]*3} if iv
                        else {'dob': ['1950-01-01']*3, 'dod_hosp': [None]*3})
        write(filename('patients'), patients)

    def test_missing_required_recording_column_fails_closed(self):
        from correction09_timestamps import raw_schema, table_specs
        spec = table_specs('mimic_iv')[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/spec.filename
            pd.DataFrame(columns=[c for c in spec.columns() if c != 'storetime']).to_csv(path,index=False,compression='gzip')
            with self.assertRaisesRegex(RuntimeError, 'storetime'):
                raw_schema(path, spec)

    def test_fresh_splits_reproduce_hash_partitions_without_parent_files(self):
        import landmark_jobs as jobs
        from landmark_protocol import load_protocol
        config = load_protocol()[2]
        arrays = {'group_keys': np.repeat([f'synthetic-person-{i}' for i in range(60)], 2),
                  'mortality': np.tile([0, 1], 60), 'values': np.zeros((120, 1, 1))}
        with tempfile.TemporaryDirectory(dir=jobs.WORKSPACE_ROOT) as tmp:
            root = Path(tmp)
            with patch.object(jobs, 'REPO_ROOT', root), patch.object(jobs, 'WORKSPACE_ROOT', root/'benchmark'), patch.object(jobs, 'assert_safe_output_path', side_effect=lambda path: path):
                for seed in config['seeds']:
                    bindings = dict(plan_hash='synthetic-plan', config_hash='synthetic-config',
                                    etl_manifest_hash='synthetic-etl', config=config)
                    indices, audit, _ = jobs.load_or_create_source_split('eicu', seed, arrays, **bindings)
                    resumed, _, _ = jobs.load_or_create_source_split('eicu', seed, arrays, **bindings)
                    for partition, rows in indices.items():
                        np.testing.assert_array_equal(rows, resumed[partition])
                        for group in arrays['group_keys'][rows]:
                            score = jobs._group_score(group, seed)
                            self.assertEqual(partition, 'train' if score < .6 else 'validation' if score < .8 else 'test')
                    self.assertFalse(audit['label_stratification'])
                    self.assertFalse(any(audit['group_overlap'].values()))


if __name__=='__main__': unittest.main()
