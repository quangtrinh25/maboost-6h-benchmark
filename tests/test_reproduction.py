from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark/src'))
from reproduction_contract import verify_code
from landmark_protocol import load_protocol
from landmark_jobs import build_job_matrix
from landmark_firewall import load_source_training_data
from correction07_ensemble import prepare_seed_ensemble


class ReproductionTests(unittest.TestCase):
    def test_code_lock_rejects_drift_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'model.py'
            source.write_text('x = 1\n')
            manifest = {'files': [{'file': 'model.py', 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]}
            (root / 'CODE_MANIFEST.json').write_text(json.dumps(manifest))
            verify_code(root)
            source.write_text('x = 2\n')
            with self.assertRaises(RuntimeError):
                verify_code(root)
            source.unlink()
            with self.assertRaises(RuntimeError):
                verify_code(root)

    def test_locked_full_matrix_and_landmark(self):
        plan, ph, config, _ = load_protocol()
        self.assertEqual(config['analysis_plan_sha256'], ph)
        self.assertEqual(config['landmark_hours'], 6.0)
        self.assertEqual(config['sequence_length'], 96)
        self.assertEqual(len(config['stress_scenarios']), 28)
        jobs = build_job_matrix(config)
        self.assertEqual(len(jobs), 180)
        self.assertEqual(len({j.job_id for j in jobs}), 180)
        self.assertEqual(len(plan['feature_contract']['static_features']), 12)

    def test_historical_target_cannot_be_used_for_training(self):
        with self.assertRaises(PermissionError):
            load_source_training_data('mimic_iii_carevue')

    def test_bounded_training_cannot_mark_a_real_run_complete(self):
        from run_landmark_job import run_job
        with self.assertRaises(PermissionError):
            run_job('mimic_iv__xgboost_flat__full_only__seed42', bounded=True)

    def test_reporting_table_aliases_verify_real_files_and_reject_drift(self):
        from reproduction_contract import _verify_outputs
        from correction07_contract import CORRECTION07_PHASE_G_TABLE_PATHS
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'tables'
            directory.mkdir()
            files = {}
            for alias in ('table_1.csv', 'table_2.csv', 'table_3A.csv', 'table_3B.csv'):
                path = directory/CORRECTION07_PHASE_G_TABLE_PATHS[alias]
                path.write_text('synthetic,aggregate\n1,2\n')
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                path.with_suffix('.csv.sha256').write_text(digest+'  '+path.name+'\n')
                files[alias] = digest
            _verify_outputs(files, directory)
            path.write_text('changed\n')
            with self.assertRaisesRegex(RuntimeError, 'Output artifact drift'):
                _verify_outputs(files, directory)

    def test_analyze_resumes_partial_canonical_directory(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('public_reproduce', ROOT/'reproduce.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        import landmark_protocol, reproduction_contract
        analysis = Mock()
        analysis.run_analysis.return_value = ({'status': 'pass'}, 0)
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results/'canonical').mkdir()
            with patch.object(sys, 'argv', ['reproduce.py', 'analyze']), patch.object(landmark_protocol, 'RESULTS_ROOT', results), patch.object(reproduction_contract, 'require_run', return_value={}), patch.object(reproduction_contract, 'require_analysis_report') as verify, patch.dict(sys.modules, {'landmark_analysis': analysis}):
                self.assertEqual(cli.main(), 0)
                analysis.run_analysis.assert_called_once_with()
                verify.assert_not_called()
                analysis.run_analysis.reset_mock()
                (results/'phase_reports').mkdir()
                (results/'phase_reports/PHASE_REPORT_E_ANALYSIS.json').write_text('{}')
                self.assertEqual(cli.main(), 0)
                verify.assert_called_once_with()
                analysis.run_analysis.assert_not_called()


if __name__ == '__main__':
    unittest.main()
