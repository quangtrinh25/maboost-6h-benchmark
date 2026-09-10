from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
