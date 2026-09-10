"""Bounded training/reload/firewall smoke in an isolated synthetic checkout."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def worker():
    if not (ROOT / 'SYNTHETIC_SMOKE_ONLY').is_file():
        raise RuntimeError('Synthetic smoke worker requires its own isolated checkout')
    sys.path.insert(0, str(ROOT / 'benchmark/src'))
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from reproduction_contract import initialize_run, require_run
    from landmark_protocol import load_protocol, write_phase_report
    from landmark_etl import _write_dataset
    from landmark_firewall import load_features, open_target_labels_for_locked_evaluation
    from run_landmark_job import run_job
    from landmark_jobs import build_job_matrix, LandmarkJobStore
    initialize_run()
    _, ph, config, ch = load_protocol()
    rng = np.random.default_rng(42)
    n = 120
    for dataset in config['datasets']:
        mask = (rng.random((n,96,40)) > .8).astype(np.uint8)
        values = rng.normal(size=mask.shape).astype(np.float32)
        values[mask == 0] = np.nan
        arrays = {'values':values,'mask':mask,'tau':np.full((n,96),225.,dtype=np.float32),
                  'elapsed':np.tile(np.linspace(0,21600,96,dtype=np.float32),(n,1)),
                  'static':rng.normal(size=(n,12)).astype(np.float32),
                  'mortality':np.tile([0,1],n//2).astype(np.int8),
                  'los':rng.uniform(.1,4,size=n).astype(np.float32),
                  'stay_keys':np.array([f'synthetic-stay-{i}' for i in range(n)]),
                  'group_keys':np.array([f'synthetic-person-{i}' for i in range(n)]),
                  'age':np.full(n,75,dtype=np.float32),'sex_male':np.tile([0,1],n//2).astype(np.int8),
                  'icu_type_code':np.zeros(n,dtype=np.int16)}
        _write_dataset(dataset,arrays,{'synthetic_only':True},ch,ph,config)
        assert set(load_features(dataset)) == {'values','mask','tau','elapsed','static'}
    write_phase_report('B_ETL',{'status':'pass','synthetic_fixture_only':True,
                              'analysis_plan_sha256':ph,'config_sha256':ch})
    try:
        open_target_labels_for_locked_evaluation('eicu',prediction_path=ROOT/'missing.npz',source_choices_path=ROOT/'missing.json')
    except (PermissionError,FileNotFoundError,RuntimeError):
        pass
    else:
        raise AssertionError('Target labels opened before prediction lock')
    jobs = [j for j in build_job_matrix(config) if j.source_dataset == 'mimic_iv' and j.seed == 42 and j.model in {'XGBOOST-FLAT','LSTM'} and j.training_regime == 'full_only']
    summaries = []
    for job in jobs:
        summaries.append(run_job(job.job_id,bounded=True))
        assert LandmarkJobStore().is_complete(job)
        assert run_job(job.job_id,bounded=True)['status'] == 'already_complete'
    assert len(summaries) == 2
    require_run()
    print(json.dumps({'status':'pass','synthetic_only':True,'bounded_jobs':2,
                      'scenarios_per_job':28,'routes_per_job':3,
                      'reload_and_prediction_lock':'pass','verified_resume':'pass',
                      'full_cohort_or_mamba_training_run':False},indent=2))


def main():
    if '--worker' in sys.argv:
        worker();return
    from tempfile import TemporaryDirectory
    # No governed data or existing run files are copied into the smoke checkout.
    manifest = json.loads((ROOT/'CODE_MANIFEST.json').read_text())
    with TemporaryDirectory(prefix='maboost6h-synthetic-') as folder:
        child = Path(folder)
        for item in manifest['files']:
            target = child/item['file'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/item['file'],target)
        shutil.copy2(ROOT/'CODE_MANIFEST.json',child/'CODE_MANIFEST.json')
        (child/'SYNTHETIC_SMOKE_ONLY').write_text('Generated synthetic fixture only.\n')
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES']=''
        env['OMP_NUM_THREADS']='2'
        env['MKL_NUM_THREADS']='2'
        subprocess.run([sys.executable,'-B',str(child/'tools/synthetic_smoke.py'),'--worker'],cwd=child,env=env,check=True)


if __name__ == '__main__':main()
