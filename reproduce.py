#!/usr/bin/env python3
"""Run the fixed-6-hour benchmark from a clean checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'benchmark/src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('init','check','matrix','build-kernel','preflight','etl','analyze','explain','report','smoke','all'):
        sub.add_parser(name)
    training = sub.add_parser('train')
    training.add_argument('--job', help='One registered job ID; omit for all 180')
    training.add_argument('--resume', action='store_true', help='Skip hash-verified completed jobs (also the default)')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'benchmark/private/matplotlib'))
    from reproduction_contract import initialize_run, require_run, verify_code
    from landmark_protocol import load_protocol, write_phase_report, WORKSPACE_ROOT, RESULTS_ROOT
    if args.command == 'init':
        result = initialize_run()
    elif args.command == 'check':
        import importlib.util
        result = verify_code()
        missing = [name for name in ('numpy','pandas','sklearn','scipy','torch','xgboost','shap','matplotlib','mamba_ssm','psutil','PIL') if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError(f'Missing dependencies: {missing}')
        result['dependencies'] = 'importable'
    elif args.command == 'matrix':
        from dataclasses import asdict
        from landmark_jobs import build_job_matrix
        result = [asdict(job) for job in build_job_matrix(load_protocol()[2])]
    elif args.command == 'smoke':
        subprocess.run([sys.executable,'-B',str(ROOT/'tools/synthetic_smoke.py')],check=True)
        return 0
    elif args.command == 'all':
        for command in ('init','check','build-kernel','preflight','etl','train','analyze','explain','report'):
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),command],check=True)
        return 0
    else:
        identity = require_run()
        if args.command == 'build-kernel':
            from build_correction07_rust import build_correction07_rust
            from reproduction_contract import qualify_kernel
            build_correction07_rust()
            subprocess.run(['cargo','test','--release','--locked'],cwd=WORKSPACE_ROOT/'rust_analysis_kernel',check=True)
            result = qualify_kernel()
        elif args.command == 'train':
            from landmark_jobs import build_job_matrix, LandmarkJobStore
            from landmark_protocol import load_verified_phase_report
            from landmark_analysis import _completion_inventory_sha256
            load_verified_phase_report('B_ETL', allowed_statuses={'pass'})
            _, ph, config, ch = load_protocol()
            jobs = build_job_matrix(config)
            if args.job and args.job not in {j.job_id for j in jobs}:
                raise ValueError(f'Unknown job: {args.job}')
            selected = [j for j in jobs if not args.job or j.job_id == args.job]
            store = LandmarkJobStore()
            logs = RESULTS_ROOT/'logs/training'
            logs.mkdir(parents=True, exist_ok=True)
            for index, job in enumerate(selected, 1):
                if store.is_complete(job):
                    print(f'[{index}/{len(selected)}] verified complete: {job.job_id}',flush=True)
                    continue
                print(f'[{index}/{len(selected)}] training: {job.job_id}; log: {logs.relative_to(ROOT)}/{job.job_id}.log',flush=True)
                with (logs/f'{job.job_id}.log').open('a') as log:
                    subprocess.run([sys.executable,'-B','-u',str(WORKSPACE_ROOT/'src/run_landmark_job.py'),'--job-id',job.job_id],stdout=log,stderr=subprocess.STDOUT,check=True)
            complete = sum(store.is_complete(j) for j in jobs)
            result = {'completed_jobs':complete,'expected_jobs':len(jobs)}
            if complete == len(jobs):
                write_phase_report('D_FULL',{'status':'pass','mode':'fresh_public_reproduction',
                    'analysis_plan_sha256':ph,'config_sha256':ch,'jobs':complete,
                    'run_registration_sha256':identity['record_sha256'],
                    'completion_inventory_sha256':_completion_inventory_sha256(jobs)})
        else:
            import importlib
            functions = {'preflight':('landmark_preflight','run_preflight'),
                         'etl':('landmark_etl','run_etl'), 'analyze':('landmark_analysis','run_analysis'),
                         'explain':('landmark_explainability','run_explainability'),
                         'report':('landmark_reporting','run_reporting')}
            module, name = functions[args.command]
            if args.command == 'analyze' and (RESULTS_ROOT/'canonical').exists():
                from reproduction_contract import require_analysis_report
                require_analysis_report()
                result, code = {'status':'already_complete_verified'}, 0
            else:
                result, code = getattr(importlib.import_module(module),name)()
            print(json.dumps(result,indent=2))
            return code
    print(json.dumps(result,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
