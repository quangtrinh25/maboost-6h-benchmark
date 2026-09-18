"""Fresh-run integrity for the public benchmark, independent of author artifacts."""
from pathlib import Path
import hashlib
import json
import platform

from landmark_protocol import REPO_ROOT, WORKSPACE_ROOT, RESULTS_ROOT, load_protocol, load_verified_phase_report, now_utc
from v13_guard import safe_write_json, safe_write_text, sha256_file

REGISTRATION = WORKSPACE_ROOT / 'private/run_registration.json'
BUILD = WORKSPACE_ROOT / 'private/correction07_rust_build/BUILD_MANIFEST.json'
QUALIFICATION = WORKSPACE_ROOT / 'private/kernel_qualification.json'


def verify_code(root=REPO_ROOT):
    root = Path(root).resolve()
    path = root / 'CODE_MANIFEST.json'
    manifest = json.loads(path.read_text())
    if not manifest.get('files'):
        raise RuntimeError('Empty code manifest')
    for record in manifest['files']:
        file = (root / record['file']).resolve()
        if not file.is_relative_to(root) or not file.is_file() or sha256_file(file) != record['sha256']:
            raise RuntimeError(f"Code/configuration drift: {record['file']}")
    return {'code_manifest_sha256': sha256_file(path), 'verified_files': len(manifest['files'])}


def locked_json(path):
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + '.sha256')
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f'Missing local run artifact: {path.name}')
    if sidecar.read_text().split()[0] != sha256_file(path):
        raise RuntimeError(f'Artifact hash mismatch: {path.name}')
    return json.loads(path.read_text())


def write_locked(path, payload):
    safe_write_json(path, payload)
    safe_write_text(path.with_suffix(path.suffix + '.sha256'),
                    f'{sha256_file(path)}  {path.relative_to(REPO_ROOT)}\n')


def initialize_run():
    code = verify_code()
    plan, ph, config, ch = load_protocol()
    if REGISTRATION.exists():
        return require_run()
    payload = {'schema_version': '1.0.0', 'mode': 'fresh_public_reproduction',
               'created_at_utc': now_utc(), 'analysis_plan_sha256': ph,
               'config_sha256': ch, **code, 'python': platform.python_version(),
               'analysis_id': plan['analysis_id'],
               'availability_execution_mode': plan['availability_correction']['execution_mode'],
               'availability_correction_sha256': config['availability_correction_sha256'],
               'historical_parent_comparison_performed': False,
               'historical_author_approval_inherited': False,
               'historical_results_imported': False}
    write_locked(REGISTRATION, payload)
    return require_run()


def require_run():
    code = verify_code()
    payload = locked_json(REGISTRATION)
    plan, ph, config, ch = load_protocol()
    for key, value in {'mode': 'fresh_public_reproduction', 'analysis_plan_sha256': ph,
                       'config_sha256': ch, 'code_manifest_sha256': code['code_manifest_sha256'],
                       'analysis_id': plan['analysis_id'],
                       'availability_execution_mode': plan['availability_correction']['execution_mode'],
                       'availability_correction_sha256': config['availability_correction_sha256'],
                       'historical_parent_comparison_performed': False}.items():
        if payload.get(key) != value:
            raise RuntimeError(f'Run registration mismatch: {key}; use a fresh checkout for a new recipe')
    return {**payload, 'record_sha256': sha256_file(REGISTRATION)}


def qualify_kernel():
    import numpy as np
    from correction07_bootstrap import BootstrapInput, RustBootstrapKernel, generate_draw_batch, legacy_differences_for_draws
    from correction07_contract import bounded_thread_count
    require_run()
    engine = json.loads((WORKSPACE_ROOT / 'config/analysis_engine.json').read_text())
    kernel = RustBootstrapKernel.load(BUILD)
    threads = bounded_thread_count(engine)
    # Every patient has both labels, so even repeated single-patient draws have
    # valid AUROC. The same concrete draws exercise Python and Rust kernels.
    codes = np.repeat(np.arange(6), 2)
    labels = np.tile([0., 1.], 6)
    a = np.array([.1,.8,.3,.7,.2,.9,.4,.6,.2,.75,.35,.85])
    b = np.array([.3,.7,.2,.8,.4,.6,.5,.55,.4,.6,.45,.65])
    draws = generate_draw_batch(np.random.default_rng(42), 32, 6)
    checks = []
    for metric in ('auroc','auprc','brier','los_mae','los_rmse'):
        inputs = BootstrapInput.from_arrays(metric, labels, a, b, codes)
        expected = legacy_differences_for_draws(inputs, draws)
        observed = kernel.evaluate_draws(inputs, draws, threads=threads)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
        single = kernel.evaluate_draws(inputs, draws, threads=1)
        np.testing.assert_array_equal(observed, single)
        checks.append({'metric':metric,'max_absolute_error':float(np.max(np.abs(observed-expected)))})
    result = {'status':'pass','data':'synthetic_only','default_threads':threads,
              'build_manifest_sha256':sha256_file(BUILD), 'run_registration_sha256':sha256_file(REGISTRATION),
              'checks':checks,'clinical_cohort_qualification':False}
    write_locked(QUALIFICATION, result)
    return result


def require_analysis_readiness(**unused_legacy_paths):
    from correction07_contract import bounded_thread_count
    from correction07_bootstrap import RustBootstrapKernel
    run = require_run()
    engine = json.loads((WORKSPACE_ROOT / 'config/analysis_engine.json').read_text())
    qualified = locked_json(QUALIFICATION)
    if (qualified.get('status') != 'pass' or qualified.get('default_threads') != bounded_thread_count(engine)
        or qualified.get('build_manifest_sha256') != sha256_file(BUILD)
        or qualified.get('run_registration_sha256') != run['record_sha256']):
        raise RuntimeError('Kernel qualification drift; rerun build-kernel in this environment')
    RustBootstrapKernel.load(BUILD)
    phase_d = load_verified_phase_report('D_FULL', allowed_statuses={'pass'})
    if phase_d.get('run_registration_sha256') != run['record_sha256']:
        raise RuntimeError('Training report belongs to a different run')
    chain = {'mode':'fresh_public_reproduction', 'run_registration_sha256':run['record_sha256'],
             'phase_d_report_sha256':sha256_file(RESULTS_ROOT/'phase_reports/PHASE_REPORT_D_FULL.json'),
             'qualified_threads':qualified['default_threads'],
             'kernel_qualification_sha256':sha256_file(QUALIFICATION)}
    return {'_verified_chain':chain}


def load_analysis_runtime():
    from correction07_bootstrap import RustBootstrapKernel
    readiness = require_analysis_readiness()
    path = WORKSPACE_ROOT/'config/analysis_engine.json'
    engine = json.loads(path.read_text())
    engine['_qualified_threads'] = readiness['_verified_chain']['qualified_threads']
    return engine, sha256_file(path), RustBootstrapKernel.load(BUILD), readiness['_verified_chain']


def _verify_outputs(files, directory):
    # Phase G historically used four logical table names in its inventory.
    # Resolve only the registered aliases, and only in the reporting directory.
    from correction07_contract import CORRECTION07_PHASE_G_TABLE_PATHS
    for name, expected in files.items():
        filename = CORRECTION07_PHASE_G_TABLE_PATHS.get(name, name) if directory.name == 'tables' else name
        path = directory / filename
        locked = path.with_suffix(path.suffix+'.sha256')
        if not path.is_file() or not locked.is_file() or sha256_file(path) != expected or locked.read_text().split()[0] != expected:
            raise RuntimeError(f'Output artifact drift: {name}')


def require_analysis_report():
    chain = require_analysis_readiness()['_verified_chain']
    payload = load_verified_phase_report('E_ANALYSIS', allowed_statuses={'pass'})
    if payload.get('approval_chain') != chain:
        raise RuntimeError('Analysis run binding drift')
    _verify_outputs(payload['files'], RESULTS_ROOT/'canonical')
    report = RESULTS_ROOT/'canonical/canonical_analysis_report.json'
    locked_json(report)
    if sha256_file(report) != payload['canonical_report_sha256']:
        raise RuntimeError('Canonical report binding drift')
    return {**payload, '_phase_report_sha256':sha256_file(RESULTS_ROOT/'phase_reports/PHASE_REPORT_E_ANALYSIS.json')}


def require_explanation_report():
    e = require_analysis_report()
    f = load_verified_phase_report('F_EXPLAINABILITY', allowed_statuses={'pass'})
    if f.get('phase_e_report_sha256') != e['_phase_report_sha256']:
        raise RuntimeError('Explanation predecessor drift')
    _verify_outputs(f['files'], RESULTS_ROOT/'explainability')
    _verify_outputs(f['figures'], RESULTS_ROOT/'figures')
    return {**f,'_verified_phase_e':e,'_phase_report_sha256':sha256_file(RESULTS_ROOT/'phase_reports/PHASE_REPORT_F_EXPLAINABILITY.json')}
