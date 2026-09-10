"""Create the distributed code/configuration inventory before a new run."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'.py','.json','.sha256','.toml','.lock','.rs','.txt','.sh','.yml','.yaml'}


def main():
    if (ROOT/'benchmark/private/run_registration.json').exists():
        raise RuntimeError('Refusing to change code identity after run initialization; use a fresh checkout')
    paths = []
    for prefix in ('benchmark/src','benchmark/config','benchmark/rust_analysis_kernel','src','tests','tools','.github'):
        for path in (ROOT/prefix).rglob('*'):
            if path.is_file() and 'target' not in path.parts and '__pycache__' not in path.parts and path.suffix in ALLOWED:
                paths.append(path)
    paths += [ROOT/'reproduce.py',ROOT/'requirements.txt',ROOT/'benchmark/analysis_plan_v13_landmark_amendment_02.json',ROOT/'benchmark/analysis_plan_v13_landmark_amendment_02.sha256']
    for optional in ('requirements-cpu-tests.txt','setup.sh'):
        if (ROOT/optional).exists():paths.append(ROOT/optional)
    payload = {'schema_version':'1.0.0','purpose':'Public code/configuration integrity; no clinical results',
               'files':[{'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(paths))]}
    (ROOT/'CODE_MANIFEST.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(f'Locked {len(payload["files"])} source/configuration files')


if __name__ == '__main__':main()
