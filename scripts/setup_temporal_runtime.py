"""Inspect or prepare the isolated AST/AudioMAE compatibility runtime on Windows.

Run from the prepared transfer environment. Preparation shares its verified
CUDA packages and installs only the pinned legacy dependencies in a new venv.
The default is read-only and never installs or starts scientific execution.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE_PYTHON = ROOT / '.venv-transfer/Scripts/python.exe'
LEGACY = ROOT / '.venv-temporal-legacy'
LEGACY_PYTHON = LEGACY / 'Scripts/python.exe'


def query(python, code):
    result = subprocess.run([str(python), '-X', 'utf8', '-c', code], cwd=ROOT,
                            capture_output=True, text=True, encoding='utf-8', check=True)
    return json.loads(result.stdout)


def setup(*, prepare=False):
    if not BASE_PYTHON.is_file():
        raise FileNotFoundError('Prepare .venv-transfer with the pinned CUDA stack first')
    if prepare:
        import psutil
        lock = ROOT / 'data/audits/transfer_learning/temporal_interpolation_execution.lock'
        if lock.exists() and psutil.pid_exists(json.loads(lock.read_text())['pid']):
            raise RuntimeError('Do not modify the runtime while the experiment coordinator is alive')
        if not LEGACY_PYTHON.is_file():
            subprocess.run([str(BASE_PYTHON), '-m', 'venv', str(LEGACY)], cwd=ROOT, check=True)
        paths = query(BASE_PYTHON, "import json,sys; from pathlib import Path; print(json.dumps([str(Path(p).resolve()) for p in sys.path if p and Path(p).name=='site-packages' and Path(p).is_dir()]))")
        if not paths:
            raise RuntimeError('Cannot locate the prepared runtime package paths')
        own_packages = LEGACY / 'Lib/site-packages'
        if not own_packages.is_dir():
            raise RuntimeError('Unexpected venv layout; this setup script targets Windows')
        (own_packages / 'shared_transfer_runtime.pth').write_text('\n'.join(dict.fromkeys(paths))+'\n', encoding='utf-8')
        subprocess.run([str(LEGACY_PYTHON), '-m', 'pip', 'install', '--no-deps', '-r',
                        str(ROOT / 'requirements-temporal-legacy.txt')], cwd=ROOT, check=True)
    if not LEGACY_PYTHON.is_file():
        raise FileNotFoundError('Legacy runtime missing; pass --prepare to create it')
    inventory = query(LEGACY_PYTHON, "import json,torch,timm,importlib.metadata as m; print(json.dumps({'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'timm':timm.__version__,'safetensors':m.version('safetensors'),'wget':m.version('wget')}))")
    if inventory['timm'] != '0.4.5' or not inventory['cuda_available']:
        raise RuntimeError(f'Legacy runtime is not ready: {inventory}')
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return inventory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Create/install the isolated runtime; no training')
    setup(prepare=parser.parse_args().prepare)
