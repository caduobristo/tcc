"""Hidden Windows launcher for an explicitly authorized frozen probe run.

Use pythonw through Win32_Process.Create to keep this launcher outside the
interactive tool host's process job. It installs no service or scheduled task.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def launch(*, authorized=False):
    if not authorized:
        raise RuntimeError('Pass --run explicitly; this starts extraction and frozen heads')
    from scripts.run_temporal_interpolation import LOCK, coordinator_alive
    if LOCK.exists() and coordinator_alive(json.loads(LOCK.read_text())):
        raise RuntimeError('A coordinator is already running')
    audit = ROOT/'data/audits/transfer_learning'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    stem = audit/f'temporal_detached_{stamp}'
    with stem.with_suffix('.log').open('w', encoding='utf-8') as stdout, stem.with_suffix('.stderr.log').open('w', encoding='utf-8') as stderr:
        process = subprocess.Popen([str(ROOT/'.venv-transfer/Scripts/python.exe'), '-u', '-X', 'utf8',
            str(ROOT/'scripts/run_temporal_interpolation.py'), '--run'], cwd=ROOT,
            stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        record = {'launcher_pid': __import__('os').getpid(), 'child_pid': process.pid,
                  'log': stem.with_suffix('.log').relative_to(ROOT).as_posix(),
                  'stderr': stem.with_suffix('.stderr.log').relative_to(ROOT).as_posix(), 'status': 'running'}
        path = audit/'temporal_background_launch.json'
        path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        record.update(exit_code=process.wait(), status='finished')
        path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    launch(authorized=parser.parse_args().run)
