"""Execute authorized frozen probes with preserved controls and isolated outputs."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUDIT = ROOT/'data/audits/transfer_learning'
STATE = AUDIT/'temporal_interpolation_execution.json'
LOCK = AUDIT/'temporal_interpolation_execution.lock'
PLAN = ROOT/'configs/linear_probe/temporal_interpolation.json'


def coordinator_alive(saved):
    """A reused Windows PID alone is not evidence of a live coordinator."""
    import psutil
    try:
        process = psutil.Process(saved['pid'])
        command = process.cmdline()
        expected = str(Path(__file__).resolve()).casefold()
        if not any(str(Path(arg).resolve()).casefold() == expected for arg in command):
            return False
        if '--worker' in command:
            return False
        return 'process_created_at' not in saved or abs(process.create_time()-saved['process_created_at']) < .01
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        raise RuntimeError('Cannot verify the saved coordinator; inspect it before retrying')


def runtime(model):
    return ROOT/('.venv-transfer' if model == 'passt' else '.venv-temporal-legacy')/'Scripts/python.exe'


def original_passt(scenario):
    from src.data.transfer import sha256
    from src.training.linear_probe import load_config, validate_cache
    stem = 'transfer_learning_consolidated' if scenario == 'mono_disc' else f'transfer_learning_{scenario}_consolidated'
    portable = json.loads((ROOT/'results/passt_htsat_transfer'/f'{stem}_summary.json').read_text())
    cache = validate_cache(load_config(ROOT/'configs/linear_probe'/f'passt_{scenario}.json'))
    protocol = portable['protocol']
    document = json.loads(PLAN.read_text())
    for key, value in document['head_protocol'].items():
        if protocol[key] != value:
            raise ValueError('Historical PaSST control used a different head protocol')
    for key in ('split_seed', 'search_seed', 'selection_seeds', 'final_seeds'):
        if protocol[key] != document[key]:
            raise ValueError('Historical PaSST control used different seeds/partitions')
    local = json.loads((ROOT/portable['session']/'summary.json').read_text())
    for record in local['final_runs']['passt']:
        folder = ROOT/record['folder']
        if sha256(folder/'best_head.pt') != record['head_sha256'] or sha256(folder/'preprocessing.pt') != record['preprocessing_sha256']:
            raise ValueError('Original control head/scaler changed')
    return {'portable': portable, 'local': local, 'cache': cache}


def protected_artifacts():
    from src.data.transfer import sha256
    prior = json.loads((AUDIT/'no_ts9_final_integrity_20261003.json').read_text())
    files = {item['current_path']: item['sha256'] for item in prior['checks']}
    for folder in ('results/ast_transfer', 'results/audiomae_transfer'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file():
                files.setdefault(path.relative_to(ROOT).as_posix(), sha256(path))
    for name, expected in files.items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f'Protected historical artifact changed: {name}')
    return files


def verify_protected(files):
    from src.data.transfer import sha256
    for name, expected in files.items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f'Historical artifact overwritten: {name}')


def worker(args):
    import torch
    from src.training.temporal_experiment import extract, consolidate, PLAN_PATH
    torch.set_num_threads(4)
    if args.worker == 'extract':
        plan = json.loads(PLAN_PATH.read_text())
        extract(args.scenario, args.model, batch_size=plan['execution']['embedding_batch_sizes'][args.model])
    elif args.worker == 'train':
        consolidate(args.scenario, args.model, args.condition)
    elif args.worker == 'verify':
        from scripts.verify_temporal_interpolation import verify
        verify(args.scenario, args.model)
    elif args.worker == 'preflight':
        from scripts.preflight_temporal_interpolation import check
        check(args.model)


def run(*, authorized=False):
    if not authorized:
        raise RuntimeError('Pass --run explicitly; frozen probes only, no fine-tuning')
    import psutil
    from src.training.consolidation import write_json, utc_now
    AUDIT.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        saved = json.loads(LOCK.read_text())
        if coordinator_alive(saved):
            raise RuntimeError('A temporal experiment coordinator is already alive')
        LOCK.rename(LOCK.with_suffix('.stale_'+str(time.time_ns())))
    with LOCK.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'process_created_at': psutil.Process().create_time(), 'started_at_utc': utc_now()}, stream)
    state = json.loads(STATE.read_text()) if STATE.exists() else {'experiment': 'temporal_interpolation_v1', 'started_at_utc': utc_now(), 'jobs': {}}
    document = json.loads(PLAN.read_text())
    invocation_started = time.perf_counter()
    invocation = {'pid': os.getpid(), 'started_at_utc': utc_now(), 'status': 'running'}
    state.setdefault('invocations', []).append(invocation)
    try:
        if 'protected_artifacts' not in state:
            state['protected_artifacts'] = protected_artifacts()
        verify_protected(state['protected_artifacts'])
        state.update(status='preflight')
        write_json(STATE, state)
        for model in document['models']:
            subprocess.run([str(runtime(model)), '-X', 'utf8', str(Path(__file__).resolve()), '--worker', 'preflight', '--model', model], cwd=ROOT, check=True)
        for scenario in document['scenarios']:
            control = original_passt(scenario)
            state.setdefault('historical_passt_controls_preserved', {})[scenario] = {'session': control['portable']['session'], 'cache_sha256': control['cache']['sha256'], 'validated': True, 'used_for_paired_comparison': False}
            for model in ('passt', 'ast', 'audiomae'):
                key = scenario+'/'+model
                state.update(status='extracting', active_job=key)
                state['jobs'].setdefault(key, {}).setdefault('extraction_started_at_utc', utc_now())
                write_json(STATE, state)
                command = [str(runtime(model)), '-X', 'utf8', str(Path(__file__).resolve()), '--scenario', scenario, '--model', model]
                subprocess.run(command+['--worker', 'extract'], cwd=ROOT, check=True)
                for condition in ('control', 'bicubic'):
                    state.update(status='training', active_job=key+'/'+condition)
                    write_json(STATE, state)
                    subprocess.run(command+['--worker', 'train', '--condition', condition], cwd=ROOT, check=True)
                state.update(status='verifying', active_job=key)
                write_json(STATE, state)
                subprocess.run(command+['--worker', 'verify'], cwd=ROOT, check=True)
                state['jobs'][key].update(status='complete', completed_at_utc=utc_now())
                write_json(STATE, state)
        verify_protected(state['protected_artifacts'])
        state.update(status='complete', completed_at_utc=utc_now(), active_job=None, protected_hashes_unchanged=True)
        invocation['status'] = 'complete'
        write_json(STATE, state)
        print('TEMPORAL_ALL_COMPLETE', STATE, flush=True)
    except Exception as error:
        invocation['status'] = 'failed'
        state.update(status='failed', error=str(error), failed_at_utc=utc_now())
        write_json(STATE, state)
        raise
    finally:
        invocation.update(finished_at_utc=utc_now(), elapsed_wall_seconds=time.perf_counter()-invocation_started)
        write_json(STATE, state)
        if LOCK.exists() and json.loads(LOCK.read_text()).get('pid') == os.getpid():
            LOCK.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', choices=('extract', 'train', 'verify', 'preflight'))
    parser.add_argument('--scenario', choices=('mono_disc', 'mono_cont', 'poly_disc', 'poly_cont'))
    parser.add_argument('--model', choices=('ast', 'audiomae', 'passt'))
    parser.add_argument('--condition', choices=('control', 'bicubic'), default='bicubic')
    args = parser.parse_args()
    worker(args) if args.worker else run(authorized=args.run)
