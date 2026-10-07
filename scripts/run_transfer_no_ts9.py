"""Repeat the frozen-head protocol without TS9, reusing validated encoder caches."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCENARIOS = ('mono_disc', 'mono_cont', 'poly_disc', 'poly_cont')


def run(scenarios, *, allow_training=False):
    if not allow_training:
        raise RuntimeError('Head training requires --run')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from scripts.report_transfer_consolidation import report
    from src.data.paths import data_root
    from src.data.transfer import read_manifest, GuitarWaveforms, manifest_path, sha256
    from src.models.consolidated_probe import load_consolidated_probe
    from src.training.consolidation import (run_consolidation, write_json, utc_now,
        validate_plan, dataset_view, classes_for_plan)
    from src.training.linear_probe import load_config, validate_cache, paths_for

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('The validated BF16 embedding recipe requires CUDA')
    audit = data_root() / 'audits/transfer_learning'
    audit.mkdir(parents=True, exist_ok=True)
    lock_path = audit / 'no_ts9_execution.lock'
    # A stale lock is preserved for inspection rather than silently removed.
    with lock_path.open('x', encoding='utf-8') as lock_file:
        json.dump({'pid': os.getpid(), 'started_at_utc': utc_now()}, lock_file)
    state_path = audit / 'no_ts9_execution.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
        'started_at_utc': utc_now(), 'excluded_effects': ['TS9'], 'fine_tuning': False, 'scenarios': {}}
    try:
        for scenario in scenarios:
            plan = json.loads((ROOT / f'configs/linear_probe/consolidation_no_ts9_{scenario}.json').read_text())
            validate_plan(plan)
            if plan['excluded_effects'] != ['TS9']:
                raise ValueError('This runner excludes TS9 only')
            _, view = dataset_view(read_manifest(scenario), classes_for_plan(plan))
            print('DATA_VIEW', scenario, json.dumps(view), flush=True)
            for model in plan['models']:
                cache = validate_cache(load_config(ROOT / f'configs/linear_probe/{model}_{scenario}.json'))
                print('CACHE_VERIFIED', scenario, model, cache['sha256'], flush=True)
        for scenario in scenarios:
            started = time.perf_counter()
            item = state['scenarios'].setdefault(scenario, {'started_at_utc': utc_now()})
            state.update(status='running', active_scenario=scenario)
            write_json(state_path, state)
            if 'session' in item:
                session = ROOT / item['session']
                if json.loads((session / 'protocol.json').read_text())['status'] != 'complete':
                    raise RuntimeError(f'Incomplete session preserved; inspect before resuming: {session}')
                print('REUSE_COMPLETED_TRAINING', scenario, session, flush=True)
            else:
                item['status'] = 'training'
                write_json(state_path, state)
                def remember(session):
                    item['session'] = session.relative_to(ROOT).as_posix()
                    write_json(state_path, state)
                session, _ = run_consolidation(ROOT / f'configs/linear_probe/consolidation_no_ts9_{scenario}.json',
                    allow_training=True, on_session_started=remember)
            item['status'] = 'verification'
            write_json(state_path, state)
            summary = json.loads((session / 'summary.json').read_text(encoding='utf-8'))
            rows = read_manifest(scenario)
            retained, view = dataset_view(rows, classes_for_plan(summary['protocol']))
            checks = {}
            for model in summary['protocol']['models']:
                chosen = max(summary['final_runs'][model], key=lambda r: (r['best_validation_macro_f1'], -r['training_seed']))
                probe = load_consolidated_probe(ROOT / chosen['folder']).eval()
                selected_rows = [rows[i] for i in retained[:16]]
                waveform, _, _ = next(iter(DataLoader(GuitarWaveforms(selected_rows), batch_size=16, num_workers=0)))
                config = load_config(ROOT / f'configs/linear_probe/{model}_{scenario}.json')
                _, vectors, _ = paths_for(config)
                cached = np.load(vectors, mmap_mode='r', allow_pickle=False)
                with torch.inference_mode():
                    actual = probe(waveform.cuda())
                    expected = probe.forward_embeddings(torch.from_numpy(cached[retained[:16]].copy()).cuda())
                if not torch.equal(actual, expected) or not bool(torch.isfinite(actual).all()):
                    raise ValueError('Waveform inference differs from the original frozen embedding cache')
                checks[model] = {'samples': 16, 'classes': list(probe.classes), 'exact_match': True,
                                 'max_logit_difference': float((actual-expected).abs().max()),
                                 'representative_seed': chosen['training_seed']}
                del probe, actual, expected, cached
                torch.cuda.empty_cache()
            write_json(session / 'waveform_inference_check.json', checks)
            public = report(session)
            item.update(status='complete', completed_at_utc=utc_now(), dataset_view=view,
                        waveform_inference_check=checks, all_predictions_reproduced=public['all_predictions_reproduced'],
                        original_manifest_sha256=sha256(manifest_path(scenario)),
                        execution_seconds=time.perf_counter()-started)
            write_json(state_path, state)
            print('NO_TS9_SCENARIO_COMPLETE', scenario, item['session'], flush=True)
        state.update(status='complete', active_scenario=None, completed_at_utc=utc_now())
        write_json(state_path, state)
        return state
    except Exception as error:
        state.update(status='failed', error=repr(error), failed_at_utc=utc_now())
        write_json(state_path, state)
        raise
    finally:
        lock_path.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    args = parser.parse_args()
    if not args.run:
        print('Plan only: exclude TS9 from all splits, preserve source partitions, reuse original frozen caches, '
              'train 34 new 12-class heads/scenario, verify and report. Use --run to execute.')
    else:
        run(args.scenarios, allow_training=True)
