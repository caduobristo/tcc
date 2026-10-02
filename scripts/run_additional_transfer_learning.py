"""Prepare and consolidate the three remaining scenarios; explicit --run only.

Each stage is saved before proceeding. A completed session is never retrained
to consult its results. Source notebooks remain disabled by default.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')


def now():
    return datetime.now(timezone.utc).isoformat()


def run(scenarios, *, allow_training=False, prepare_data=False):
    if not allow_training:
        raise RuntimeError('Use --run to authorize extraction and head training')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from scripts.prepare_additional_transfer_data import prepare
    from scripts.report_transfer_consolidation import report
    from src.data.paths import data_root, SCENARIOS
    from src.data.transfer import build_manifest, manifest_path, sha256, read_manifest, GuitarWaveforms
    from src.data.transfer_metadata import ensure_validated_metadata
    from src.models.consolidated_probe import load_consolidated_probe
    from src.training.linear_probe import extract_embeddings, load_config, paths_for, validate_cache
    from src.training.consolidation import run_consolidation, write_json

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    state_path = data_root()/'audits/transfer_learning/additional_scenarios_execution.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
        'started_at_utc':now(), 'fine_tuning':False, 'scenarios':{}}
    for scenario in scenarios:
        started = time.perf_counter()
        item = state['scenarios'].setdefault(scenario, {'started_at_utc':now()})
        state['active_scenario'] = scenario
        if item.get('status')=='complete':
            print('REUSE_COMPLETED_SCENARIO',scenario,item['session'],flush=True)
            continue
        item['status'] = 'data_preparation'
        write_json(state_path,state)
        receipt = data_root()/'archives/guitar_fx_dist/official'/SCENARIOS[scenario]/'selected_wavs_receipt.json'
        if not receipt.exists():
            if not prepare_data:
                raise FileNotFoundError('WAV preparation required; use --prepare-data')
            prepare(scenario,workers=2,discard_downloads=True)
        item['data_preparation_receipt'] = str(receipt.relative_to(data_root()))
        item['metadata_alignment'] = ensure_validated_metadata(scenario)
        if not manifest_path(scenario).exists():
            build_manifest(scenario)
        index = json.loads(manifest_path(scenario).with_suffix('.json').read_text(encoding='utf-8'))
        if sha256(manifest_path(scenario))!=index['manifest_sha256'] or not index['wav_integrity_ok'] or index['identical_wav_leakage']:
            raise ValueError('Manifest integrity or split leakage failure')
        item['manifest'] = index
        item['status'] = 'extraction'
        write_json(state_path,state)
        for model in ('passt','htsat'):
            config = load_config(ROOT/'configs/linear_probe'/f'{model}_{scenario}.json')
            extract_embeddings(config,allow_extraction=True)
            cache = validate_cache(config)
            item.setdefault('embedding_caches',{})[model] = {
                'sha256':cache['sha256'],'samples':cache['samples'],'elapsed_seconds':cache['elapsed_seconds']}
            write_json(state_path,state)
        item['status'] = 'consolidation'
        write_json(state_path,state)
        if 'session' in item:
            session = ROOT/item['session']
            if json.loads((session/'protocol.json').read_text())['status']!='complete':
                raise RuntimeError('Incomplete session preserved: inspect before resuming training')
        else:
            def remember_session(session):
                item['session'] = session.relative_to(ROOT).as_posix()
                write_json(state_path,state)
            session,_ = run_consolidation(ROOT/'configs/linear_probe'/f'consolidation_{scenario}.json',
                allow_training=True,on_session_started=remember_session)
            item['session'] = session.relative_to(ROOT).as_posix()
            write_json(state_path,state)
        # Confirm the waveform encoder/head/scaler path against cached vectors.
        summary = json.loads((session/'summary.json').read_text(encoding='utf-8'))
        rows = read_manifest(scenario)
        checks = {}
        for model in ('passt','htsat'):
            representative = max(summary['final_runs'][model],key=lambda r:(r['best_validation_macro_f1'],-r['training_seed']))
            probe = load_consolidated_probe(ROOT/representative['folder']).eval()
            waveform,_,indices = next(iter(DataLoader(GuitarWaveforms(rows[:16]),batch_size=16,num_workers=0)))
            config = load_config(ROOT/'configs/linear_probe'/f'{model}_{scenario}.json')
            _,vectors,_ = paths_for(config)
            cached = np.load(vectors,mmap_mode='r',allow_pickle=False)
            with torch.inference_mode():
                actual = probe(waveform.cuda())
                expected = probe.forward_embeddings(torch.from_numpy(cached[:16].copy()).cuda())
            if not torch.equal(actual,expected) or not bool(torch.isfinite(actual).all()):
                raise ValueError('Waveform and embedding predictions differ')
            checks[model] = {'samples':16,'max_logit_difference':float((actual-expected).abs().max()),
                             'representative_seed':representative['training_seed'],'exact_match':True}
            del probe,actual,expected,cached
            torch.cuda.empty_cache()
        write_json(session/'waveform_inference_check.json',checks)
        report(session)
        item.update(status='complete',completed_at_utc=now(),execution_seconds_excluding_prior_completed_stages=time.perf_counter()-started,
                    waveform_inference_check=checks)
        write_json(state_path,state)
        print('SCENARIO_COMPLETE',scenario,item['session'],flush=True)
    state.update(status='complete',completed_at_utc=now(),active_scenario=None)
    write_json(state_path,state)
    return state


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--prepare-data',action='store_true')
    parser.add_argument('--scenarios',nargs='+',choices=['mono_cont','poly_cont','poly_disc'],default=['mono_cont','poly_disc','poly_cont'])
    args = parser.parse_args()
    if not args.run:
        print('Plan: source WAV preparation, two native frozen caches, 34 heads/scenario, five final seeds/model, verification/report. No training started.')
    else:
        run(args.scenarios,allow_training=True,prepare_data=args.prepare_data)
