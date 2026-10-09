"""Isolated, resumable caches and validation-only frozen linear probes."""
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.archive import GuitarFxArchive
from src.data.spectrogram_resize import adapt_spectrogram_time
from src.data.temporal_inputs import reconcile_features, read_checked_feature
from src.data.transfer import EFFECTS, GuitarWaveforms, manifest_path, read_manifest, sha256
from src.models.transfer import PROJECT_ROOT, WEIGHTS_ROOT, WEIGHTS, WEIGHT_SHA256
from src.training.consolidation import (SelectionData, candidate_grid, candidate_id, dataset_view,
    evaluate, fit_candidate, rank_finalists, utc_now, write_json)

ROOT = PROJECT_ROOT
OUTPUT = ROOT/'results/transfer_learning/ablations/temporal_interpolation_v1'
PLAN_PATH = ROOT/'configs/linear_probe/temporal_interpolation.json'


def folder_for(scenario, model, condition):
    if scenario not in ('mono_disc', 'mono_cont', 'poly_disc', 'poly_cont') or model not in ('ast', 'audiomae', 'passt') or condition not in ('control', 'bicubic'):
        raise ValueError('Unknown temporal experiment condition')
    return OUTPUT/scenario/model/condition


def cache_identity(scenario, model, condition, metadata=None):
    import timm
    sources = json.loads((ROOT/'checkpoints/pretrained_models/temporal_sources.json').read_text())
    files = ['src/training/temporal_experiment.py', 'src/data/spectrogram_resize.py', 'src/models/temporal_encoders.py']
    if model == 'passt':
        path = WEIGHTS_ROOT/WEIGHTS[model]
        checkpoint = WEIGHT_SHA256[model]
        files += ['src/models/temporal_resize.py', 'src/models/transfer.py', 'src/data/transfer.py']
    else:
        path = ROOT/sources[model]['path']
        checkpoint = sources[model]['sha256']
        files += ['src/data/temporal_inputs.py', 'src/data/archive.py', 'src/models/temporal_encoders.py', f'src/models/{model}_models.py']
    if sha256(path) != checkpoint:
        raise ValueError('Pretrained checkpoint checksum differs')
    return {'schema': 1, 'experiment': 'temporal_interpolation_v1', 'scenario': scenario,
            'model': model, 'condition': condition, 'manifest_sha256': sha256(manifest_path(scenario)),
            'classes': list(EFFECTS), 'checkpoint_sha256': checkpoint,
            'encoder_frozen': True, 'encoder_eval': True, 'encoder_precision': 'bfloat16',
            'embedding_dtype': 'float32', 'embedding_dim': 768,
            'temporal_adaptation': 'bicubic_align_corners_true' if condition == 'bicubic' else ('native_short' if model == 'passt' else 'pad_after_normalization'),
            'target_frames': (None if condition == 'control' else 998) if model == 'passt' else 1024, 'frequency_bands': 128,
            'feature_normalization': None if model == 'passt' else {'mean': -4.27, 'std': 4.57, 'divide_std_by': 'multiply denominator by 2'},
            'attention': 'same_pretrained_qkv_with_equivalent_pytorch_sdpa',
            'feature_metadata': metadata,
            'implementation': {f: sha256(ROOT/f) for f in files},
            'packages': {name: importlib.metadata.version(name) for name in ('torch', 'torchaudio', 'numpy', 'timm')},
            'gpu': torch.cuda.get_device_name(0), 'timm_version': timm.__version__}


def validate_new_cache(scenario, model, condition, metadata=None):
    folder = folder_for(scenario, model, condition)
    record = json.loads((folder/'embedding_cache.json').read_text())
    path = folder/'embeddings.npy'
    if not record['complete'] or record['identity'] != cache_identity(scenario, model, condition, metadata) or sha256(path) != record['sha256']:
        raise ValueError('Temporal cache is incomplete, stale or changed')
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.shape != (len(read_manifest(scenario)), 768) or array.dtype != np.float32:
        raise ValueError('Temporal cache shape/dtype mismatch')
    for start in range(0, len(array), 4096):
        if not np.isfinite(array[start:start+4096]).all():
            raise ValueError('Nonfinite temporal embeddings')
    return record


def extraction_buffers(scenario, model, conditions, count, metadata):
    buffers, identities, starts = {}, {}, {}
    for condition in conditions:
        folder = folder_for(scenario, model, condition)
        folder.mkdir(parents=True, exist_ok=True)
        identities[condition] = cache_identity(scenario, model, condition, metadata)
        if (folder/'embedding_cache.json').exists():
            validate_new_cache(scenario, model, condition, metadata)
            continue
        if (folder/'embeddings.npy').exists():
            raise ValueError('Complete vectors without sidecar; inspect before proceeding')
        partial, progress = folder/'embeddings.partial.npy', folder/'extraction_progress.json'
        if progress.exists():
            previous = json.loads(progress.read_text())
            if previous['identity'] != identities[condition] or not partial.exists():
                raise ValueError('Interrupted extraction identity differs')
            starts[condition] = previous['written']
            buffers[condition] = np.lib.format.open_memmap(partial, mode='r+')
            if buffers[condition].shape != (count, 768):
                raise ValueError('Partial cache shape changed')
        elif partial.exists():
            raise ValueError('Partial vectors without progress; inspect before retrying')
        else:
            starts[condition] = 0
            buffers[condition] = np.lib.format.open_memmap(partial, mode='w+', dtype='float32', shape=(count, 768))
            write_json(progress, {'identity': identities[condition], 'written': 0, 'elapsed_seconds': 0})
    return buffers, identities, starts


def extract(scenario, model, *, batch_size=32):
    if not torch.cuda.is_available():
        raise RuntimeError('This full experiment requires the prepared CUDA environment')
    torch.set_num_threads(4)
    metadata, members, checks = None, None, None
    if model != 'passt':
        rows, members, checks, metadata = reconcile_features(scenario)
    else:
        rows = read_manifest(scenario)
    conditions = ('control', 'bicubic')
    buffers, identities, starts = extraction_buffers(scenario, model, conditions, len(rows), metadata)
    if not buffers:
        return {c: validate_new_cache(scenario, model, c, metadata) for c in conditions}
    if model == 'passt':
        from src.models.transfer import load_frozen_encoder
        from src.models.temporal_encoders import enable_sdpa
        from src.models.temporal_resize import TemporallyResizedPasst
        encoder = enable_sdpa(load_frozen_encoder('passt')).cuda().eval()
        resized_encoder = TemporallyResizedPasst(encoder).cuda().eval()
        beginning = min(starts.values())
        dataset = GuitarWaveforms(rows[beginning:])
        batches = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
    else:
        from src.models.temporal_encoders import load_legacy_encoder
        encoder = load_legacy_encoder(model).cuda().eval()
        beginning = min(starts.values())
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError('Encoder is not frozen')
    started = time.perf_counter()
    accumulated = {c: json.loads((folder_for(scenario, model, c)/'extraction_progress.json').read_text())['elapsed_seconds'] for c in buffers}
    last_progress = started

    def write_batch(raw, start):
        nonlocal last_progress
        raw = raw.cuda(non_blocking=True).float()
        for condition, array in buffers.items():
            if start < starts[condition]:
                continue
            if model == 'passt':
                active_encoder = resized_encoder if condition == 'bicubic' else encoder
                inputs = raw
            else:
                normalized = (raw+4.27)/(4.57*2)
                inputs = adapt_spectrogram_time(normalized, 1024, mode='pad' if condition == 'control' else 'bicubic')
                if model == 'audiomae':
                    inputs = inputs.unsqueeze(1)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                values = (active_encoder if model == 'passt' else encoder)(inputs).float().cpu().numpy()
            if values.shape != (len(raw), 768) or not np.isfinite(values).all():
                raise ValueError('Invalid frozen encoder output')
            array[start:start+len(raw)] = values
        now = time.perf_counter()
        if now-last_progress > 20 or start+len(raw) == len(rows):
            for condition, array in buffers.items():
                array.flush()
                write_json(folder_for(scenario, model, condition)/'extraction_progress.json', {
                    'identity': identities[condition], 'written': max(starts[condition], start+len(raw)),
                    'elapsed_seconds': accumulated[condition]+now-started})
            print('EXTRACTION_PROGRESS', scenario, model, start+len(raw), len(rows), f'{now-started:.1f}s', flush=True)
            last_progress = now

    if model == 'passt':
        for wave, _, index in batches:
            write_batch(wave, beginning+int(index[0]))
    else:
        with GuitarFxArchive(scenario, 'mel16') as archive, ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(beginning, len(rows), batch_size):
                batch = members[start:start+batch_size]
                values = list(pool.map(lambda m: read_checked_feature(archive, m, checks[m]), batch))
                write_batch(torch.from_numpy(np.stack(values)), start)
    del encoder
    if model == 'passt':
        del resized_encoder
    torch.cuda.empty_cache()
    for condition in list(buffers):
        buffers[condition].flush()
        del buffers[condition]
        folder = folder_for(scenario, model, condition)
        path = folder/'embeddings.npy'
        (folder/'embeddings.partial.npy').rename(path)
        write_json(folder/'embedding_cache.json', {'identity': identities[condition], 'complete': True,
            'shape': [len(rows), 768], 'dtype': 'float32', 'samples': len(rows), 'sha256': sha256(path),
            'elapsed_seconds': accumulated[condition]+time.perf_counter()-started,
            'elapsed_scope': 'paired control+variant extraction including feature IO',
            'embedding_batch_size': batch_size, 'finished_at_utc': utc_now()})
    print('EXTRACTION_COMPLETE', scenario, model, flush=True)
    return {c: validate_new_cache(scenario, model, c, metadata) for c in conditions}


def plan_for(scenario, model, condition):
    document = json.loads(PLAN_PATH.read_text())
    return {'schema': 1, 'scenario': scenario, 'models': [model], 'condition': condition,
            'split_seed': document['split_seed'], 'search_seed': document['search_seed'],
            'selection_seeds': document['selection_seeds'], 'final_seeds': document['final_seeds'],
            **document['head_protocol'], 'train_encoder': False, 'test_used_for_selection': False,
            'historical_test_already_inspected': True}


def selection_data(scenario, model, cache, path):
    rows = read_manifest(scenario)
    _, view = dataset_view(rows, EFFECTS)
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    values = {}
    for split in ('train', 'validation'):
        indices = [i for i, row in enumerate(rows) if row['split'] == split]
        values[split+'_x'] = torch.from_numpy(array[indices].copy()).cuda()
        values[split+'_y'] = torch.tensor([int(rows[i]['label']) for i in indices], device='cuda')
    return SelectionData(**values, cache=cache, model=model, classes=tuple(EFFECTS), view=view)


def fit_or_reuse(data, candidate, seed, plan, folder):
    record_path = folder/'run.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if (record['status'] == 'complete' and record['candidate'] == candidate and record['training_seed'] == seed
                and record['cache_identity'] == data.cache['identity'] and record['embedding_sha256'] == data.cache['sha256']
                and record['head_sha256'] == sha256(folder/'best_head.pt')
                and record['preprocessing_sha256'] == sha256(folder/'preprocessing.pt')):
            record['folder'] = folder.relative_to(ROOT).as_posix()
            return record
        if record['status'] != 'complete':
            backup = folder.with_name(folder.name+'_interrupted_'+str(time.time_ns()))
            folder.rename(backup)  # preserve partial training; only this isolated experiment
        else:
            raise ValueError('Previously completed head changed; no overwrite permitted')
    record = fit_candidate(data, candidate, seed, plan, folder, allow_training=True)
    record['folder'] = folder.relative_to(ROOT).as_posix()
    return record


def consolidate(scenario, model, condition):
    torch.set_num_threads(4)
    plan = plan_for(scenario, model, condition)
    metadata = reconcile_features(scenario)[3] if model != 'passt' else None
    cache = validate_new_cache(scenario, model, condition, metadata)
    folder = folder_for(scenario, model, condition)
    vectors = folder/'embeddings.npy'
    session = folder/'consolidation'
    session.mkdir(exist_ok=True)
    snapshot_path = session/'protocol.json'
    immutable = {'plan': plan, 'plan_sha256': sha256(PLAN_PATH), 'cache_identity': cache['identity'],
                 'embedding_sha256': cache['sha256'], 'head_implementation_sha256': sha256(ROOT/'src/training/consolidation.py')}
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        if snapshot['immutable'] != immutable:
            raise ValueError('Protocol or inputs changed during a saved session')
        if snapshot['status'] == 'complete':
            return json.loads((session/'summary.json').read_text())
    else:
        snapshot = {'immutable': immutable, 'started_at_utc': utc_now(), 'status': 'selection'}
        write_json(snapshot_path, snapshot)
    started = time.perf_counter()
    data = selection_data(scenario, model, cache, vectors)
    search = []
    for candidate in candidate_grid(plan):
        ident = candidate_id(candidate)
        record = fit_or_reuse(data, candidate, plan['search_seed'], plan, session/'search'/ident/f'seed_{plan["search_seed"]}')
        search.append({'candidate_id': ident, 'candidate': candidate, 'runs': [record]})
        write_json(session/'search_results.json', rank_finalists(search))
    finalists = rank_finalists(search)[:plan['finalists_per_model']]
    for item in finalists:
        for seed in plan['selection_seeds']:
            if seed != plan['search_seed']:
                item['runs'].append(fit_or_reuse(data, item['candidate'], seed, plan, session/'selection'/item['candidate_id']/f'seed_{seed}'))
    selected = rank_finalists(finalists)[0]
    write_json(session/'finalist_ranking.json', finalists)
    lock_path = session/'selection_locked.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock['selected']['candidate'] != selected['candidate'] or lock['final_seeds'] != plan['final_seeds']:
            raise ValueError('Saved validation selection differs')
    else:
        lock = {'locked_at_utc': utc_now(), 'selected': selected, 'final_seeds': plan['final_seeds'],
                'test_used_for_selection': False, 'criterion': 'highest mean validation macro F1; ties by candidate ID'}
        write_json(lock_path, lock)
    finals = []
    for seed in plan['final_seeds']:
        finals.append(fit_or_reuse(data, selected['candidate'], seed, plan, session/'final'/f'seed_{seed}'))
    del data
    heads_lock = session/'all_confirmation_heads_locked.json'
    if heads_lock.exists():
        earlier = json.loads(heads_lock.read_text())
        keys = ('training_seed', 'head_sha256', 'preprocessing_sha256', 'folder')
        if [{k: r[k] for k in keys} for r in earlier] != [{k: r[k] for k in keys} for r in finals]:
            raise ValueError('Confirmation heads changed after the saved lock')
    else:
        write_json(heads_lock, finals)
    snapshot.update(status='final_evaluation')
    write_json(snapshot_path, snapshot)
    # Test vectors first become available only after all five heads are frozen.
    rows = read_manifest(scenario)
    indices = [i for i, row in enumerate(rows) if row['split'] == 'test']
    array = np.load(vectors, mmap_mode='r', allow_pickle=False)
    x = torch.from_numpy(array[indices].copy()).cuda()
    y = torch.tensor([int(rows[i]['label']) for i in indices], device='cuda')
    for record in finals:
        run_folder = ROOT/record['folder']
        prep = torch.load(run_folder/'preprocessing.pt', weights_only=True, map_location='cuda')
        head = nn.Linear(768, len(EFFECTS)).cuda().eval()
        head.load_state_dict(torch.load(run_folder/'best_head.pt', weights_only=True, map_location='cuda'), strict=True)
        metrics, prediction = evaluate(head, (x-prep['mean'])/prep['scale'], y, classes=EFFECTS)
        if record.get('test_evaluated'):
            saved = np.load(run_folder/'test_predictions.npz', allow_pickle=False)
            if not np.array_equal(saved['predicted'], prediction):
                raise ValueError('Completed test predictions changed')
        else:
            write_json(run_folder/'test_metrics.json', metrics)
            np.savez_compressed(run_folder/'test_predictions.npz', indices=np.asarray(indices), true=y.cpu().numpy(), predicted=prediction)
            original = json.loads((run_folder/'run.json').read_text())
            original.update(test_evaluated=True, test_evaluated_at_utc=utc_now(), selection_lock_sha256=sha256(lock_path))
            write_json(run_folder/'run.json', original)
        record.update(test_metrics=metrics, test_evaluated=True)
        print('TEMPORAL_FINAL', scenario, model, condition, record['training_seed'], metrics['accuracy'], metrics['macro_f1'], flush=True)
    summary = {'scenario': scenario, 'model': model, 'condition': condition,
               'session': session.relative_to(ROOT).as_posix(), 'protocol': plan,
               'selection': selected, 'cache': cache, 'final_runs': finals,
               'classes': list(EFFECTS), 'encoder_frozen': True, 'fine_tuning': False,
               'selection_lock_sha256': sha256(lock_path), 'all_confirmation_heads_sha256': sha256(session/'all_confirmation_heads_locked.json'),
               'completed_at_utc': utc_now(), 'elapsed_seconds_this_invocation': time.perf_counter()-started,
               'dataset_view': dataset_view(rows, EFFECTS)[1], 'aggregate': {}}
    for key in ('accuracy', 'macro_f1', 'macro_precision', 'macro_recall'):
        values = [record['test_metrics'][key] for record in finals]
        summary['aggregate'][key] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)), 'values': values}
    write_json(session/'summary.json', summary)
    snapshot.update(status='complete', completed_at_utc=utc_now())
    write_json(snapshot_path, snapshot)
    print('TEMPORAL_CONSOLIDATED', scenario, model, condition, flush=True)
    return summary
