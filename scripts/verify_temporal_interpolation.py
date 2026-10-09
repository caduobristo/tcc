"""Reload final heads, refit train-only preprocessing, and reproduce inputs/cache."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(scenario, model):
    import numpy as np
    import torch
    from torch import nn
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from src.data.archive import GuitarFxArchive
    from src.data.spectrogram_resize import adapt_spectrogram_time
    from src.data.temporal_inputs import reconcile_features, read_checked_feature
    from src.data.transfer import EFFECTS, GuitarWaveforms, read_manifest, sha256
    from src.training.consolidation import class_weights, evaluate, fit_preprocessing, write_json
    from src.training.temporal_experiment import folder_for, validate_new_cache, selection_data
    torch.set_num_threads(4)
    rows = read_manifest(scenario)
    test_indices = [i for i, row in enumerate(rows) if row['split'] == 'test']
    selected = [next(i for i in test_indices if rows[i]['effect'] == effect) for effect in EFFECTS]
    selected += [i for i in test_indices if i not in selected][:3]
    metadata, members, feature_checks = None, None, None
    if model != 'passt':
        _, members, feature_checks, metadata = reconcile_features(scenario)
    report = {'scenario': scenario, 'model': model, 'selected_test_inputs': selected,
              'test_classes_covered': sorted({rows[i]['effect'] for i in selected}), 'conditions': {}}
    for condition in ('control', 'bicubic'):
        folder = folder_for(scenario, model, condition)
        summary = json.loads((folder/'consolidation/summary.json').read_text())
        records, aggregate = summary['final_runs'], summary['aggregate']
        cache = validate_new_cache(scenario, model, condition, metadata)
        vector_path = folder/'embeddings.npy'
        batch_size = cache['embedding_batch_size']
        configuration = summary['selection']['candidate']
        array = np.load(vector_path, mmap_mode='r', allow_pickle=False)
        train = selection_data(scenario, model, cache, vector_path)
        mean, scale = fit_preprocessing(train.train_x, configuration['normalization'])
        weights = class_weights(train.train_y, configuration['class_weighting'], classes=len(EFFECTS))
        del train
        x = torch.from_numpy(array[test_indices].copy()).cuda()
        y = torch.tensor([int(rows[i]['label']) for i in test_indices], device='cuda')
        scores, run_checks = {}, []
        for record in records:
            run = ROOT/record['folder']
            assert sha256(run/'best_head.pt') == record['head_sha256']
            assert sha256(run/'preprocessing.pt') == record['preprocessing_sha256']
            prep = torch.load(run/'preprocessing.pt', map_location='cuda', weights_only=True)
            for key, expected in (('mean', mean), ('scale', scale), ('class_weights', weights)):
                assert torch.equal(prep[key], expected), f'Train-only preprocessing differs: {key}'
            head = nn.Linear(768, len(EFFECTS)).cuda().eval()
            head.load_state_dict(torch.load(run/'best_head.pt', map_location='cuda', weights_only=True), strict=True)
            recalculated, predicted = evaluate(head, (x-prep['mean'])/prep['scale'], y, classes=EFFECTS)
            with np.load(run/'test_predictions.npz', allow_pickle=False) as saved:
                assert np.array_equal(saved['indices'], test_indices)
                assert np.array_equal(saved['true'], y.cpu().numpy())
                assert np.array_equal(saved['predicted'], predicted)
            truth = y.cpu().numpy()
            independent = {'accuracy': accuracy_score(truth, predicted),
                'macro_f1': f1_score(truth, predicted, average='macro', zero_division=0),
                'macro_precision': precision_score(truth, predicted, average='macro', zero_division=0),
                'macro_recall': recall_score(truth, predicted, average='macro', zero_division=0)}
            for key, value in independent.items():
                assert abs(value-record['test_metrics'][key]) < 1e-12
                scores.setdefault(key, []).append(value)
            assert recalculated['confusion_matrix'] == record['test_metrics']['confusion_matrix']
            run_checks.append({'seed': record['training_seed'], 'head_sha256': record['head_sha256'],
                               'scaler_sha256': record['preprocessing_sha256'], 'exact_test_predictions': True})
        for key, values in scores.items():
            assert abs(np.mean(values)-aggregate[key]['mean']) < 1e-12
            assert abs(np.std(values, ddof=1)-aggregate[key]['std']) < 1e-12
        # Recompute entire original batches so kernel batching is identical.
        if model == 'passt':
            from src.models.transfer import load_frozen_encoder
            from src.models.temporal_resize import load_temporally_resized_passt
            from src.models.temporal_encoders import enable_sdpa
            encoder = enable_sdpa(load_frozen_encoder('passt') if condition == 'control' else load_temporally_resized_passt()).cuda().eval()
        else:
            from src.models.temporal_encoders import load_legacy_encoder
            encoder = load_legacy_encoder(model).cuda().eval()
        max_difference, checked_inputs = 0.0, 0
        recomputed = {}
        def input_batch(start):
            batch_rows = rows[start:start+batch_size]
            if model == 'passt':
                data = GuitarWaveforms(batch_rows)
                return torch.stack([data[i][0] for i in range(len(data))]).cuda()
            with GuitarFxArchive(scenario, 'mel16') as archive:
                raw = torch.from_numpy(np.stack([read_checked_feature(archive, members[i], feature_checks[members[i]])
                                                for i in range(start, start+len(batch_rows))])).cuda()
            normalized = (raw+4.27)/(4.57*2)
            value = adapt_spectrogram_time(normalized, 1024, mode='pad' if condition == 'control' else 'bicubic')
            return value.unsqueeze(1) if model == 'audiomae' else value
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in sorted({(i//batch_size)*batch_size for i in selected}):
                value = input_batch(start)
                actual = encoder(value).float().cpu().numpy()
                expected = np.asarray(array[start:start+len(actual)])
                difference = float(np.abs(actual-expected).max())
                max_difference = max(max_difference, difference)
                assert np.array_equal(actual, expected), f'Input/cache mismatch: {model}/{condition}, abs={difference}'
                checked_inputs += len(actual)
                for i in selected:
                    if start <= i < start+len(actual):
                        recomputed[i] = actual[i-start]
        # Compare logits from the selected raw inputs and from saved embeddings.
        raw_features = torch.from_numpy(np.stack([recomputed[i] for i in selected])).cuda()
        cached_features = torch.from_numpy(array[selected].copy()).cuda()
        for record in records:
            run = ROOT/record['folder']
            prep = torch.load(run/'preprocessing.pt', weights_only=True, map_location='cuda')
            head = nn.Linear(768, len(EFFECTS)).cuda().eval()
            head.load_state_dict(torch.load(run/'best_head.pt', weights_only=True, map_location='cuda'), strict=True)
            with torch.inference_mode():
                assert torch.equal(head((raw_features-prep['mean'])/prep['scale']), head((cached_features-prep['mean'])/prep['scale']))
        del encoder, x, y
        torch.cuda.empty_cache()
        report['conditions'][condition] = {'final_runs': run_checks, 'train_only_preprocessing_exact': True,
            'metrics_independently_recomputed_sklearn': True, 'aggregate_mean_sample_std_verified': True,
            'input_cache_batches_checked': checked_inputs, 'input_cache_max_absolute_error': max_difference,
            'selected_input_logits_exact_for_all_five_heads': True}
    write_json(ROOT/'data/audits/transfer_learning'/f'temporal_verified_{scenario}_{model}.json', report)
    print('TEMPORAL_VERIFIED', scenario, model, flush=True)
    return report
