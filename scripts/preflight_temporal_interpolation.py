"""Check pretrained encoders, attention equivalence and short batch timing."""
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(model):
    import numpy as np
    import torch
    from src.data.archive import GuitarFxArchive
    from src.data.spectrogram_resize import adapt_spectrogram_time
    from src.data.temporal_inputs import reconcile_features, read_checked_feature
    from src.data.transfer import GuitarWaveforms, read_manifest
    from src.models.temporal_encoders import enable_sdpa, load_legacy_encoder
    from src.training.consolidation import utc_now, write_json
    from src.training.temporal_experiment import PLAN_PATH
    torch.set_num_threads(4)
    document = json.loads(PLAN_PATH.read_text())
    batch_size = document['execution']['embedding_batch_sizes'][model]
    report = {'model': model, 'checked_at_utc': utc_now(), 'encoder_frozen': True,
              'full_extraction_started': False, 'training_started': False,
              'gpu': torch.cuda.get_device_name(0), 'batch_size': batch_size, 'conditions': {}}
    if model == 'passt':
        from src.models.transfer import load_frozen_encoder
        from src.models.temporal_resize import TemporallyResizedPasst
        encoder = load_frozen_encoder('passt').cuda().eval()
        rows = [row for row in read_manifest('mono_disc') if row['split'] == 'train'][:batch_size]
        data = GuitarWaveforms(rows)
        raw = torch.stack([data[i][0] for i in range(len(rows))]).cuda()
        with torch.inference_mode():
            native = encoder(raw[:2]).float().cpu()
            enable_sdpa(encoder)
            efficient = encoder(raw[:2]).float().cpu()
        torch.testing.assert_close(native, efficient, rtol=1e-4, atol=1e-4)
        report['attention_float32_equivalence_max_absolute_error'] = float((native-efficient).abs().max())
        inputs = {'control': raw, 'bicubic': raw}
        long_encoder = TemporallyResizedPasst(encoder).cuda().eval()
    else:
        rows, members, checks, report['feature_reconciliation'] = reconcile_features('mono_disc')
        indices = [i for i, row in enumerate(rows) if row['split'] == 'train'][:batch_size]
        with GuitarFxArchive('mono_disc', 'mel16') as archive:
            raw = torch.from_numpy(np.stack([read_checked_feature(archive, members[i], checks[members[i]]) for i in indices])).cuda()
        normalized = (raw+4.27)/(4.57*2)
        inputs = {condition: adapt_spectrogram_time(normalized, 1024, mode=mode)
                  for condition, mode in (('control', 'pad'), ('bicubic', 'bicubic'))}
        if model == 'audiomae':
            inputs = {key: value.unsqueeze(1) for key, value in inputs.items()}
        encoder = load_legacy_encoder(model, use_sdpa=False).cuda().eval()
        small = inputs['control'][:2]
        with torch.inference_mode():
            native = encoder(small).float().cpu()
            enable_sdpa(encoder)
            efficient = encoder(small).float().cpu()
        torch.testing.assert_close(native, efficient, rtol=1e-4, atol=1e-4)
        report['attention_float32_equivalence_max_absolute_error'] = float((native-efficient).abs().max())
        report['strict_backbone_load'] = True
    assert all(not p.requires_grad for p in encoder.parameters())
    for condition, value in inputs.items():
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            active = long_encoder if model == 'passt' and condition == 'bicubic' else encoder
            active(value)  # warm up kernels
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(3):
                features = active(value)
            torch.cuda.synchronize()
        elapsed = (time.perf_counter()-started)/3
        assert features.shape == (batch_size, 768) and torch.isfinite(features).all()
        report['conditions'][condition] = {'batch_forward_seconds': elapsed,
            'samples_per_second_compute_only': batch_size/elapsed,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'output_shape': list(features.shape), 'finite': True}
    write_json(ROOT/'data/audits/transfer_learning'/f'temporal_preflight_{model}.json', report)
    print('TEMPORAL_PREFLIGHT', json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    check(sys.argv[1])
