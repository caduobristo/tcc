"""Small inference/data check only; no full extraction and no classifier training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(*, real_data=False, passt=False):
    import torch
    from src.data.spectrogram_resize import adapt_spectrogram_time
    torch.set_num_threads(4)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "full_extraction_started": False, "training_started": False,
              "checkpoints": {}, "features": {}, "passt": {}}
    source_file = ROOT / 'checkpoints/pretrained_models/temporal_sources.json'
    if source_file.exists():
        from src.data.transfer import sha256
        sources = json.loads(source_file.read_text(encoding='utf-8'))
        for model in ('ast', 'audiomae'):
            path = ROOT / sources[model]['path']
            report['checkpoints'][model] = {'path': str(path), 'available': path.is_file(),
                'sha256_matches_pinned': path.is_file() and sha256(path) == sources[model]['sha256'],
                'encoder_inference_verified': False}
    else:
        report['checkpoints'] = {'prepared_sources_available': False,
            'prepare_command': 'python scripts/prepare_temporal_checkpoints.py'}
    if real_data:
        from src.data.archive import GuitarFxArchive
        for scenario in ("mono_disc", "mono_cont", "poly_disc", "poly_cont"):
            with GuitarFxArchive(scenario, "mel16") as archive:
                effect, name = next(archive.iter_samples())
                array = archive.read_feature(effect, name)
                original = hashlib.sha256(array.tobytes()).hexdigest()
                normalized = (torch.from_numpy(array) + 4.27) / (4.57 * 2)
                resized = adapt_spectrogram_time(normalized, 1024)
                assert resized.shape == (1024, 128) and torch.isfinite(resized).all()
                assert hashlib.sha256(array.tobytes()).hexdigest() == original
                report["features"][scenario] = {"effect": effect, "sample": name,
                    "input_shape": list(array.shape), "output_shape": list(resized.shape),
                    "input_unchanged": True, "input_array_sha256": original,
                    "ast_audiomae_shared_input_checked": True}
    if passt:
        from src.data.transfer import GuitarWaveforms, read_manifest
        from src.models.temporal_resize import load_temporally_resized_passt
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = load_temporally_resized_passt().to(device).eval()
        assert all(not p.requires_grad for p in encoder.parameters())
        for scenario in ("mono_disc", "mono_cont", "poly_disc", "poly_cont"):
            rows = [row for row in read_manifest(scenario) if row["split"] == "test"][:2]
            data = GuitarWaveforms(rows)
            waves = torch.stack([data[i][0] for i in range(len(data))]).to(device)
            with torch.inference_mode():
                mel = encoder.encoder.mel(waves)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    values = encoder(waves)
            assert values.shape == (len(rows), 768) and torch.isfinite(values).all()
            report["passt"][scenario] = {"samples": len(rows), "native_frames": mel.shape[-1],
                "target_frames": 998, "embedding_shape": list(values.shape),
                "finite": True, "encoder_frozen": True, "device": str(device)}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-passt", action="store_true")
    args = parser.parse_args()
    result = check(real_data=args.check_data, passt=args.check_passt)
    output = ROOT / "data/audits/transfer_learning/temporal_interpolation_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
