import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT.parents[1]))
from src.data.paths import guitar_fx_root

GUITARFX_ROOT = guitar_fx_root("raw")
from urllib.request import urlopen

import numpy as np
import soundfile as sf
import torch
import torchaudio


AUDIOMAE_MEAN = -4.2677393
AUDIOMAE_STD = 4.5689974
AUDIOMAE_TARGET_LENGTH = 1024
AUDIOMAE_MEL_BINS = 128
AUDIOSET_LABEL_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"


def parse_args():
    parser = argparse.ArgumentParser("Run AudioMAE AudioSet checkpoint on GUITAR-FX WAVs")
    parser.add_argument("--root", default=str(EXPERIMENT_ROOT))
    parser.add_argument(
        "--dataset-audio",
        default=str(GUITARFX_ROOT / "Mono_Continuous" / "Audio"),
    )
    parser.add_argument("--checkpoint", default=str(EXPERIMENT_ROOT / "ckpt" / "finetuned.pth"))
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENT_ROOT / "outputs" / "guitarfx_audiomae_mono_continuous_sample"),
    )
    parser.add_argument("--sample-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-nofx", action="store_true", default=True)
    return parser.parse_args()


def load_audioset_labels(root):
    label_path = Path(root) / "data" / "audioset" / "class_labels_indices.csv"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not label_path.exists():
        with urlopen(AUDIOSET_LABEL_URL, timeout=30) as response:
            label_path.write_bytes(response.read())

    labels = {}
    with label_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels[int(row["index"])] = {
                "mid": row["mid"],
                "display_name": row["display_name"],
            }
    return labels, label_path


def discover_wavs(dataset_audio, sample_per_class, include_nofx, seed):
    rng = np.random.default_rng(seed)
    dataset_audio = Path(dataset_audio)
    class_dirs = sorted(p for p in dataset_audio.iterdir() if p.is_dir() and not p.name.startswith("_NoFX"))
    if include_nofx:
        nofx_dirs = [p for p in dataset_audio.iterdir() if p.is_dir() and p.name.endswith("_preprocessed")]
        class_dirs.extend(sorted(nofx_dirs))

    rows = []
    for class_dir in class_dirs:
        wavs = sorted(class_dir.glob("*.wav"))
        if not wavs:
            continue
        if sample_per_class > 0 and len(wavs) > sample_per_class:
            idx = np.sort(rng.choice(len(wavs), size=sample_per_class, replace=False))
            wavs = [wavs[i] for i in idx]
        label = "NoFX" if class_dir.name.startswith("_NoFX") else class_dir.name
        for wav in wavs:
            rows.append({"source_label": label, "wav": wav})
    return rows


def wav_to_audiomae_fbank(path):
    waveform_np, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=1)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform - waveform.mean(),
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=AUDIOMAE_MEL_BINS,
        dither=0.0,
        frame_shift=10,
    )
    frames_before_pad = int(fbank.shape[0])
    if fbank.shape[0] < AUDIOMAE_TARGET_LENGTH:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, AUDIOMAE_TARGET_LENGTH - fbank.shape[0]))(fbank)
    else:
        fbank = fbank[:AUDIOMAE_TARGET_LENGTH]
    fbank = (fbank - AUDIOMAE_MEAN) / (AUDIOMAE_STD * 2)
    return fbank.unsqueeze(0), frames_before_pad


def build_model(root, checkpoint):
    repo = Path(root) / "repo"
    sys.path.insert(0, str(repo))

    import models_vit
    import util.misc as misc
    from main_finetune_as import PatchEmbed_new

    model = models_vit.vit_base_patch16(
        num_classes=527,
        drop_path_rate=0.1,
        global_pool=True,
        mask_2d=True,
        use_custom_patch=False,
    )
    model.patch_embed = PatchEmbed_new(
        img_size=(AUDIOMAE_TARGET_LENGTH, AUDIOMAE_MEL_BINS),
        patch_size=(16, 16),
        in_chans=1,
        embed_dim=768,
        stride=16,
    )
    model.pos_embed = torch.nn.Parameter(
        torch.zeros(1, model.patch_embed.num_patches + 1, 768),
        requires_grad=False,
    )
    ckpt = misc.torch_load_compat(str(checkpoint), map_location="cpu")
    msg = model.load_state_dict(ckpt["model"], strict=False)
    return model, ckpt, msg


def run():
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, label_path = load_audioset_labels(root)
    rows = discover_wavs(args.dataset_audio, args.sample_per_class, args.include_nofx, args.seed)
    if not rows:
        raise RuntimeError(f"No WAV files found under {args.dataset_audio}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, load_msg = build_model(root, Path(args.checkpoint))
    model = model.to(device).eval()

    all_logits = []
    all_probs = []
    result_rows = []

    batch_samples = []
    batch_meta = []
    for item in rows:
        fbank, frames_before_pad = wav_to_audiomae_fbank(item["wav"])
        batch_samples.append(fbank)
        batch_meta.append({**item, "frames_before_pad": frames_before_pad})

        if len(batch_samples) == args.batch_size:
            flush_batch(model, device, batch_samples, batch_meta, labels, all_logits, all_probs, result_rows)
            batch_samples = []
            batch_meta = []

    if batch_samples:
        flush_batch(model, device, batch_samples, batch_meta, labels, all_logits, all_probs, result_rows)

    logits = np.concatenate(all_logits, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    source_labels = np.array([row["source_label"] for row in result_rows], dtype=object)
    wav_paths = np.array([row["wav"] for row in result_rows], dtype=object)

    np.savez_compressed(
        output_dir / "audiomae_logits_probs_paths.npz",
        logits=logits,
        probs=probs,
        source_labels=source_labels,
        wav_paths=wav_paths,
    )

    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source_label",
            "wav",
            "frames_before_pad",
            "top1_index",
            "top1_mid",
            "top1_name",
            "top1_logit",
            "top1_prob",
            "top5",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    summary_rows = build_summary(result_rows)
    with (output_dir / "summary_by_guitarfx_label.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["source_label", "n", "top1_name", "count", "fraction"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    config = {
        "dataset_audio": str(Path(args.dataset_audio)),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": ckpt.get("epoch"),
        "sample_per_class": args.sample_per_class,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_files": len(result_rows),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "audioset_label_csv": str(label_path),
        "missing_keys": list(load_msg.missing_keys),
        "unexpected_keys": list(load_msg.unexpected_keys),
        "note": "The checkpoint predicts AudioSet classes, not GUITAR-FX pedal classes.",
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(json.dumps(config, indent=2))
    print(f"Saved predictions to {output_dir / 'predictions.csv'}")
    print(f"Saved summary to {output_dir / 'summary_by_guitarfx_label.csv'}")
    print(f"Saved logits/probabilities to {output_dir / 'audiomae_logits_probs_paths.npz'}")


def flush_batch(model, device, batch_samples, batch_meta, labels, all_logits, all_probs, result_rows):
    sample = torch.stack(batch_samples, dim=0).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(sample)
        else:
            logits = model(sample)
    logits_np = logits.float().cpu().numpy()
    probs_np = torch.sigmoid(logits.float()).cpu().numpy()
    all_logits.append(logits_np)
    all_probs.append(probs_np)

    topk = np.argsort(-logits_np, axis=1)[:, :5]
    for i, meta in enumerate(batch_meta):
        top_items = []
        for idx in topk[i]:
            label = labels[int(idx)]
            top_items.append(
                {
                    "index": int(idx),
                    "mid": label["mid"],
                    "name": label["display_name"],
                    "logit": float(logits_np[i, idx]),
                    "prob": float(probs_np[i, idx]),
                }
            )
        top1 = top_items[0]
        result_rows.append(
            {
                "source_label": meta["source_label"],
                "wav": str(meta["wav"]),
                "frames_before_pad": meta["frames_before_pad"],
                "top1_index": top1["index"],
                "top1_mid": top1["mid"],
                "top1_name": top1["name"],
                "top1_logit": f"{top1['logit']:.6f}",
                "top1_prob": f"{top1['prob']:.6f}",
                "top5": json.dumps(top_items, ensure_ascii=False),
            }
        )


def build_summary(result_rows):
    by_label = defaultdict(Counter)
    for row in result_rows:
        by_label[row["source_label"]][row["top1_name"]] += 1

    summary_rows = []
    for source_label in sorted(by_label):
        total = sum(by_label[source_label].values())
        for name, count in by_label[source_label].most_common(10):
            summary_rows.append(
                {
                    "source_label": source_label,
                    "n": total,
                    "top1_name": name,
                    "count": count,
                    "fraction": f"{count / total:.6f}",
                }
            )
    return summary_rows


if __name__ == "__main__":
    run()
