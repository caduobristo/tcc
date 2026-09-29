import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
GUITARFX_ROOT = EXPERIMENT_ROOT.parent / "fxnet_reproduction" / "data" / "GUITAR-FX"
from urllib.request import urlopen

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


AUDIOMAE_MEAN = -4.2677393
AUDIOMAE_STD = 4.5689974
AUDIOMAE_TARGET_LENGTH = 1024
AUDIOMAE_MEL_BINS = 128
LABEL_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
DATASETS = ("Mono_Continuous", "Mono_Discrete", "Poly_Continuous", "Poly_Discrete")
INTERESTING_AUDIOSET_CLASSES = (
    "Music",
    "Musical instrument",
    "Guitar",
    "Bass guitar",
    "Distortion",
    "Effects unit",
    "Electronic tuner",
    "Noise",
    "Synthesizer",
)


class GuitarFxDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        fbank, frames_before_pad = wav_to_fbank(entry["path"])
        return {
            "sample": fbank,
            "source_label": entry["source_label"],
            "path": str(entry["path"]),
            "frames_before_pad": frames_before_pad,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--guitarfx-root", default=str(GUITARFX_ROOT))
    parser.add_argument("--checkpoint", default=str(EXPERIMENT_ROOT / "ckpt" / "finetuned.pth"))
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENT_ROOT / "outputs" / "guitarfx_audiomae_all_datasets_fxnet_comparison"),
    )
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-per-class", type=int, default=0, help="0 means all files")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-label", action="append", default=["MT2"])
    parser.add_argument("--save-predictions", action="store_true", default=True)
    return parser.parse_args()


def load_audioset_labels(root):
    label_path = Path(root) / "data" / "audioset" / "class_labels_indices.csv"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not label_path.exists():
        with urlopen(LABEL_URL, timeout=30) as response:
            label_path.write_bytes(response.read())

    by_index = {}
    by_name = {}
    with label_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["index"])
            item = {"index": idx, "mid": row["mid"], "display_name": row["display_name"]}
            by_index[idx] = item
            by_name[row["display_name"]] = item
    return by_index, by_name, label_path


def discover_entries(guitarfx_root, dataset_name, excluded, sample_per_class, seed):
    rng = np.random.default_rng(seed)
    audio_dir = Path(guitarfx_root) / dataset_name / "Audio"
    if not audio_dir.exists():
        raise FileNotFoundError(audio_dir)

    entries = []
    for class_dir in sorted(p for p in audio_dir.iterdir() if p.is_dir()):
        if class_dir.name.startswith("_NoFX") or class_dir.name in excluded:
            continue
        wavs = sorted(class_dir.glob("*.wav"))
        if sample_per_class > 0 and len(wavs) > sample_per_class:
            chosen = np.sort(rng.choice(len(wavs), size=sample_per_class, replace=False))
            wavs = [wavs[i] for i in chosen]
        entries.extend({"source_label": class_dir.name, "path": wav} for wav in wavs)
    return entries


def wav_to_fbank(path):
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


def collate_batch(items):
    return {
        "sample": torch.stack([item["sample"] for item in items], dim=0),
        "source_label": [item["source_label"] for item in items],
        "path": [item["path"] for item in items],
        "frames_before_pad": [item["frames_before_pad"] for item in items],
    }


def main():
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    audioset_labels, audioset_by_name, label_csv = load_audioset_labels(root)
    interesting = {
        name: audioset_by_name[name]["index"]
        for name in INTERESTING_AUDIOSET_CLASSES
        if name in audioset_by_name
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, load_msg = build_model(root, Path(args.checkpoint))
    model = model.to(device).eval()

    run_config = {
        "datasets": DATASETS,
        "guitarfx_root": str(Path(args.guitarfx_root)),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": ckpt.get("epoch"),
        "excluded_labels": args.exclude_label,
        "sample_per_class": args.sample_per_class,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "audioset_label_csv": str(label_csv),
        "missing_keys": list(load_msg.missing_keys),
        "unexpected_keys": list(load_msg.unexpected_keys),
        "note": "AudioMAE checkpoint predicts AudioSet classes; FXNet reports predict GUITAR-FX pedal classes.",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(run_config, indent=2) + "\n")

    combined_rows = []
    for dataset_name in DATASETS:
        start = time.time()
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        entries = discover_entries(
            args.guitarfx_root,
            dataset_name,
            set(args.exclude_label),
            args.sample_per_class,
            args.seed,
        )
        run_dataset(
            dataset_name,
            entries,
            dataset_dir,
            model,
            device,
            audioset_labels,
            interesting,
            args,
            combined_rows,
        )
        elapsed = time.time() - start
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"{dataset_name}: {len(entries)} files, {elapsed:.2f}s\n")

    with (output_dir / "combined_dataset_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "n",
                "top1_name",
                "top1_count",
                "top1_fraction",
                "music_top1_fraction",
                "distortion_top1_fraction",
                "distortion_top5_fraction",
                "guitar_top5_fraction",
                "effects_unit_top5_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"Saved outputs under {output_dir}")


def run_dataset(dataset_name, entries, output_dir, model, device, audioset_labels, interesting, args, combined_rows):
    ds = GuitarFxDataset(entries)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    top1_counts = Counter()
    top1_by_source = defaultdict(Counter)
    top5_by_source = defaultdict(Counter)
    top5_counts = Counter()
    interesting_sums = {
        name: {"logit": 0.0, "prob": 0.0, "top5": 0}
        for name in interesting
    }
    interesting_by_source = defaultdict(
        lambda: {name: {"logit": 0.0, "prob": 0.0, "top5": 0, "n": 0} for name in interesting}
    )

    pred_file = None
    pred_writer = None
    if args.save_predictions:
        pred_file = (output_dir / "predictions_top5.csv").open("w", encoding="utf-8", newline="")
        pred_writer = csv.DictWriter(
            pred_file,
            fieldnames=[
                "dataset",
                "source_label",
                "wav",
                "frames_before_pad",
                "top1_index",
                "top1_name",
                "top1_logit",
                "top1_prob",
                "top5_names",
                "top5_indices",
                "top5_logits",
                "top5_probs",
            ],
        )
        pred_writer.writeheader()

    total = 0
    try:
        for batch in tqdm(loader, desc=dataset_name):
            samples = batch["sample"].to(device, non_blocking=True)
            with torch.no_grad():
                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(samples)
                else:
                    logits = model(samples)
            logits = logits.float().cpu()
            probs = torch.sigmoid(logits)
            top_vals, top_idx = torch.topk(logits, k=5, dim=1)
            top_prob_vals = torch.gather(probs, 1, top_idx)

            batch_size = logits.shape[0]
            total += batch_size

            for name, idx in interesting.items():
                idx_probs = probs[:, idx].numpy()
                idx_logits = logits[:, idx].numpy()
                idx_top5 = (top_idx == idx).any(dim=1).numpy()
                interesting_sums[name]["prob"] += float(idx_probs.sum())
                interesting_sums[name]["logit"] += float(idx_logits.sum())
                interesting_sums[name]["top5"] += int(idx_top5.sum())

            for i in range(batch_size):
                source = batch["source_label"][i]
                names = [audioset_labels[int(idx)]["display_name"] for idx in top_idx[i]]
                indices = [int(idx) for idx in top_idx[i]]
                logits_i = [float(x) for x in top_vals[i]]
                probs_i = [float(x) for x in top_prob_vals[i]]

                top1_name = names[0]
                top1_counts[top1_name] += 1
                top1_by_source[source][top1_name] += 1
                for name in names:
                    top5_counts[name] += 1
                    top5_by_source[source][name] += 1

                for name, idx in interesting.items():
                    interesting_by_source[source][name]["n"] += 1
                    interesting_by_source[source][name]["prob"] += float(probs[i, idx])
                    interesting_by_source[source][name]["logit"] += float(logits[i, idx])
                    if idx in indices:
                        interesting_by_source[source][name]["top5"] += 1

                if pred_writer is not None:
                    pred_writer.writerow(
                        {
                            "dataset": dataset_name,
                            "source_label": source,
                            "wav": batch["path"][i],
                            "frames_before_pad": batch["frames_before_pad"][i],
                            "top1_index": indices[0],
                            "top1_name": top1_name,
                            "top1_logit": f"{logits_i[0]:.6f}",
                            "top1_prob": f"{probs_i[0]:.6f}",
                            "top5_names": "|".join(names),
                            "top5_indices": "|".join(str(x) for x in indices),
                            "top5_logits": "|".join(f"{x:.6f}" for x in logits_i),
                            "top5_probs": "|".join(f"{x:.6f}" for x in probs_i),
                        }
                    )
    finally:
        if pred_file is not None:
            pred_file.close()

    write_top_counts(output_dir / "overall_top1_counts.csv", top1_counts, total, "top1_name")
    write_top_counts(output_dir / "overall_top5_counts.csv", top5_counts, total, "top5_name")
    write_by_source(output_dir / "top1_by_guitarfx_label.csv", top1_by_source)
    write_interesting(output_dir / "interesting_audioset_classes_by_guitarfx_label.csv", interesting_by_source)

    top1_name, top1_count = top1_counts.most_common(1)[0]
    combined_rows.append(
        {
            "dataset": dataset_name,
            "n": total,
            "top1_name": top1_name,
            "top1_count": top1_count,
            "top1_fraction": f"{top1_count / total:.6f}",
            "music_top1_fraction": f"{top1_counts.get('Music', 0) / total:.6f}",
            "distortion_top1_fraction": f"{top1_counts.get('Distortion', 0) / total:.6f}",
            "distortion_top5_fraction": f"{top5_counts.get('Distortion', 0) / total:.6f}",
            "guitar_top5_fraction": f"{top5_counts.get('Guitar', 0) / total:.6f}",
            "effects_unit_top5_fraction": f"{top5_counts.get('Effects unit', 0) / total:.6f}",
        }
    )


def write_top_counts(path, counts, total, field):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field, "count", "fraction"])
        writer.writeheader()
        for name, count in counts.most_common():
            writer.writerow({field: name, "count": count, "fraction": f"{count / total:.6f}"})


def write_by_source(path, counts_by_source):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_label", "n", "top1_name", "count", "fraction"],
        )
        writer.writeheader()
        for source in sorted(counts_by_source):
            total = sum(counts_by_source[source].values())
            for name, count in counts_by_source[source].most_common(10):
                writer.writerow(
                    {
                        "source_label": source,
                        "n": total,
                        "top1_name": name,
                        "count": count,
                        "fraction": f"{count / total:.6f}",
                    }
                )


def write_interesting(path, by_source):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_label",
                "audioset_class",
                "n",
                "mean_logit",
                "mean_prob",
                "top5_count",
                "top5_fraction",
            ],
        )
        writer.writeheader()
        for source in sorted(by_source):
            for name in INTERESTING_AUDIOSET_CLASSES:
                if name not in by_source[source]:
                    continue
                item = by_source[source][name]
                n = item["n"]
                writer.writerow(
                    {
                        "source_label": source,
                        "audioset_class": name,
                        "n": n,
                        "mean_logit": f"{item['logit'] / n:.6f}",
                        "mean_prob": f"{item['prob'] / n:.6f}",
                        "top5_count": item["top5"],
                        "top5_fraction": f"{item['top5'] / n:.6f}",
                    }
                )


if __name__ == "__main__":
    main()
