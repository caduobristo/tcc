import argparse
import csv
import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
GUITARFX_ROOT = EXPERIMENT_ROOT.parent / "fxnet_reproduction" / "data" / "GUITAR-FX"

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


AUDIOMAE_MEAN = -4.2677393
AUDIOMAE_STD = 4.5689974
AUDIOMAE_TARGET_LENGTH = 1024
AUDIOMAE_MEL_BINS = 128


class GuitarFxEmbeddingDataset(Dataset):
    def __init__(self, entries, cycle_to_10s):
        self.entries = entries
        self.cycle_to_10s = cycle_to_10s

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        fbank, frames_before_pad = wav_to_fbank(entry["path"], self.cycle_to_10s)
        return {
            "sample": fbank,
            "dataset": entry["dataset"],
            "source_label": entry["source_label"],
            "path": str(entry["path"]),
            "frames_before_pad": frames_before_pad,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--guitarfx-root", default=str(GUITARFX_ROOT))
    parser.add_argument("--checkpoint", default=str(EXPERIMENT_ROOT / "ckpt" / "finetuned.pth"))
    parser.add_argument("--dataset", action="append", default=["Mono_Continuous"])
    parser.add_argument("--sample-per-class", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-nofx", action="store_true", default=True)
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--cycle-to-10s", action="store_true", default=True)
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENT_ROOT / "outputs" / "guitarfx_audiomae_embeddings_mono_continuous"),
    )
    return parser.parse_args()


def discover_entries(guitarfx_root, dataset_names, sample_per_class, include_nofx, excluded, seed):
    rng = np.random.default_rng(seed)
    entries = []
    for dataset_name in dataset_names:
        audio_dir = Path(guitarfx_root) / dataset_name / "Audio"
        if not audio_dir.exists():
            raise FileNotFoundError(audio_dir)
        class_dirs = sorted(p for p in audio_dir.iterdir() if p.is_dir())
        for class_dir in class_dirs:
            if class_dir.name.startswith("_NoFX"):
                if not include_nofx or not class_dir.name.endswith("_preprocessed"):
                    continue
                label = "NoFX"
            else:
                label = class_dir.name
            if label in excluded:
                continue
            wavs = sorted(class_dir.glob("*.wav"))
            if not wavs:
                continue
            if sample_per_class > 0 and len(wavs) > sample_per_class:
                chosen = np.sort(rng.choice(len(wavs), size=sample_per_class, replace=False))
                wavs = [wavs[i] for i in chosen]
            entries.extend({"dataset": dataset_name, "source_label": label, "path": wav} for wav in wavs)
    return entries


def wav_to_fbank(path, cycle_to_10s):
    waveform_np, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=1)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if cycle_to_10s and waveform.shape[1] < sr * 10:
        repeats = int(np.ceil((sr * 10) / waveform.shape[1]))
        waveform = waveform.repeat(1, repeats)[:, : sr * 10]

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
    state = ckpt["model"]
    state = {
        k: v
        for k, v in state.items()
        if not k.startswith("decoder_")
        and not k.startswith("mask_token")
        and k not in {"head.weight", "head.bias"}
    }
    msg = model.load_state_dict(state, strict=False)
    return model, ckpt, msg


def collate(items):
    return {
        "sample": torch.stack([item["sample"] for item in items], dim=0),
        "dataset": [item["dataset"] for item in items],
        "source_label": [item["source_label"] for item in items],
        "path": [item["path"] for item in items],
        "frames_before_pad": [item["frames_before_pad"] for item in items],
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = discover_entries(
        args.guitarfx_root,
        args.dataset,
        args.sample_per_class,
        args.include_nofx,
        set(args.exclude_label),
        args.seed,
    )
    if not entries:
        raise RuntimeError("No entries selected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, load_msg = build_model(args.root, args.checkpoint)
    model = model.to(device).eval()

    ds = GuitarFxEmbeddingDataset(entries, args.cycle_to_10s)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    embeddings = []
    meta_rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="embeddings"):
            samples = batch["sample"].to(device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    emb = model.forward_features(samples)
            else:
                emb = model.forward_features(samples)
            embeddings.append(emb.float().cpu().numpy())
            for i, path in enumerate(batch["path"]):
                meta_rows.append(
                    {
                        "dataset": batch["dataset"][i],
                        "source_label": batch["source_label"][i],
                        "wav": path,
                        "frames_before_pad": batch["frames_before_pad"][i],
                    }
                )

    embeddings = np.concatenate(embeddings, axis=0)
    np.save(output_dir / "embeddings_768.npy", embeddings)

    with (output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "source_label", "wav", "frames_before_pad"])
        writer.writeheader()
        writer.writerows(meta_rows)

    pca = PCA(n_components=2, random_state=args.seed)
    xy = pca.fit_transform(embeddings)
    labels = np.array([row["source_label"] for row in meta_rows])
    write_pca(output_dir / "pca_2d.csv", xy, meta_rows)
    plot_pca(output_dir / "pca_2d.png", xy, labels)

    config = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": ckpt.get("epoch"),
        "datasets": args.dataset,
        "sample_per_class": args.sample_per_class,
        "n_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "cycle_to_10s": args.cycle_to_10s,
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "silhouette_score_by_source_label_on_embedding": safe_silhouette(embeddings, labels),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "missing_keys": list(load_msg.missing_keys),
        "unexpected_keys": list(load_msg.unexpected_keys),
        "note": "embeddings_768.npy contains model.forward_features output before the AudioSet classification head.",
    }
    (output_dir / "embedding_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


def write_pca(path, xy, meta_rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pca_x", "pca_y", "dataset", "source_label", "wav"])
        writer.writeheader()
        for point, row in zip(xy, meta_rows):
            writer.writerow(
                {
                    "pca_x": f"{float(point[0]):.8f}",
                    "pca_y": f"{float(point[1]):.8f}",
                    "dataset": row["dataset"],
                    "source_label": row["source_label"],
                    "wav": row["wav"],
                }
            )


def plot_pca(path, xy, labels):
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", len(unique_labels))
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.65, label=label, color=cmap(idx))
    ax.set_title("AudioMAE encoder embeddings - PCA 2D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=2, fontsize=8, ncols=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def safe_silhouette(embeddings, labels):
    try:
        if len(set(labels)) < 2:
            return None
        return float(silhouette_score(embeddings, labels, metric="cosine"))
    except Exception:
        return None


if __name__ == "__main__":
    main()
