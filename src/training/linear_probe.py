"""Explicitly gated embedding extraction and training of the classifier head only."""
from __future__ import annotations

from contextlib import nullcontext
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data.transfer import EFFECTS, GuitarWaveforms, manifest_path, read_manifest, sha256
from src.data.paths import SCENARIOS
from src.models.transfer import PROJECT_ROOT, WEIGHTS_ROOT, WEIGHTS, WEIGHT_SHA256, load_frozen_encoder


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["model"] not in WEIGHTS or config["scenario"] not in SCENARIOS or config["sample_rate"] != 32000 or config["clip_seconds"] != 2:
        raise ValueError("Unsupported encoder or audio recipe")
    if config.get("train_encoder", False):
        raise ValueError("This implementation supports frozen-encoder linear probing only")
    return config


def device_for(config):
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def autocast_for(device, config):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and config.get("amp_bfloat16", False) else nullcontext()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def paths_for(config):
    if config["scenario"] not in SCENARIOS or config["model"] not in WEIGHTS:
        raise ValueError("Unknown transfer-learning model/scenario")
    folder = PROJECT_ROOT / "results/transfer_learning" / config["scenario"] / config["model"]
    return folder, folder / "embeddings.npy", folder / "embedding_cache.json"


def cache_identity(config):
    files = [PROJECT_ROOT / "src/models/transfer.py", PROJECT_ROOT / "src/data/transfer.py"]
    if config["model"] == "htsat":
        files += list((PROJECT_ROOT / "src/vendor/htsat").glob("*.py"))
    packages = ("torch", "torchaudio", "hear21passt", "torchlibrosa", "librosa", "timm")
    return {"schema": 1, "model": config["model"], "scenario": config["scenario"],
            "manifest_sha256": sha256(manifest_path(config["scenario"])),
            "checkpoint_sha256": sha256(WEIGHTS_ROOT / WEIGHTS[config["model"]]),
            "sample_rate": 32000, "clip_seconds": 2, "encoder_eval": True,
            "encoder_precision": "bfloat16" if device_for(config).type == "cuda" and config.get("amp_bfloat16", False) else "float32",
            "embedding_dim": 768,
            "implementation": {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in files},
            "packages": {p: importlib.metadata.version(p) for p in packages}}


def preflight(config):
    errors = []
    target = manifest_path(config["scenario"])
    if not target.is_file():
        errors.append(f"Audited WAV manifest missing: {target}")
    weights = WEIGHTS_ROOT / WEIGHTS[config["model"]]
    if not weights.is_file():
        errors.append("Official pretrained checkpoint missing")
    elif sha256(weights) != WEIGHT_SHA256[config["model"]]:
        errors.append("Pretrained checkpoint checksum mismatch")
    count = 0
    counts = {}
    if target.is_file():
        rows = read_manifest(config["scenario"])
        count = len(rows)
        report = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        if report["manifest_sha256"] != sha256(target) or report["seed"] != config["seed"]:
            errors.append("Dataset index changed or seed differs from its audited split")
        groups = {}
        for row in rows:
            if row["effect"] not in EFFECTS or int(row["label"]) != EFFECTS.index(row["effect"]):
                errors.append("Class mapping mismatch")
                break
            group, split = row["source_group"], row["split"]
            if group in groups and groups[group] != split:
                errors.append("Source recording leaks across splits")
                break
            groups[group] = split
        counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}
        for split in counts:
            if {r["effect"] for r in rows if r["split"] == split} != set(EFFECTS):
                errors.append(f"Split {split} lacks a class")
        dataset = GuitarWaveforms(rows)
        # Full WAVs were checked during indexing. Recheck a selection here, failing on changed bytes.
        for index in sorted(set([0, len(rows)//2, len(rows)-1])):
            try:
                dataset[index]
            except Exception as exc:
                errors.append(str(exc))
    return {"ready": not errors, "errors": errors, "device": str(device_for(config)),
            "model": config["model"], "samples": count, "split_counts": counts,
            "classes": list(EFFECTS), "encoder_training": False,
            "representation": "official 128-band PaSST frontend" if config["model"] == "passt" else "official 64-band HTS-AT frontend + native short-clip interpolation"}


def extract_embeddings(config, *, allow_extraction=False):
    if not allow_extraction:
        raise RuntimeError("Embedding extraction disabled. Set RUN_EXTRACTION=True explicitly in the notebook.")
    status = preflight(config)
    if not status["ready"]:
        raise RuntimeError(status["errors"])
    seed_all(config["seed"])
    rows = read_manifest(config["scenario"])
    folder, target, sidecar = paths_for(config)
    identity = cache_identity(config)
    if target.exists() or sidecar.exists():
        validate_cache(config)
        print("Reusing the complete matching embedding cache.")
        return target
    folder.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("embeddings.partial.npy")
    if temporary.exists():
        raise FileExistsError("Incomplete cache present; inspect it before retrying. No complete cache is overwritten.")
    cache = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(rows), 768))
    device = device_for(config)
    model = load_frozen_encoder(config["model"]).to(device).eval()
    loader = DataLoader(GuitarWaveforms(rows), batch_size=config["embedding_batch_size"],
                        num_workers=config["num_workers"], shuffle=False, pin_memory=device.type == "cuda")
    started = time.perf_counter()
    written = 0
    with torch.inference_mode():
        for waveforms, labels, indices in tqdm(loader, desc="Extracting frozen embeddings"):
            with autocast_for(device, config):
                features = model(waveforms.to(device, non_blocking=True))
            values = features.float().cpu().numpy()
            if values.shape != (len(indices), 768) or not np.isfinite(values).all():
                raise ValueError("Invalid encoder output")
            cache[indices.numpy()] = values
            written += len(indices)
    if written != len(rows):
        raise RuntimeError("Incomplete embedding coverage")
    cache.flush()
    del cache
    temporary.rename(target)
    record = {"identity": identity, "complete": True, "samples": written,
              "shape": [written, 768], "dtype": "float32", "sha256": sha256(target),
              "elapsed_seconds": time.perf_counter() - started}
    temp_json = sidecar.with_suffix(".tmp")
    temp_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temp_json.rename(sidecar)
    return target


def validate_cache(config):
    _, target, sidecar = paths_for(config)
    if not target.exists() or not sidecar.exists():
        raise FileNotFoundError("Complete embedding cache missing; run the explicit extraction stage first")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if not record["complete"] or record["identity"] != cache_identity(config) or sha256(target) != record["sha256"]:
        raise ValueError("Embedding cache is stale, incomplete or changed")
    array = np.load(target, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(read_manifest(config["scenario"])), 768) or array.dtype != np.float32:
        raise ValueError("Embedding shape/dtype mismatch")
    for start in range(0, len(array), 4096):
        if not np.isfinite(array[start:start+4096]).all():
            raise ValueError("Non-finite embedding")
    return record


class Embeddings(Dataset):
    def __init__(self, path, rows, split):
        self.array = np.load(path, mmap_mode="r", allow_pickle=False)
        self.indices = [i for i, row in enumerate(rows) if row["split"] == split]
        self.labels = [int(row["label"]) for row in rows]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        at = self.indices[index]
        return torch.from_numpy(self.array[at].copy()), self.labels[at]


def train_probe(config, *, allow_training=False):
    """Train only a fresh linear layer; this function cannot unfreeze an encoder."""
    if not allow_training:
        raise RuntimeError("Training disabled. Set RUN_TRAINING=True explicitly in the notebook.")
    record = validate_cache(config)
    seed_all(config["seed"])
    rows = read_manifest(config["scenario"])
    folder, embeddings, _ = paths_for(config)
    run = folder / time.strftime("probe_%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    device = device_for(config)
    head = nn.Linear(768, len(EFFECTS)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(config["seed"])
    loaders = {s: DataLoader(Embeddings(embeddings, rows, s), batch_size=config["head_batch_size"],
                            shuffle=s == "train", generator=generator if s == "train" else None,
                            num_workers=0, pin_memory=device.type == "cuda") for s in ("train", "validation", "test")}
    best = -1.
    history = []
    start = time.perf_counter()
    for epoch in range(config["epochs"]):
        stats = {"epoch": epoch + 1}
        for split in ("train", "validation"):
            head.train(split == "train")
            loss_total, correct, count = 0., 0, 0
            context = torch.enable_grad() if split == "train" else torch.no_grad()
            with context:
                for features, labels in loaders[split]:
                    features, labels = features.to(device), labels.to(device)
                    logits = head(features)
                    loss = criterion(logits, labels)
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    count += len(labels)
                    correct += int((logits.argmax(1) == labels).sum())
                    loss_total += float(loss.detach()) * len(labels)
            stats[split + "_loss"] = loss_total/count
            stats[split + "_accuracy"] = correct/count
        history.append(stats)
        print(stats)
        if stats["validation_accuracy"] > best:
            best = stats["validation_accuracy"]
            torch.save(head.state_dict(), run / "best_head.pt")
    head.load_state_dict(torch.load(run / "best_head.pt", map_location=device, weights_only=True))
    head.eval()
    true, predicted = [], []
    with torch.no_grad():
        for features, labels in loaders["test"]:
            predicted.extend(head(features.to(device)).argmax(1).cpu().tolist())
            true.extend(labels.tolist())
    labels = list(range(len(EFFECTS)))
    report = classification_report(true, predicted, labels=labels, target_names=EFFECTS, output_dict=True, zero_division=0)
    metrics = {"accuracy": float(accuracy_score(true, predicted)),
               "macro_precision": report["macro avg"]["precision"],
               "macro_recall": report["macro avg"]["recall"], "macro_f1": report["macro avg"]["f1-score"],
               "classification_report": report,
               "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist()}
    (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"config": config, "embedding_cache": record,
        "classes": list(EFFECTS), "encoder_frozen": True, "training_seconds": time.perf_counter()-start}, indent=2), encoding="utf-8")
    from src.utils.metrics import plot_confusion_matrix
    plot_confusion_matrix(np.asarray(metrics["confusion_matrix"]), list(EFFECTS), str(run / "confusion_matrix.png"))
    return run, metrics
