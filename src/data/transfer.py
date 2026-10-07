"""Shared raw-waveform index and source-group splits for frozen-encoder probes."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import torchaudio

from src.data.paths import SCENARIOS, data_root

EFFECTS = ("808", "BD2", "BMF", "DPL", "DS1", "FFC", "MGS", "OD1", "RAT", "RBM", "SD1", "TS9", "VTB")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_group(filename, effect):
    """Reconstruct the original IDMT recording ID, including its final file ID."""
    stem = Path(filename).stem
    marker = f"-{effect}-"
    if stem.count(marker) != 1:
        raise ValueError(f"Cannot identify source recording: {filename}")
    prefix, settings_id = stem.split(marker)
    original_id = settings_id.rsplit("-", 1)[-1]
    if not original_id.isdigit() or len(prefix.split("-")) != 2:
        raise ValueError(f"Unexpected GUITAR-FX filename: {filename}")
    return f"{prefix}-1111-{original_id}"


def group_splits(groups, seed=42):
    """Approximately 72/8/20 by source group; sizes by sample can differ."""
    groups = sorted(set(groups))
    if len(groups) < 10:
        raise ValueError("Too few independent source recordings for the split")
    random.Random(seed).shuffle(groups)
    n_train, n_val = int(len(groups) * .72), int(len(groups) * .08)
    return {g: "train" if i < n_train else "validation" if i < n_train + n_val else "test"
            for i, g in enumerate(groups)}


def manifest_path(scenario="mono_disc"):
    return data_root() / "manifests/transfer_learning" / scenario / "samples.csv"


def build_manifest(scenario="mono_disc", seed=42, workers=4):
    """Audit selected WAVs and save a shared index; no feature extraction/training."""
    root = data_root()
    name = SCENARIOS[scenario]
    audio = root / "raw/guitar_fx_dist" / name / "Audio"
    if not audio.is_dir():
        raise FileNotFoundError(f"WAVs missing: {audio}. Run scripts/prepare_transfer_data.py --extract.")
    rows = []
    for label, effect in enumerate(EFFECTS):
        metadata = root / "metadata/guitar_fx_dist/validated" / name / effect / "proc_settings.csv"
        with metadata.open(encoding="utf-8", newline="") as stream:
            settings = sorted(csv.DictReader(stream), key=lambda row: row["filename"])
        # Source archive may have nested audio folders. Match names unambiguously.
        paths = {}
        for path in sorted((audio / effect).rglob("*.wav")):
            if path.name in paths:
                raise ValueError(f"Ambiguous WAV basename: {path.name}")
            paths[path.name] = path
        if not paths:
            raise FileNotFoundError(f"WAV class missing: {audio / effect}")
        for setting in settings:
            filename = setting["filename"]
            if filename not in paths:
                raise FileNotFoundError(f"Validated metadata has no WAV: {effect}/{filename}")
            rows.append({"filename": filename, "effect": effect, "label": label,
                         "source_group": source_group(filename, effect),
                         "path": paths[filename].relative_to(root).as_posix()})
    assignments = group_splits([row["source_group"] for row in rows], seed)

    def audit(row):
        path = root / row["path"]
        payload = path.read_bytes()
        waveform, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        if rate != 44100 or waveform.shape[1] != 1 or len(waveform) not in (88200, 88201) or not np.isfinite(waveform).all():
            raise ValueError(f"Unexpected original waveform: {path}: {rate}, {waveform.shape}")
        return {**row, "split": assignments[row["source_group"]], "sha256": hashlib.sha256(payload).hexdigest(),
                "sample_rate": rate, "frames": len(waveform), "bytes": len(payload)}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        audited = list(executor.map(audit, rows))
    for split in ("train", "validation", "test"):
        if {r["effect"] for r in audited if r["split"] == split} != set(EFFECTS):
            raise ValueError(f"Split {split} lacks a class")
    hash_splits = {}
    duplicate_rows = 0
    for row in audited:
        if row["sha256"] in hash_splits:
            duplicate_rows += 1
            if hash_splits[row["sha256"]] != row["split"]:
                raise ValueError("Identical WAV content crosses partitions; group duplicates before training")
        hash_splits[row["sha256"]] = row["split"]
    target = manifest_path(scenario)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audited[0]))
        writer.writeheader()
        writer.writerows(audited)
    temporary.replace(target)
    report = {"scenario": scenario, "seed": seed, "classes": list(EFFECTS), "samples": len(audited),
              "source_groups": len(assignments), "source_group_rule": "original IDMT filename including final file ID",
              "split_counts": {s: sum(r["split"] == s for r in audited) for s in ("train", "validation", "test")},
              "group_counts": {s: sum(v == s for v in assignments.values()) for s in ("train", "validation", "test")},
              "wav_integrity_ok": True, "duplicate_wav_rows": duplicate_rows,
              "identical_wav_leakage": False, "manifest_sha256": sha256(target)}
    target.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def read_manifest(scenario="mono_disc"):
    path = manifest_path(scenario)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty transfer-learning manifest")
    return rows


class GuitarWaveforms(Dataset):
    """Read checked WAVs, convert to 32 kHz, preserve the real two-second clip."""
    def __init__(self, rows, target_rate=32000):
        self.rows = rows
        self.root = data_root().resolve()
        self.target_rate = target_rate
        self.resamplers = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("WAV path escapes the data root")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"WAV changed since audit: {path}")
        wave, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        if rate != int(row["sample_rate"]) or len(wave) != int(row["frames"]) or not np.isfinite(wave).all():
            raise ValueError(f"Invalid WAV: {path}")
        tensor = torch.from_numpy(wave.T.copy()).mean(0, keepdim=True)
        if rate != self.target_rate:
            if rate not in self.resamplers:
                self.resamplers[rate] = torchaudio.transforms.Resample(rate, self.target_rate)
            tensor = self.resamplers[rate](tensor)
        # Official files can contain 88,201 samples (inclusive endpoint).
        # Standardize to exactly two seconds after resampling, trimming that endpoint.
        tensor = tensor[:, :self.target_rate * 2]
        if tensor.shape != (1, self.target_rate * 2):
            raise ValueError(f"Clip length mismatch: {path}")
        return tensor.squeeze(0), int(row["label"]), index
