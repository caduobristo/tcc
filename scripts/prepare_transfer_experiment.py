"""Index/preflight/inference benchmark only. There is deliberately no train command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data.paths import data_root
from src.data.transfer import build_manifest, manifest_path, read_manifest, GuitarWaveforms
from src.models.transfer import load_frozen_encoder
from src.training.linear_probe import load_config, preflight, device_for, autocast_for, seed_all


def benchmark(config, batches=8):
    """Small inference-only timing; no gradients, optimizer or persisted embeddings."""
    seed_all(config["seed"])
    torch.set_num_threads(4)
    device = device_for(config)
    batch_size = config["embedding_batch_size"]
    model = load_frozen_encoder(config["model"]).to(device).eval()
    count = 123552
    preparation = None
    measured_input = "synthetic_two_second_waveforms"
    sample_file = data_root() / "audits/transfer_learning/benchmark_sample.json"
    if manifest_path(config["scenario"]).is_file() or sample_file.is_file():
        complete_manifest = manifest_path(config["scenario"]).is_file()
        rows = read_manifest(config["scenario"]) if complete_manifest else json.loads(sample_file.read_text(encoding="utf-8"))["rows"]
        count = len(rows) if complete_manifest else 123552
        import random
        selection = random.Random(42).sample(rows, min(len(rows), batch_size * batches))
        dataset = GuitarWaveforms(selection)
        started = time.perf_counter()
        # Serial timing is a conservative bound; the extraction loader has four workers.
        examples = [dataset[i][0] for i in range(len(dataset))]
        preparation = (time.perf_counter()-started)/len(examples)
        wave_batches = [torch.stack(examples[i:i+batch_size]).to(device)
                        for i in range(0, len(examples), batch_size)]
        measured_input = "audited_real_wavs" if complete_manifest else "crc_checked_real_benchmark_subset"
    else:
        wave_batches = [(torch.randn(batch_size, 64000) * .05).to(device)] * batches
    waveform = wave_batches[0]

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(2):
            with autocast_for(device, config):
                output = model(waveform)
        sync()
        started = time.perf_counter()
        for waveform in wave_batches:
            with autocast_for(device, config):
                output = model(waveform)
        sync()
        elapsed = time.perf_counter() - started
    if output.shape != (len(waveform), 768) or not torch.isfinite(output).all():
        raise ValueError("Pretrained encoder smoke test failed")
    assert all(not p.requires_grad for p in model.parameters())
    measured_count = sum(len(batch) for batch in wave_batches)
    rate = measured_count / elapsed
    # Forward-only head timing; a multiplier covers backward/AdamW and loading overhead.
    head = nn.Linear(768, 13).to(device).eval()
    features = torch.randn(config["head_batch_size"], 768, device=device)
    with torch.inference_mode():
        for _ in range(10):
            head(features)
        sync()
        started = time.perf_counter()
        for _ in range(100):
            head(features)
        sync()
        head_forward = (time.perf_counter()-started)/100
    gpu_seconds = count/rate
    io_seconds_serial = count*preparation if preparation is not None else None
    # Include per-file CPU work and head iteration overhead rather than promise GPU-only speed.
    extraction_low = gpu_seconds * 1.25
    extraction_high = (gpu_seconds + (io_seconds_serial or gpu_seconds)) * 1.75
    head_batches = count * .80 / config["head_batch_size"] * config["epochs"]
    head_low = max(20., head_batches * max(head_forward * 5, .004))
    head_high = max(120., head_batches * max(head_forward * 10, .03))
    result = {"model": config["model"], "device": str(device),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "measured_input": measured_input, "inference_samples_measured": measured_count,
              "batch_size": len(waveform), "inference_seconds": elapsed,
              "samples_per_second": rate, "serial_wav_preparation_seconds_per_sample": preparation,
              "gpu_memory_peak_GiB": torch.cuda.max_memory_allocated()/2**30 if device.type == "cuda" else None,
              "projected_samples": count, "epochs": config["epochs"],
              "first_run_seconds_range": [extraction_low + head_low, extraction_high + head_high],
              "head_only_seconds_range": [head_low, head_high],
              "estimate_limitations": "Short inference timing, extrapolated. Head backward/optimizer not timed. Download/setup excluded. I/O caching and worker contention can change throughput.",
              "training_started": False, "optimizer_steps": 0}
    target = data_root() / "audits/transfer_learning" / (config["model"] + "_benchmark.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["index", "preflight", "benchmark"])
    parser.add_argument("--scenario", choices=["mono_disc", "mono_cont", "poly_disc", "poly_cont"], default="mono_disc")
    parser.add_argument("--model", choices=["passt", "htsat"], default="passt")
    args = parser.parse_args()
    if args.action == "index":
        result = build_manifest(args.scenario)
    else:
        config = load_config(ROOT / "configs/linear_probe" / f"{args.model}_mono_disc.json")
        config["scenario"] = args.scenario
        result = preflight(config) if args.action == "preflight" else benchmark(config)
    print(json.dumps(result, indent=2))
    if result.get("ready") is False:
        sys.exit(1)
