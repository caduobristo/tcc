import os
import sys
import subprocess
import argparse

CONFIGS = [
    "config/baseline_mono_disc.yaml",
    "config/baseline_mono_cont.yaml",
    "config/baseline_poly_disc.yaml",
    "config/baseline_poly_cont.yaml",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs baseline (FxNet) experiments across all 4 GUITAR-FX sub-datasets"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count for all experiments",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'dml', or 'cpu')",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Batch limit per epoch for quick test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    python_exe = sys.executable

    # Detect default device (prefer DirectML for AMD GPUs on Windows if available)
    default_device = args.device
    if not default_device:
        try:
            import torch_directml
            default_device = "dml"
        except ImportError:
            pass

    print("=" * 70)
    print("STARTING BASELINE RUN FOR ALL 4 GUITAR-FX SUB-DATASETS")
    print("=" * 70)

    for cfg in CONFIGS:
        cfg_path = os.path.abspath(cfg)
        if not os.path.exists(cfg_path):
            print(f"[WARNING] Config file not found: {cfg_path}")
            continue

        print(f"\n>>> Running experiment: {cfg}")
        cmd = [python_exe, "train_baseline.py", "--config", cfg]
        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])
        if default_device:
            cmd.extend(["--device", default_device])
        if args.max_batches:
            cmd.extend(["--max_batches", str(args.max_batches)])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ERROR] Experiment failed for {cfg} (exit code: {result.returncode})")
        else:
            print(f"[SUCCESS] Completed: {cfg}")

    print("\n" + "=" * 70)
    print("ALL BASELINE EXPERIMENTS COMPLETED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
