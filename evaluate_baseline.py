import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn

from src.data.dataset import FxDataset
from src.data.datasplit import DataSplit
from src.models.fxnet import FxNet
from src.utils.metrics import compute_metrics, plot_confusion_matrix
from src.utils.logger import setup_logger
from src.utils.device import get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluates baseline FxNet model on GUITAR-FX dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/baseline_mono_disc.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../external/gfx-classifier_models_and_results/models/20201027_fxnet_mono_disc_best",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset root path (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'dml', or 'cpu')",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Sample limit for quick evaluation test",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.abspath(args.config)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    data_cfg = cfg.get("data", {})
    output_cfg = cfg.get("output", {})

    dataset_root = args.data_dir or data_cfg.get(
        "dataset_root", "../datasets/GUITAR-FX-DIST/Mono_Discrete"
    )
    spectra_folder = data_cfg.get("spectra_folder", "mel_22050_1024_512")
    excl_folders = data_cfg.get("excl_folders", ["TS9", "MT2"])
    seed = data_cfg.get("seed", 42)

    results_dir = output_cfg.get("results_dir", "results/baseline")
    logger = setup_logger("evaluate_baseline", os.path.join(results_dir, "eval.log"))

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    cache_in_ram = data_cfg.get("cache_in_ram", True)
    logger.info(f"Loading dataset from: {os.path.abspath(dataset_root)} (RAM caching: {cache_in_ram})")
    dataset = FxDataset(
        root=dataset_root,
        excl_folders=excl_folders,
        spectra_folder=spectra_folder,
        cache_in_ram=cache_in_ram,
    )
    dataset.init_dataset()

    logger.info(f"Total dataset samples: {len(dataset)}")
    logger.info(f"Number of effect classes: {dataset.num_fx}")

    split = DataSplit(
        dataset,
        test_train_split=data_cfg.get("test_train_split", 0.8),
        val_train_split=data_cfg.get("val_train_split", 0.1),
        shuffle=data_cfg.get("shuffle", True),
        seed=seed,
    )
    _, _, test_loader = split.get_split(batch_size=100, num_workers=0)
    logger.info(f"Test set size: {len(split.test_sampler)}")

    model = FxNet(n_classes=dataset.num_fx).to(device)

    checkpoint_path = os.path.abspath(args.checkpoint)
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
        try:
            # Map legacy modules for full object checkpoints
            import src.models as models_pkg
            import src.models.fxnet as fxnet_mod

            sys.modules["model"] = models_pkg
            sys.modules["model.models"] = fxnet_mod

            checkpoint_obj = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint_obj, nn.Module):
                state_dict = checkpoint_obj.state_dict()
            elif isinstance(checkpoint_obj, dict):
                if "state_dict" in checkpoint_obj:
                    state_dict = checkpoint_obj["state_dict"]
                else:
                    state_dict = checkpoint_obj
            else:
                state_dict = {}

            # Resize output layer if checkpoint class count differs
            if "out.weight" in state_dict:
                ckpt_classes = state_dict["out.weight"].shape[0]
                if ckpt_classes != model.n_classes:
                    logger.info(
                        f"Resizing model output layer to {ckpt_classes} classes to match checkpoint."
                    )
                    model.out = nn.Linear(in_features=60, out_features=ckpt_classes).to(device)
                    model.n_classes = ckpt_classes

            model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    else:
        logger.warning(
            f"Checkpoint not found at '{checkpoint_path}'. Evaluating with random weights."
        )

    model.eval()
    all_preds = []
    all_targets = []

    start_time = time.time()
    processed_samples = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if data is None:
                continue
            inputs, labels, _, _, _ = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            processed_samples += len(labels)
            if args.max_samples and processed_samples >= args.max_samples:
                logger.info(f"Reached max_samples={args.max_samples} limit.")
                break

    eval_time = time.time() - start_time
    logger.info(
        f"Inference completed in {eval_time:.2f}s ({eval_time/max(processed_samples, 1)*1000:.2f} ms/sample)"
    )

    class_names = [dataset.label_to_fx[i] for i in range(dataset.num_fx)]
    metrics = compute_metrics(all_targets, all_preds, class_names=class_names)

    logger.info("=" * 60)
    logger.info(f"EVALUATION RESULTS (Tested samples: {len(all_targets)})")
    logger.info(f"Accuracy:        {metrics['accuracy']*100:.2f}%")
    logger.info(f"Macro F1-Score:  {metrics['macro_f1']*100:.2f}%")
    logger.info(f"Macro Precision: {metrics['macro_precision']*100:.2f}%")
    logger.info(f"Macro Recall:    {metrics['macro_recall']*100:.2f}%")
    logger.info("=" * 60)

    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        classes=class_names,
        save_path=cm_path,
        normalize=True,
        title="Confusion Matrix - FxNet Baseline",
    )
    logger.info(f"Confusion matrix saved to: {os.path.abspath(cm_path)}")


if __name__ == "__main__":
    main()
