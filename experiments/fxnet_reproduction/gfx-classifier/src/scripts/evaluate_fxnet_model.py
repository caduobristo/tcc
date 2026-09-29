#!/usr/bin/env python
"""Evaluate a saved FxNet model on a configured dataset split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


SCRIPT_PATH = Path(__file__).resolve()
SRC_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset.dataset import FxDataset  # noqa: E402
from datasplit.datasplit import DataSplit  # noqa: E402
import utils  # noqa: E402


KNOWN_MODELS = {
    "20201024_fxnet_mono_cont_best": {
        "dataset_root": WORKSPACE_ROOT / "data" / "GUITAR-FX" / "Mono_Continuous" / "Features",
        "exclude_folders": ["MT2"],
    },
    "20201025_fxnet_poly_cont_best": {
        "dataset_root": WORKSPACE_ROOT / "data" / "GUITAR-FX" / "Poly_Continuous" / "Features",
        "exclude_folders": ["MT2"],
    },
    "20201210_fxnet_mono_disc_noTS9_best": {
        "dataset_root": WORKSPACE_ROOT / "data" / "GUITAR-FX" / "Mono_Discrete" / "Features",
        "exclude_folders": ["TS9", "MT2"],
    },
    "20201211_fxnet_poly_disc_noTS9_best": {
        "dataset_root": WORKSPACE_ROOT / "data" / "GUITAR-FX" / "Poly_Discrete" / "Features",
        "exclude_folders": ["TS9", "MT2"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved FxNet model.")
    parser.add_argument("--model-name", required=True, help="Model file name in models_and_results/models.")
    parser.add_argument("--dataset-root", help="Override dataset root. Defaults to known config for the model.")
    parser.add_argument(
        "--exclude-folders",
        nargs="*",
        default=None,
        help="Optional excluded folders. Defaults to known config for the model.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "full"],
        default="test",
        help="Dataset split to evaluate. `full` evaluates all samples.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for evaluation.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Evaluation device.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before splitting train/val/test. Off by default for deterministic splits.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used when --shuffle is enabled.")
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "models_and_results" / "results" / "_model_eval"),
        help="Directory used for summary/prediction outputs.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_config(args: argparse.Namespace) -> tuple[Path, Path, list[str]]:
    model_path = WORKSPACE_ROOT / "models_and_results" / "models" / args.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    known = KNOWN_MODELS.get(args.model_name, {})
    dataset_root = Path(args.dataset_root) if args.dataset_root else known.get("dataset_root")
    exclude_folders = args.exclude_folders if args.exclude_folders is not None else known.get("exclude_folders", [])

    if dataset_root is None:
        raise ValueError(
            "No default dataset config for this model. Pass --dataset-root and optionally --exclude-folders."
        )

    if not Path(dataset_root).exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    return model_path, Path(dataset_root), list(exclude_folders)


def make_dataset(dataset_root: Path, exclude_folders: list[str]) -> FxDataset:
    dataset = FxDataset(
        root=str(dataset_root),
        excl_folders=exclude_folders,
        spectra_folder="mel_22050_1024_512",
        processed_settings_csv="proc_settings.csv",
        max_num_settings=3,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    dataset.init_dataset()
    return dataset


def make_loader(
    dataset: FxDataset,
    split_name: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tuple[DataLoader, int]:
    if split_name == "full":
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return loader, len(dataset)

    if shuffle:
        np.random.seed(seed)

    split = DataSplit(dataset, test_train_split=0.8, val_train_split=0.1, shuffle=shuffle)
    train_loader, val_loader, test_loader = split.get_split(batch_size=batch_size, num_workers=0)

    if split_name == "train":
        return train_loader, len(split.train_sampler)
    if split_name == "val":
        return val_loader, len(split.val_sampler)
    return test_loader, len(split.test_sampler)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    sample_count: int,
    dataset: FxDataset,
    device: torch.device,
) -> tuple[float, int, list[dict[str, object]], list[int], list[int]]:
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    predictions: list[dict[str, object]] = []
    all_true_labels: list[int] = []
    all_pred_labels: list[int] = []

    with torch.no_grad():
        for mels, labels, _settings, filenames, indices in loader:
            mels = mels.to(device)
            labels = labels.to(device)

            preds = model(mels)
            loss = loss_function(preds, labels)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int(utils.get_num_correct_labels(preds, labels))

            pred_labels = preds.argmax(dim=1)
            for idx, filename in enumerate(filenames):
                true_id = int(labels[idx].item())
                pred_id = int(pred_labels[idx].item())
                all_true_labels.append(true_id)
                all_pred_labels.append(pred_id)
                predictions.append(
                    {
                        "index": int(indices[idx].item()),
                        "filename": filename,
                        "pred_label": pred_id,
                        "pred_fx": dataset.label_to_fx[pred_id],
                        "true_label": true_id,
                        "true_fx": dataset.label_to_fx[true_id],
                    }
                )

    avg_loss = total_loss / max(sample_count, 1)
    return avg_loss, total_correct, predictions, all_true_labels, all_pred_labels


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def compute_classification_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    dataset: FxDataset,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, list[dict[str, object]]]:
    label_ids = sorted(dataset.label_to_fx.keys())
    num_classes = len(label_ids)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_id, pred_id in zip(true_labels, pred_labels):
        confusion[true_id, pred_id] += 1

    total_samples = int(confusion.sum())
    total_correct = int(np.trace(confusion))
    accuracy = safe_div(total_correct, total_samples)

    per_class_rows: list[dict[str, object]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    weighted_f1_sum = 0.0

    for class_id in label_ids:
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        support = int(confusion[class_id, :].sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        weighted_precision_sum += precision * support
        weighted_recall_sum += recall * support
        weighted_f1_sum += f1 * support

        per_class_rows.append(
            {
                "label_id": class_id,
                "fx_name": dataset.label_to_fx[class_id],
                "support": support,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    weighted_precision = safe_div(weighted_precision_sum, total_samples)
    weighted_recall = safe_div(weighted_recall_sum, total_samples)
    weighted_f1 = safe_div(weighted_f1_sum, total_samples)

    top_confusions: list[dict[str, object]] = []
    for true_id in label_ids:
        for pred_id in label_ids:
            if true_id == pred_id:
                continue
            count = int(confusion[true_id, pred_id])
            if count == 0:
                continue
            top_confusions.append(
                {
                    "true_label": true_id,
                    "true_fx": dataset.label_to_fx[true_id],
                    "pred_label": pred_id,
                    "pred_fx": dataset.label_to_fx[pred_id],
                    "count": count,
                }
            )
    top_confusions.sort(key=lambda row: (-int(row["count"]), str(row["true_fx"]), str(row["pred_fx"])))
    top_confusions = top_confusions[:15]

    summary_metrics = {
        "samples": total_samples,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": macro_recall,
    }
    return summary_metrics, per_class_rows, confusion, top_confusions


def save_outputs(
    output_dir: Path,
    model_name: str,
    split_name: str,
    dataset_root: Path,
    exclude_folders: list[str],
    sample_count: int,
    avg_loss: float,
    total_correct: int,
    predictions: list[dict[str, object]],
    metrics: dict[str, object],
    per_class_rows: list[dict[str, object]],
    confusion: np.ndarray,
    top_confusions: list[dict[str, object]],
    label_names: list[str],
    device: torch.device,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{Path(model_name).name}_{split_name}_{stamp}"
    summary_path = output_dir / f"{stem}_summary.json"
    predictions_path = output_dir / f"{stem}_predictions.csv"
    per_class_path = output_dir / f"{stem}_per_class.csv"
    confusion_path = output_dir / f"{stem}_confusion_matrix.csv"

    accuracy = total_correct / max(sample_count, 1)
    summary = {
        "model_name": model_name,
        "split": split_name,
        "dataset_root": str(dataset_root),
        "exclude_folders": exclude_folders,
        "device": str(device),
        "samples": sample_count,
        "avg_loss": avg_loss,
        "total_correct": total_correct,
        "accuracy": accuracy,
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "predictions_csv": str(predictions_path),
        "per_class_csv": str(per_class_path),
        "confusion_matrix_csv": str(confusion_path),
        "top_confusions": top_confusions,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["index", "filename", "pred_label", "pred_fx", "true_label", "true_fx"],
        )
        writer.writeheader()
        writer.writerows(predictions)

    with per_class_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "label_id",
                "fx_name",
                "support",
                "true_positive",
                "false_positive",
                "false_negative",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(per_class_rows)

    with confusion_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true/pred"] + label_names)
        for idx, row in enumerate(confusion):
            writer.writerow([label_names[idx]] + [int(value) for value in row.tolist()])

    return summary_path, predictions_path, per_class_path, confusion_path


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    model_path, dataset_root, exclude_folders = resolve_model_config(args)

    print(f"Loading model: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)

    print(f"Building dataset from: {dataset_root}")
    dataset = make_dataset(dataset_root, exclude_folders)
    loader, sample_count = make_loader(dataset, args.split, args.batch_size, args.shuffle, args.seed)

    avg_loss, total_correct, predictions, true_labels, pred_labels = evaluate_model(
        model, loader, sample_count, dataset, device
    )
    metrics, per_class_rows, confusion, top_confusions = compute_classification_metrics(
        true_labels, pred_labels, dataset
    )
    accuracy = float(metrics["accuracy"])
    label_names = [dataset.label_to_fx[idx] for idx in sorted(dataset.label_to_fx.keys())]

    summary_path, predictions_path, per_class_path, confusion_path = save_outputs(
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        split_name=args.split,
        dataset_root=dataset_root,
        exclude_folders=exclude_folders,
        sample_count=sample_count,
        avg_loss=avg_loss,
        total_correct=total_correct,
        predictions=predictions,
        metrics=metrics,
        per_class_rows=per_class_rows,
        confusion=confusion,
        top_confusions=top_confusions,
        label_names=label_names,
        device=device,
    )

    print("")
    print(f"Samples: {sample_count}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average loss: {avg_loss:.6f}")
    print(f"Macro F1: {float(metrics['macro_f1']) * 100:.2f}%")
    print(f"Weighted F1: {float(metrics['weighted_f1']) * 100:.2f}%")
    print(f"Summary: {summary_path}")
    print(f"Predictions CSV: {predictions_path}")
    print(f"Per-class CSV: {per_class_path}")
    print(f"Confusion matrix CSV: {confusion_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
