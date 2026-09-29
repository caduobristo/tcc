#!/usr/bin/env python
"""Generate a consolidated Markdown report from FxNet evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_EVAL_DIR = WORKSPACE_ROOT / "models_and_results" / "results" / "_model_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown report for FxNet evaluation runs.")
    parser.add_argument("--model-name", required=True, help="Model name used in evaluation.")
    parser.add_argument("--split", default="full", help="Evaluation split to include.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR), help="Directory containing summary JSON files.")
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def latest_summaries(eval_dir: Path, model_name: str, split_name: str) -> list[dict[str, object]]:
    summaries_by_dataset: dict[str, tuple[Path, dict[str, object]]] = {}
    pattern = f"{model_name}_{split_name}_*_summary.json"
    for path in sorted(eval_dir.glob(pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset_name = Path(str(data["dataset_root"])).parents[0].name
        summaries_by_dataset[dataset_name] = (path, data)
    return [row for _, row in sorted(summaries_by_dataset.values(), key=lambda item: Path(str(item[1]["dataset_root"])).parents[0].name)]


def build_report(eval_dir: Path, model_name: str, split_name: str) -> str:
    summaries = latest_summaries(eval_dir, model_name, split_name)
    if not summaries:
        raise FileNotFoundError(f"No summaries found for {model_name} with split={split_name} in {eval_dir}")

    lines: list[str] = []
    lines.append(f"# Evaluation Report: {model_name}")
    lines.append("")
    lines.append(f"- Split: `{split_name}`")
    lines.append(f"- Source dir: `{eval_dir}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Dataset | Samples | Accuracy | Avg Loss | Macro P | Macro R | Macro F1 | Weighted F1 | Balanced Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for data in summaries:
        dataset_name = Path(str(data["dataset_root"])).parents[0].name
        lines.append(
            f"| {dataset_name} | {data['samples']} | {pct(float(data['accuracy']))} | "
            f"{float(data['avg_loss']):.6f} | {pct(float(data['macro_precision']))} | "
            f"{pct(float(data['macro_recall']))} | {pct(float(data['macro_f1']))} | "
            f"{pct(float(data['weighted_f1']))} | {pct(float(data['balanced_accuracy']))} |"
        )

    for data in summaries:
        dataset_name = Path(str(data["dataset_root"])).parents[0].name
        lines.append("")
        lines.append(f"## {dataset_name}")
        lines.append("")
        lines.append(f"- Samples: `{data['samples']}`")
        lines.append(f"- Accuracy: `{pct(float(data['accuracy']))}`")
        lines.append(f"- Avg loss: `{float(data['avg_loss']):.6f}`")
        lines.append(f"- Macro precision: `{pct(float(data['macro_precision']))}`")
        lines.append(f"- Macro recall: `{pct(float(data['macro_recall']))}`")
        lines.append(f"- Macro F1: `{pct(float(data['macro_f1']))}`")
        lines.append(f"- Weighted precision: `{pct(float(data['weighted_precision']))}`")
        lines.append(f"- Weighted recall: `{pct(float(data['weighted_recall']))}`")
        lines.append(f"- Weighted F1: `{pct(float(data['weighted_f1']))}`")
        lines.append(f"- Balanced accuracy: `{pct(float(data['balanced_accuracy']))}`")
        lines.append(f"- Predictions CSV: `{Path(str(data['predictions_csv'])).name}`")
        lines.append(f"- Per-class CSV: `{Path(str(data['per_class_csv'])).name}`")
        lines.append(f"- Confusion matrix CSV: `{Path(str(data['confusion_matrix_csv'])).name}`")

        top_confusions = list(data.get("top_confusions", []))
        if top_confusions:
            lines.append("")
            lines.append("| Main confusion | Count |")
            lines.append("|---|---:|")
            for row in top_confusions[:8]:
                lines.append(f"| {row['true_fx']} -> {row['pred_fx']} | {row['count']} |")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    report = build_report(eval_dir, args.model_name, args.split)
    out_path = eval_dir / f"{args.model_name}_all_datasets_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
