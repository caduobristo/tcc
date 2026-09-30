#!/usr/bin/env python
"""Run the training notebooks sequentially with persistent logs and resume support."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


@dataclass
class Paths:
    repo_root: Path
    workspace_root: Path
    src_root: Path
    results_root: Path
    run_root: Path
    logs_root: Path
    executed_nb_root: Path
    state_json: Path
    summary_md: Path
    monitor_log: Path


DEFAULT_NOTEBOOKS: List[str] = [
    "train_fxnet_on_mono_cont.ipynb",
    "train_fxnet_on_poly_cont.ipynb",
    "train_fxnet_on_mono_disc.ipynb",
    "train_fxnet_on_poly_disc.ipynb",
    "train_setnetcond_on_mono_cont.ipynb",
    "train_setnetcond_on_poly_cont.ipynb",
    "train_setnetcond_on_mono_disc.ipynb",
    "train_setnetcond_on_poly_disc.ipynb",
    "train_multinet_on_mono_disc.ipynb",
    "train_setnet_on_mono_disc.ipynb",
    "train_fxnet_and_setnetcond_on_mono_disc.ipynb",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def configure_cache_dirs(paths: Paths) -> None:
    cache_root = paths.workspace_root / ".cache"
    temp_dir = cache_root / "temp"
    mpl_dir = cache_root / "matplotlib"
    numba_dir = cache_root / "numba"
    jupyter_root = cache_root / "jupyter"
    jupyter_config = jupyter_root / "config"
    jupyter_data = jupyter_root / "data"
    jupyter_runtime = jupyter_root / "runtime"
    ipython_dir = jupyter_root / "ipython"

    for p in [
        temp_dir,
        mpl_dir,
        numba_dir,
        jupyter_config,
        jupyter_data,
        jupyter_runtime,
        ipython_dir,
        paths.logs_root,
        paths.executed_nb_root,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_dir)
    os.environ["JUPYTER_CONFIG_DIR"] = str(jupyter_config)
    os.environ["JUPYTER_DATA_DIR"] = str(jupyter_data)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(jupyter_runtime)
    os.environ["IPYTHONDIR"] = str(ipython_dir)
    os.environ["HOME"] = str(paths.workspace_root)


def build_paths() -> Paths:
    script_path = Path(__file__).resolve()
    src_root = script_path.parents[1]
    repo_root = script_path.parents[2]
    workspace_root = repo_root.parent
    results_root = workspace_root / "models_and_results" / "results"
    run_root = results_root / "_article_repro"

    return Paths(
        repo_root=repo_root,
        workspace_root=workspace_root,
        src_root=src_root,
        results_root=results_root,
        run_root=run_root,
        logs_root=run_root / "logs",
        executed_nb_root=run_root / "executed_notebooks",
        state_json=run_root / "pipeline_state.json",
        summary_md=run_root / "pipeline_summary.md",
        monitor_log=run_root / "pipeline_monitor.log",
    )


def load_state(path: Path) -> Dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "run_started_at": now_iso(),
        "last_updated_at": now_iso(),
        "python_executable": sys.executable,
        "items": {},
        "order": [],
    }


def save_state(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated_at"] = now_iso()
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def append_monitor_line(paths: Paths, message: str) -> None:
    stamped = f"[{now_iso()}] {message}"
    print(stamped, flush=True)
    with paths.monitor_log.open("a", encoding="utf-8") as f:
        f.write(stamped + "\n")


def write_summary(paths: Paths, state: Dict) -> None:
    lines: List[str] = []
    lines.append("# Article Reproduction Pipeline")
    lines.append("")
    lines.append(f"- Updated at: `{state.get('last_updated_at', now_iso())}`")
    lines.append(f"- Started at: `{state.get('run_started_at', '-')}`")
    lines.append(f"- Python: `{state.get('python_executable', '-')}`")
    lines.append("")
    lines.append("| Notebook | Status | Duration (min) | Started | Ended | Log |")
    lines.append("|---|---:|---:|---|---|---|")

    items = state.get("items", {})
    for name in state.get("order", []):
        entry = items.get(name, {})
        duration = entry.get("duration_sec")
        duration_min = f"{duration / 60:.1f}" if isinstance(duration, (int, float)) else "-"
        log_path = entry.get("log_path", "")
        if log_path:
            log_rel = Path(log_path).name
        else:
            log_rel = "-"
        lines.append(
            f"| {name} | {entry.get('status', 'pending')} | {duration_min} | "
            f"{entry.get('started_at', '-')} | {entry.get('ended_at', '-')} | {log_rel} |"
        )

    paths.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_dataset_paths(paths: Paths, notebooks: List[str]) -> None:
    required_roots = [
        paths.workspace_root / "data" / "GUITAR-FX" / "Mono_Continuous" / "Features",
        paths.workspace_root / "data" / "GUITAR-FX" / "Mono_Discrete" / "Features",
        paths.workspace_root / "data" / "GUITAR-FX" / "Poly_Continuous" / "Features",
        paths.workspace_root / "data" / "GUITAR-FX" / "Poly_Discrete" / "Features",
        paths.workspace_root / "models_and_results" / "models",
        paths.workspace_root / "models_and_results" / "results",
    ]
    for p in required_roots:
        if not p.exists():
            raise FileNotFoundError(f"Required path does not exist: {p}")

    for nb in notebooks:
        nb_path = paths.src_root / nb
        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {nb_path}")


def detect_torch_runtime() -> Dict:
    try:
        import torch  # type: ignore

        return {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


def notebook_command(paths: Paths, notebook_name: str) -> List[str]:
    nb_path = paths.src_root / notebook_name
    output_name = f"{nb_path.stem}_executed"
    return [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(nb_path),
        "--output",
        output_name,
        "--output-dir",
        str(paths.executed_nb_root),
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.allow_errors=False",
        "--ExecutePreprocessor.kernel_name=python3",
    ]


def run_one_notebook(
    paths: Paths,
    state: Dict,
    notebook_name: str,
    continue_on_error: bool,
) -> bool:
    start_ts = time.time()
    started_at = now_iso()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = paths.logs_root / f"{Path(notebook_name).stem}_{stamp}.log"

    entry = state["items"].get(notebook_name, {})
    entry.update(
        {
            "status": "running",
            "started_at": started_at,
            "ended_at": None,
            "duration_sec": None,
            "log_path": str(log_path),
            "return_code": None,
        }
    )
    state["items"][notebook_name] = entry
    save_state(paths.state_json, state)
    write_summary(paths, state)

    cmd = notebook_command(paths, notebook_name)
    append_monitor_line(paths, f"START {notebook_name}")
    append_monitor_line(paths, "CMD " + " ".join(cmd))

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[{now_iso()}] START {notebook_name}\n")
        log_file.write(f"[{now_iso()}] CMD {' '.join(cmd)}\n")

        proc = subprocess.Popen(
            cmd,
            cwd=str(paths.src_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            stamped = f"[{now_iso()}] {line.rstrip()}"
            log_file.write(stamped + "\n")
            log_file.flush()

        return_code = proc.wait()

    ended_at = now_iso()
    duration = time.time() - start_ts
    entry["ended_at"] = ended_at
    entry["duration_sec"] = duration
    entry["return_code"] = return_code
    entry["status"] = "completed" if return_code == 0 else "failed"
    save_state(paths.state_json, state)
    write_summary(paths, state)

    if return_code == 0:
        append_monitor_line(paths, f"DONE {notebook_name} ({duration / 60:.1f} min)")
        return True

    append_monitor_line(paths, f"FAIL {notebook_name} (return_code={return_code})")
    if continue_on_error:
        append_monitor_line(paths, "CONTINUE_ON_ERROR enabled, moving to next notebook.")
        return True
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full article training notebook pipeline.")
    p.add_argument(
        "--notebooks",
        nargs="*",
        default=None,
        help="Optional explicit notebook names from src/ to execute in order.",
    )
    p.add_argument("--resume", action="store_true", help="Skip notebooks already completed in state file.")
    p.add_argument("--continue-on-error", action="store_true", help="Continue pipeline if a notebook fails.")
    p.add_argument("--max-notebooks", type=int, default=0, help="Run only first N notebooks (0 = all).")
    p.add_argument("--dry-run", action="store_true", help="Validate and print plan without executing notebooks.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = build_paths()
    configure_cache_dirs(paths)

    notebooks = args.notebooks if args.notebooks else DEFAULT_NOTEBOOKS
    if args.max_notebooks and args.max_notebooks > 0:
        notebooks = notebooks[: args.max_notebooks]

    validate_dataset_paths(paths, notebooks)
    state = load_state(paths.state_json)
    state["order"] = notebooks
    state["torch_runtime"] = detect_torch_runtime()
    save_state(paths.state_json, state)
    write_summary(paths, state)

    append_monitor_line(paths, "Pipeline initialized.")
    append_monitor_line(paths, f"Torch runtime: {state.get('torch_runtime')}")
    append_monitor_line(paths, f"Notebook count: {len(notebooks)}")

    if args.dry_run:
        for nb in notebooks:
            append_monitor_line(paths, f"DRY_RUN {nb}")
        return 0

    for nb in notebooks:
        if args.resume and state["items"].get(nb, {}).get("status") == "completed":
            append_monitor_line(paths, f"SKIP {nb} (already completed)")
            continue

        ok = run_one_notebook(
            paths=paths,
            state=state,
            notebook_name=nb,
            continue_on_error=args.continue_on_error,
        )
        if not ok:
            append_monitor_line(paths, "Stopping pipeline due to error.")
            return 1

    append_monitor_line(paths, "Pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
