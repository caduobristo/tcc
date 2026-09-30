#!/usr/bin/env python
"""Keep Windows awake during long-running training jobs."""

from __future__ import annotations

import argparse
import ctypes
import time
from datetime import datetime
from pathlib import Path


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def touch_keep_awake() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    )


def release_keep_awake() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prevent Windows sleep while running.")
    parser.add_argument("--log-file", required=True, help="Path to append heartbeat log.")
    parser.add_argument("--interval-sec", type=int, default=60, help="Heartbeat interval.")
    args = parser.parse_args()

    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        touch_keep_awake()
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] keep_awake started\n")
        while True:
            touch_keep_awake()
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"[{now_iso()}] keep_awake heartbeat\n")
            time.sleep(max(10, args.interval_sec))
    except KeyboardInterrupt:
        pass
    finally:
        release_keep_awake()
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] keep_awake stopped\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
