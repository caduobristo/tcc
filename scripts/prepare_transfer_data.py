"""Restore official GUITAR-FX WAVs; never start a model or training loop."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.paths import SCENARIOS, data_root

RECORDS = {"mono_disc": "4298000", "mono_cont": "4296040",
           "poly_cont": "4298017", "poly_disc": "4298025"}


def file_hash(path, algorithm="md5"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(file, folder):
    """Resume only a correctly served byte range; verify the published checksum."""
    name = file["key"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Unsafe source filename")
    dest = folder / name
    expected = file["checksum"].split(":", 1)
    if dest.exists():
        if dest.stat().st_size != file["size"] or file_hash(dest, expected[0]) != expected[1]:
            raise ValueError(f"Existing source fails checksum: {dest}")
        print(f"Verified existing {name}", flush=True)
        return
    partial = dest.with_name(dest.name + ".partial")
    for attempt in range(8):
        start = partial.stat().st_size if partial.exists() else 0
        if start > file["size"]:
            raise ValueError(f"Oversized partial download: {partial}")
        try:
            last_report = time.monotonic()
            while start < file["size"]:
                headers = {"User-Agent": "TCC-dataset-preparation/1.0"}
                end = min(start + 32 * 1024 * 1024, file["size"]) - 1
                headers["Range"] = f"bytes={start}-{end}"
                request = urllib.request.Request(file["links"]["self"], headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{file['size']}":
                        raise ValueError("Server did not honor resume offset")
                    with partial.open("ab" if start else "wb") as stream:
                        shutil.copyfileobj(response, stream, length=256 * 1024)
                start = partial.stat().st_size
                if time.monotonic() - last_report > 30:
                    print(f"{name}: {start/1e9:.2f}/{file['size']/1e9:.2f} GB", flush=True)
                    last_report = time.monotonic()
            if partial.stat().st_size != file["size"]:
                raise OSError("Incomplete response")
            if file_hash(partial, expected[0]) != expected[1]:
                raise ValueError(f"Published checksum mismatch: {partial}")
            partial.rename(dest)
            print(f"Downloaded and verified {name}", flush=True)
            return
        except ValueError:
            raise
        except Exception as exc:
            print(f"Retry {attempt + 1}: {name}: {exc}", flush=True)
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30))


def prepare(scenario, extract=False, workers=3, seven_zip=None):
    root = data_root()
    record = RECORDS[scenario]
    folder = root / "archives/guitar_fx_dist/official" / SCENARIOS[scenario]
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "source_record.json"
    if cached.exists():
        source = json.loads(cached.read_text(encoding="utf-8"))
        if str(source["id"]) != record:
            raise ValueError("Cached Zenodo record ID mismatch")
    else:
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as response:
            source = json.load(response)
    files = sorted([f for f in source["files"] if f["key"].startswith(SCENARIOS[scenario] + ".") or f["key"] == "README.md"], key=lambda f: f["key"])
    required = sum(f["size"] for f in files if not (folder / f["key"]).exists())
    # Reserve space for extraction and the environment, not just the archives.
    if shutil.disk_usage(root).free < required + 45 * 2 ** 30:
        raise OSError("Insufficient disk space: preserve 45 GiB beyond source downloads")
    (folder / "source_record.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(f"Restoring {SCENARIOS[scenario]} from Zenodo {record}: {sum(f['size'] for f in files)/1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda f: download(f, folder), files))
    if extract:
        binary = seven_zip or shutil.which("7z") or "C:/Program Files/7-Zip/7z.exe"
        if not Path(binary).is_file():
            raise FileNotFoundError("7-Zip is required to extract the split ZIP")
        archive = folder / (SCENARIOS[scenario] + ".zip")
        listing = subprocess.run([binary, "l", "-slt", str(archive)], capture_output=True, text=True, encoding="utf-8", check=True).stdout
        members = [line[7:] for line in listing.splitlines() if line.startswith("Path = ")][1:]
        output = (root / "raw/guitar_fx_dist").resolve()
        for member in members:
            target = (output / member).resolve()
            if not target.is_relative_to(output):
                raise ValueError(f"Unsafe archive path: {member}")
        output.mkdir(parents=True, exist_ok=True)
        # Only original WAVs/settings; leave the historical baseline features in the source archive.
        subprocess.run([binary, "x", str(archive), f"-o{output}", "-aos", "-bsp0",
                        "-ir!*.wav", "-ir!proc_settings.csv"], check=True)
        print(f"WAVs restored to {output}. No training started.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=list(RECORDS), default="mono_disc")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seven-zip")
    args = parser.parse_args()
    prepare(args.scenario, args.extract, args.workers, args.seven_zip)
