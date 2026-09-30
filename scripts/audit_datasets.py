"""Audit local GUITAR-FX archives or materialize one explicit representation.

Run from any directory. Raw data, manifests and audit reports stay outside Git.
No datasets are downloaded and existing files are never silently overwritten.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from datetime import date
from zipfile import ZipFile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.paths import SCENARIOS, data_root


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_write(path, content):
    """Only create a file, or verify that an existing file has identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != hashlib.sha256(content).hexdigest():
            raise FileExistsError(f"Different existing contents: {path}")
        return
    with path.open("xb") as stream:
        stream.write(content)


def read_metadata(root, extract=False):
    path = root / "archives/guitar_fx_dist/csvs.zip"
    tables, reports = {}, []
    with ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if len(parts) != 3 or not parts[0].endswith("_CSVs") or parts[-1] != "proc_settings.csv":
                raise ValueError(f"Unexpected metadata member: {member.filename}")
            scenario = parts[0][:-5]
            effect = parts[1]
            if scenario not in SCENARIOS.values() or effect in {".", ".."}:
                raise ValueError(f"Invalid metadata location: {member.filename}")
            content = archive.read(member)  # zipfile checks the complete CRC.
            rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"), newline="")))
            header = rows.pop(0)
            if header[:2] != ["filename", "fx"]:
                raise ValueError(f"Unexpected CSV header: {member.filename}")
            stems, duplicates, issues = set(), 0, []
            row_count = 0
            for row in rows:
                if not row:
                    continue
                row_count += 1
                if len(row) < 2:
                    issues.append("row with fewer than two fields")
                    continue
                stem = Path(row[0]).stem
                duplicates += stem in stems
                stems.add(stem)
                if row[1] != effect:
                    issues.append("effect label differs from folder")
                try:
                    values = [float(v) for v in row[2:] if v.strip()]
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                        issues.append("non-finite or out-of-range parameter")
                except ValueError:
                    issues.append("non-numeric parameter")
            tables[(scenario, effect)] = stems
            reports.append({"scenario": scenario, "effect": effect, "rows": row_count,
                            "unique_filenames": len(stems), "duplicate_rows": duplicates,
                            "issues": dict(Counter(issues)), "sha256": hashlib.sha256(content).hexdigest()})
            if extract:
                checked_write(root / "metadata/guitar_fx_dist" / scenario / effect / "proc_settings.csv", content)
    return tables, {"archive": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
                    "crc_checked_members": len(reports), "tables": reports}


def audit_archive(path, root, tables):
    started = time.monotonic()
    print(f"START {path.name}: SHA-256", flush=True)
    result = {"archive": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
              "sha256": sha256_file(path), "errors": [], "validation": "full_crc_numpy_finite_sha256"}
    scenario = SCENARIOS[path.name.split("_mel", 1)[0]]
    variant = "mel" + path.name.split("_mel", 1)[1].split("-", 1)[0]
    result.update(scenario=scenario, variant=variant)
    classes, shapes, dtypes, hashes = Counter(), Counter(), Counter(), Counter()
    present = {}
    metadata_missing = Counter()
    digest = hashlib.sha256()
    minimum, maximum, checked, nonfinite = math.inf, -math.inf, 0, 0
    manifest = root / "manifests" / f"{path.stem}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path) as archive, manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["member", "bytes", "crc32", "sha256", "shape", "dtype", "source_id"])
        members = [m for m in archive.infolist() if not m.is_dir()]
        names = [m.filename for m in members]
        if len(set(names)) != len(names):
            result["errors"].append("Duplicate ZIP member names")
        for member in members:
            parts = PurePosixPath(member.filename).parts
            if len(parts) != 3 or parts[0] != scenario or ".." in parts or not parts[-1].endswith(".npy"):
                result["errors"].append(f"Unexpected member: {member.filename}")
                continue
            effect, filename = parts[1:]
            stem = filename[:-4]
            present.setdefault(effect, set()).add(stem)
            classes[effect] += 1
            try:
                content = archive.read(member)
                content_hash = hashlib.sha256(content).hexdigest()
                hashes[content_hash] += 1
                array = np.load(io.BytesIO(content), allow_pickle=False)
                shapes[str(array.shape)] += 1
                dtypes[str(array.dtype)] += 1
                if array.shape != (198, 128) or array.dtype != np.float32:
                    result["errors"].append(f"Unexpected array format: {member.filename}")
                finite = np.isfinite(array).all()
                if not finite:
                    nonfinite += 1
                    result["errors"].append(f"Non-finite values: {member.filename}")
                else:
                    minimum = min(minimum, float(array.min()))
                    maximum = max(maximum, float(array.max()))
                if not effect.startswith("_NoFX") and stem not in tables.get((scenario, effect), set()):
                    metadata_missing[effect] += 1
                source_id = stem.rsplit("-", 1)[-1]
                writer.writerow([member.filename, member.file_size, f"{member.CRC:08x}",
                                 content_hash, str(array.shape), str(array.dtype), source_id])
                digest.update(f"{member.filename}|{content_hash}\n".encode())
                checked += 1
                if checked % 25000 == 0:
                    print(f"CHECK {path.name}: {checked}/{len(members)}", flush=True)
            except Exception as exc:
                result["errors"].append(f"{member.filename}: {type(exc).__name__}: {exc}")
    missing_features = {}
    for (sc, effect), stems in tables.items():
        if sc == scenario:
            missing_features[effect] = len(stems - present.get(effect, set()))
    duplicate_groups = sum(n > 1 for n in hashes.values())
    duplicate_extra = sum(n - 1 for n in hashes.values() if n > 1)
    result.update(checked_members=checked, class_counts=dict(classes), shapes=dict(shapes),
                  dtypes=dict(dtypes), nonfinite_arrays=nonfinite,
                  minimum=minimum if math.isfinite(minimum) else None,
                  maximum=maximum if math.isfinite(maximum) else None,
                  feature_without_metadata=dict(metadata_missing), metadata_without_feature=missing_features,
                  duplicate_content_groups=duplicate_groups, duplicate_content_extra_files=duplicate_extra,
                  member_hash_manifest=manifest.relative_to(root).as_posix(),
                  content_fingerprint=digest.hexdigest(), elapsed_seconds=round(time.monotonic()-started, 2))
    result["integrity_ok"] = not result["errors"] and not metadata_missing and not any(missing_features.values())
    out = root / "audits" / f"{path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"DONE {path.name}: {checked} arrays, integrity_ok={result['integrity_ok']}", flush=True)
    return result


def audit(root, workers):
    tables, metadata = read_metadata(root, extract=True)
    paths = sorted((root / "archives/guitar_fx_dist").glob("*_mel*-*.zip"))
    expected = {(s, v) for s in SCENARIOS.values() for v in ("mel16", "mel32")}
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = [executor.submit(audit_archive, path, root, tables) for path in paths]
        for future in as_completed(tasks):
            results.append(future.result())
    found = [(r["scenario"], r["variant"]) for r in results]
    metadata_ok = all(not r["duplicate_rows"] and not r["issues"] for r in metadata["tables"])
    report = {"audited_on": date.today().isoformat(), "metadata": metadata,
              "archives": sorted(results, key=lambda r: r["archive"]),
              "missing_scenario_variants": sorted(expected-set(found)),
              "duplicate_scenario_variants": len(found) != len(set(found)),
              "integrity_ok": metadata_ok and bool(results) and not (expected-set(found))
              and len(found) == len(set(found)) and all(r["integrity_ok"] for r in results)}
    output = root / "audits/integrity_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {output}; integrity_ok={report['integrity_ok']}", flush=True)
    reconcile(root)
    return 0 if report["integrity_ok"] else 1


def reconcile(root):
    """Build derived, unambiguous CSVs; keep all original metadata and anomalies."""
    report_path = root / "audits/integrity_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["missing_scenario_variants"] or report["duplicate_scenario_variants"]:
        raise ValueError("Cannot derive metadata from missing or ambiguous scenario variants.")
    if not report["archives"] or any(r["errors"] for r in report["archives"]):
        raise ValueError("Cannot derive metadata from archives with binary or array errors.")
    feature_sets, comparisons, duplicates = {}, [], []
    for record in report["archives"]:
        scenario, variant = record["scenario"], record["variant"]
        effects, payloads = {}, {}
        fingerprint = hashlib.sha256()
        manifest = root / record["member_hash_manifest"]
        with manifest.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                parts = PurePosixPath(row["member"]).parts
                effects.setdefault(parts[1], set()).add(Path(parts[-1]).stem)
                payloads.setdefault(row["sha256"], []).append(row["member"])
                fingerprint.update(f"{row['member']}|{row['sha256']}\n".encode())
        if fingerprint.hexdigest() != record["content_fingerprint"]:
            raise ValueError(f"Audited member manifest changed: {manifest}. Run a full audit again.")
        existing = feature_sets.get(scenario)
        if existing is not None and existing != effects:
            raise ValueError(f"Variant filename sets differ for {scenario}; metadata needs a separate policy.")
        feature_sets[scenario] = effects
        duplicate_members = [members for members in payloads.values() if len(members) > 1]
        duplicates.append({"scenario": scenario, "variant": variant, "groups": duplicate_members})
        record["binary_integrity_ok"] = not record["errors"] and record["checked_members"] == sum(record["class_counts"].values())
        record["metadata_alignment_ok"] = not record["feature_without_metadata"] and not any(record["metadata_without_feature"].values())
        comparisons.append({"scenario": scenario, "variant": variant, "hashes": payloads})
    metadata_reports, rejected = [], []
    for path in sorted((root / "metadata/guitar_fx_dist").glob("*/*/proc_settings.csv")):
        scenario, effect = path.parts[-3:-1]
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [tuple(row) for row in csv.reader(stream) if row]
        header, rows = rows[0], rows[1:]
        by_name = {}
        for row in rows:
            by_name.setdefault(row[0], set()).add(row)
        selected, missing, conflicts = [], [], []
        names = feature_sets.get(scenario, {}).get(effect, set())
        for filename, values in sorted(by_name.items()):
            if len(values) > 1:
                conflicts.append(filename)
                rejected.append({"scenario": scenario, "effect": effect, "filename": filename, "reason": "conflicting rows"})
            elif Path(filename).stem not in names:
                missing.append(filename)
                rejected.append({"scenario": scenario, "effect": effect, "filename": filename, "reason": "feature absent in both variants"})
            else:
                selected.append(next(iter(values)))
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(header)
        writer.writerows(selected)
        output = root / "metadata/guitar_fx_dist/validated" / scenario / effect / "proc_settings.csv"
        checked_write(output, buffer.getvalue().encode("utf-8"))
        metadata_reports.append({"scenario": scenario, "effect": effect, "original_rows": len(rows),
                                 "exact_duplicate_rows": len(rows)-len(set(rows)),
                                 "conflicting_filenames": conflicts, "missing_feature_filenames": missing,
                                 "validated_rows": len(selected), "validated_sha256": sha256_file(output)})
    variant_comparison = []
    for scenario in sorted(feature_sets):
        pair = [r for r in comparisons if r["scenario"] == scenario]
        if len(pair) == 2:
            left, right = pair
            shared = set(left["hashes"]) & set(right["hashes"])
            same_named_payload = sum(
                len(set(left["hashes"][h]) & set(right["hashes"][h])) for h in shared
            )
            variant_comparison.append({"scenario": scenario, "same_named_payloads": same_named_payload,
                                       "same_filename_set": True, "variants_are_byte_identical":
                                       same_named_payload == sum(len(v) for v in left["hashes"].values())})
    consistency = {"audited_on": report["audited_on"], "metadata": metadata_reports, "rejected_original_rows": rejected,
                   "duplicate_array_groups": duplicates, "variant_comparison": variant_comparison,
                   "source_files_modified": False}
    (root / "audits/metadata_consistency.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")
    report["binary_integrity_ok"] = all(r["binary_integrity_ok"] for r in report["archives"])
    report["metadata_source_consistent"] = all(
        not r["exact_duplicate_rows"] and not r["conflicting_filenames"] and not r["missing_feature_filenames"]
        for r in metadata_reports
    )
    report["validated_metadata_rows"] = sum(r["validated_rows"] for r in metadata_reports)
    report["metadata_consistency_report"] = "audits/metadata_consistency.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Metadata: {sum(r['exact_duplicate_rows'] for r in metadata_reports)} exact duplicate rows; "
          f"{len(rejected)} rejected unique filenames; originals preserved.", flush=True)


def materialize(root, scenario, variant, max_per_class):
    """Preserve class names and all available classes; no baseline conversion."""
    from src.data.archive import GuitarFxArchive
    scenario = SCENARIOS.get(scenario, scenario)
    if not (root / "metadata/guitar_fx_dist/validated" / scenario).is_dir():
        raise FileNotFoundError("Run audit/reconcile first to obtain validated metadata.")
    with GuitarFxArchive(scenario, variant, root) as source:
        selected, counts = [], Counter()
        for effect, stem in source.iter_samples(exclude_effects=(), include_nofx=True):
            if max_per_class and counts[effect] >= max_per_class:
                continue
            selected.append((effect, stem))
            counts[effect] += 1
        total = sum(source.zip.getinfo(f"{scenario}/{effect}/{stem}.npy").file_size for effect, stem in selected)
        if shutil.disk_usage(root).free < total + 10 * 1024**3:
            raise OSError("Not enough disk space to extract and retain a 10 GiB reserve.")
        representation_root = root / "processed/guitar_fx_dist" / variant
        if max_per_class:
            representation_root = representation_root / "subsets" / f"first_{max_per_class}_per_class"
        destination = representation_root / scenario / "Features"
        for effect, stem in selected:
            # Full ZIP CRC and numpy format checks before creating the local file.
            source.read_feature(effect, stem)
            content = source.zip.read(f"{scenario}/{effect}/{stem}.npy")
            checked_write(destination / effect / "mel_198x128" / f"{stem}.npy", content)
        read_metadata(root, extract=True)
        for effect in counts:
            metadata = root / "metadata/guitar_fx_dist/validated" / scenario / effect / "proc_settings.csv"
            if metadata.exists():
                checked_write(destination / effect / "proc_settings.csv", metadata.read_bytes())
        record = {"scenario": scenario, "variant": variant, "source_archive": source.path.name,
                  "max_per_class": max_per_class, "class_counts": dict(counts), "feature_folder": "mel_198x128"}
        checked_write(destination.parent / "materialization.json", json.dumps(record, indent=2).encode())
    print(f"Materialized {len(selected)} features at {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_root())
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("audit", help="Check every array, ZIP CRC, SHA-256 and CSV association")
    check.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    commands.add_parser("reconcile", help="Derive metadata aligned with the fully audited feature manifests")
    extract = commands.add_parser("materialize", help="Extract one explicitly chosen variant/scenario")
    extract.add_argument("--scenario", choices=tuple(SCENARIOS)+tuple(SCENARIOS.values()), required=True)
    extract.add_argument("--variant", choices=("mel16", "mel32"), required=True)
    extract.add_argument("--max-per-class", type=int, default=0, help="0 means all samples")
    args = parser.parse_args()
    if args.command == "audit":
        return audit(args.data_root.resolve(), args.workers)
    if args.command == "reconcile":
        reconcile(args.data_root.resolve())
        return 0
    if args.max_per_class < 0:
        parser.error("--max-per-class must be nonnegative")
    materialize(args.data_root.resolve(), args.scenario, args.variant, args.max_per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
