"""Match audited mel16 ZIP members to the original WAV/source partitions."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from src.data.archive import GuitarFxArchive
from src.data.paths import data_root
from src.data.transfer import EFFECTS, manifest_path, read_manifest, sha256
from src.training.consolidation import dataset_view


def reconcile_features(scenario):
    root = data_root()
    rows = read_manifest(scenario)
    report = json.loads(manifest_path(scenario).with_suffix('.json').read_text())
    if report['manifest_sha256'] != sha256(manifest_path(scenario)) or report['seed'] != 42:
        raise ValueError('Original source-group manifest changed')
    _, view = dataset_view(rows, EFFECTS)
    expected = [(row['effect'], Path(row['filename']).stem) for row in rows]
    if len(set(expected)) != len(expected):
        raise ValueError('Ambiguous WAV/effect identity')
    with GuitarFxArchive(scenario, 'mel16') as archive:
        available = list(archive.iter_samples())
        if len(set(available)) != len(available) or set(available) != set(expected):
            raise ValueError('Features and original WAV manifest differ; no samples may be dropped')
        archive_manifest = root/'manifests'/f'{archive.path.stem}.csv'
        with archive_manifest.open(encoding='utf-8', newline='') as stream:
            entries = {row['member']: row for row in csv.DictReader(stream)}
        members = [f'{archive.scenario}/{effect}/{name}.npy' for effect, name in expected]
        checks = {}
        for member in members:
            reference, current = entries[member], archive.zip.getinfo(member)
            if (current.file_size != int(reference['bytes'])
                    or current.CRC != int(reference['crc32'], 16)):
                raise ValueError('ZIP member differs from its full original integrity audit')
            checks[member] = reference['sha256']
        metadata = {'manifest_sha256': report['manifest_sha256'], 'dataset_view': view,
                    'feature_manifest_sha256': sha256(archive_manifest),
                    'archive': archive.path.relative_to(root).as_posix(),
                    'archive_bytes': archive.path.stat().st_size,
                    'feature_shape': [198, 128], 'feature_dtype': 'float32',
                    'matched_samples': len(rows), 'matching_rule': 'effect + exact WAV stem',
                    'source_partitions_preserved': True, 'feature_sha256_verified_on_read': True}
    return rows, members, checks, metadata


def read_checked_feature(archive, member, expected_sha256):
    payload = archive.zip.read(member)  # ZipFile also verifies CRC.
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f'Feature changed since the full audit: {member}')
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if value.shape != (198, 128) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f'Invalid mel16 values: {member}')
    return value
