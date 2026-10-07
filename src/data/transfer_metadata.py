"""Align official settings with restored WAVs, retaining existing audited CSVs."""
import csv
import json
import math
from pathlib import Path

from src.data.paths import SCENARIOS, data_root
from src.data.transfer import EFFECTS, sha256


def ensure_validated_metadata(scenario):
    root = data_root()
    name = SCENARIOS[scenario]
    restored = root/'raw/guitar_fx_dist'/name
    details = {'scenario':scenario,'original_files_modified':False,'effects':{}}
    for effect in EFFECTS:
        source = restored/'Features'/effect/'proc_settings.csv'
        target = root/'metadata/guitar_fx_dist/validated'/name/effect/'proc_settings.csv'
        with source.open(encoding='utf-8-sig',newline='') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            original = list(reader)
        if not columns or not {'filename','fx'} <= set(columns):
            raise ValueError('Official metadata schema mismatch')
        by_name = {}
        duplicates = 0
        for row in original:
            filename = row['filename']
            if Path(filename).name!=filename or not filename.endswith('.wav') or row['fx']!=effect or None in row:
                raise ValueError('Invalid original filename/effect/schema')
            for key,value in row.items():
                if key not in ('filename','fx') and value not in (None,''):
                    value = float(value)
                    if not math.isfinite(value) or not 0<=value<=1:
                        raise ValueError('Invalid original parameter')
            if filename in by_name:
                if row!=by_name[filename]:
                    raise ValueError('Conflicting official settings for one filename')
                duplicates += 1
            by_name[filename] = row
        wav_names = {p.name for p in (restored/'Audio'/effect).rglob('*.wav')}
        if target.exists():
            # Preserve previously audited selection and its exact bytes.
            with target.open(encoding='utf-8',newline='') as stream:
                selected = list(csv.DictReader(stream))
            if len({r['filename'] for r in selected})!=len(selected):
                raise ValueError('Existing validated metadata repeats a filename')
            for row in selected:
                if row['filename'] not in wav_names or row['filename'] not in by_name or row['fx']!=effect:
                    raise ValueError('Existing audited metadata differs from the official WAV source')
                for key,value in row.items():
                    source_value = by_name[row['filename']].get(key)
                    if (value or '') != (source_value or ''):
                        raise ValueError('Existing validated settings differ from original source')
            preserved = True
        else:
            selected = [row for filename,row in sorted(by_name.items()) if filename in wav_names]
            if not selected:
                raise ValueError('No aligned original WAVs/settings')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('x',encoding='utf-8',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=columns)
                writer.writeheader()
                writer.writerows(selected)
            preserved = False
        details['effects'][effect] = {'original_csv_sha256':sha256(source),'validated_csv_sha256':sha256(target),
            'selected':len(selected),'existing_validated_csv_preserved':preserved,
            'exact_duplicate_original_rows':duplicates,
            'original_names_without_wav':sorted(set(by_name)-wav_names),
            'restored_wavs_outside_selection':len(wav_names-{r['filename'] for r in selected})}
    output = root/'audits/transfer_learning'/f'{scenario}_metadata_alignment.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(details,indent=2),encoding='utf-8')
    return details
