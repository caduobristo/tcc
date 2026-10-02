"""Regression checks for scenario isolation and safe selected source extraction."""
import json
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from src.data.paths import SCENARIOS
from src.data.transfer import EFFECTS
from src.training.consolidation import load_selection_data, validate_plan
from scripts.prepare_additional_transfer_data import selected_members, remaining_download_bytes, RequestPacer, retry_after_seconds, download_ranges
from src.data.transfer_metadata import ensure_validated_metadata

ROOT = Path(__file__).resolve().parents[1]


class ScenarioTests(unittest.TestCase):
    def test_grouped_ranges_reuse_blocks_and_verify_the_complete_source_checksum(self):
        payload = b'abcdefghijklmnopqrst'
        source = {'key':'example.zip','size':len(payload),'checksum':'md5:'+hashlib.md5(payload).hexdigest(),
                  'links':{'self':'https://example.invalid/official-volume'}}
        requests = []
        def response(request,timeout):
            value = request.get_header('Range')
            start,end = map(int,value.removeprefix('bytes=').split('-'))
            requests.append((start,end))
            stream = io.BytesIO(payload[start:end+1])
            stream.status = 206
            stream.headers = {'Content-Range':f'bytes {start}-{end}/{len(payload)}'}
            return stream
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks = root/'example.zip.blocks'
            chunks.mkdir()
            (chunks/'00000.part').write_bytes(payload[:4])
            (chunks/'00002.part').write_bytes(payload[8:12])
            with patch('scripts.prepare_additional_transfer_data.urllib.request.urlopen',side_effect=response), \
                 patch('scripts.prepare_additional_transfer_data.RANGE_PACER'):
                download_ranges(source,root,connections=2,block_bytes=4)
            self.assertEqual(sorted(requests),[(4,7),(12,19)])
            self.assertEqual((root/'example.zip').read_bytes(),payload)
            self.assertFalse(chunks.exists())
            (root/'example.zip').unlink()
            chunks.mkdir()
            (chunks/'00000.part').write_bytes(b'XXXX')  # right length, corrupt cached bytes
            with patch('scripts.prepare_additional_transfer_data.urllib.request.urlopen',side_effect=response), \
                 patch('scripts.prepare_additional_transfer_data.RANGE_PACER'):
                with self.assertRaisesRegex(ValueError,'Published checksum mismatch'):
                    download_ranges(source,root,connections=2,block_bytes=4)
            self.assertFalse((root/'example.zip').exists())
            self.assertTrue(chunks.exists())  # preserve evidence after failed integrity

    def test_request_spacing_and_server_cooldown_are_shared(self):
        now = [100.]
        def sleep(delay):
            now[0] += delay
        with patch('scripts.prepare_additional_transfer_data.time.monotonic',side_effect=lambda:now[0]), \
             patch('scripts.prepare_additional_transfer_data.time.sleep',side_effect=sleep):
            pacer = RequestPacer(interval=1.25)
            pacer.wait()
            pacer.wait()
            self.assertEqual(now[0],101.25)
            pacer.defer(60)
            pacer.wait()
            self.assertEqual(now[0],161.25)
        self.assertEqual(retry_after_seconds({'Retry-After':'90'},0),90)
        self.assertEqual(retry_after_seconds({},0),60)
        self.assertEqual(retry_after_seconds({},8),600)

    def test_disk_budget_counts_resumable_blocks_without_double_counting_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = [{'key':'example.zip','size':25}]
            self.assertEqual(remaining_download_bytes(files,root,block_bytes=10),25)
            blocks = root/'example.zip.blocks'
            blocks.mkdir()
            (blocks/'00000.part').write_bytes(b'0'*10)
            (root/'example.zip.partial').write_bytes(b'0'*20)
            (blocks/'00002.part').write_bytes(b'0'*4)  # invalid last block, still missing
            self.assertEqual(remaining_download_bytes(files,root,block_bytes=10),5)
            (root/'example.zip').write_bytes(b'0'*25)
            self.assertEqual(remaining_download_bytes(files,root,block_bytes=10),0)
            (root/'example.zip').write_bytes(b'0'*24)
            with self.assertRaises(ValueError):
                remaining_download_bytes(files,root,block_bytes=10)

    def test_all_protocols_keep_the_same_search_and_confirmation_budget(self):
        original = json.loads((ROOT/'configs/linear_probe/consolidation_mono_disc.json').read_text())
        for scenario in SCENARIOS:
            plan = json.loads((ROOT/f'configs/linear_probe/consolidation_{scenario}.json').read_text())
            validate_plan(plan)
            for key in original:
                if key not in ('scenario','historical_test_already_inspected'):
                    self.assertEqual(plan[key],original[key])
        original['scenario'] = '../../mono_disc'
        with self.assertRaises(ValueError):
            validate_plan(original)

    def test_selection_loads_the_requested_scenario_and_never_test_vectors(self):
        for scenario in SCENARIOS:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                vectors = root/'vectors.npy'
                # A test-only NaN sentinel must not appear in train/validation tensors.
                array = np.vstack([np.zeros((13,768),np.float32),np.ones((13,768),np.float32),np.full((13,768),np.nan,np.float32)])
                np.save(vectors,array)
                rows = [{'label':str(i),'effect':effect,'source_group':f'{split}-{i}','split':split}
                        for split in ('train','validation','test') for i,effect in enumerate(EFFECTS)]
                manifest = root/'samples.csv'
                manifest.with_suffix('.json').write_text(json.dumps({'scenario':scenario,'seed':42,'manifest_sha256':'sha','classes':list(EFFECTS)}))
                config = {'device':'cpu'}
                with patch('src.training.consolidation.load_config',return_value=config) as config_read, \
                     patch('src.training.consolidation.manifest_path',return_value=manifest) as manifest_read, \
                     patch('src.training.consolidation.sha256',return_value='sha'), \
                     patch('src.training.consolidation.validate_cache',return_value={'identity':{'scenario':scenario}}), \
                     patch('src.training.consolidation.paths_for',return_value=(root,vectors,root/'cache.json')), \
                     patch('src.training.consolidation.read_manifest',return_value=rows) as rows_read:
                    data = load_selection_data('passt',{'scenario':scenario,'split_seed':42})
                    self.assertEqual(config_read.call_args.args[0].name,f'passt_{scenario}.json')
                    rows_read.assert_called_once_with(scenario)
                    self.assertTrue(all(call.args==(scenario,) for call in manifest_read.call_args_list))
                    self.assertEqual(tuple(data.train_x.shape),(13,768))
                    self.assertTrue(bool(data.validation_x.isfinite().all()))

    def test_archive_selection_excludes_unrequested_audio_and_features(self):
        blocks = [f'Path = Mono_Continuous/Audio/{effect}/sample.wav\nFolder = -\nSize = 100\nCRC = 12345678' for effect in EFFECTS]
        blocks += ['Path = Mono_Continuous/Audio/MT2/sample.wav\nFolder = -\nSize = 100\nCRC = 12345678',
                   'Path = Mono_Continuous/Features/808/sample.npy\nFolder = -\nSize = 100\nCRC = 12345678']
        listing = 'archive header\n----------\n'+'\n\n'.join(blocks)
        self.assertEqual(len(selected_members(listing,'Mono_Continuous')),13)
        with self.assertRaises(ValueError):
            selected_members(listing+'\n\nPath = Mono_Continuous/../unsafe.wav\nFolder = -\nSize = 1\nCRC = 12345678','Mono_Continuous')

    def test_official_metadata_alignment_deduplicates_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for effect in EFFECTS:
                source = root/'raw/guitar_fx_dist/Mono_Continuous/Features'/effect/'proc_settings.csv'
                source.parent.mkdir(parents=True)
                source.write_text(f'filename,fx,level,gain\na.wav,{effect},1.0,0.5\na.wav,{effect},1.0,0.5\nmissing.wav,{effect},1.0,0.2\n')
                wav = root/'raw/guitar_fx_dist/Mono_Continuous/Audio'/effect/'a.wav'
                wav.parent.mkdir(parents=True)
                wav.write_bytes(b'filename alignment fixture; waveform validation belongs to the index')
            with patch('src.data.transfer_metadata.data_root',return_value=root):
                first = ensure_validated_metadata('mono_cont')
                second = ensure_validated_metadata('mono_cont')
                for effect in EFFECTS:
                    self.assertEqual(first['effects'][effect]['selected'],1)
                    self.assertEqual(first['effects'][effect]['exact_duplicate_original_rows'],1)
                    self.assertEqual(first['effects'][effect]['original_names_without_wav'],['missing.wav'])
                    self.assertTrue(second['effects'][effect]['existing_validated_csv_preserved'])
                    self.assertEqual(first['effects'][effect]['validated_csv_sha256'],second['effects'][effect]['validated_csv_sha256'])


if __name__=='__main__':
    unittest.main()
