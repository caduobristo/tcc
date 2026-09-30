import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch import nn

from src.data.transfer import GuitarWaveforms, source_group, group_splits
from src.models.transfer import FrozenProbe
from src.training.linear_probe import extract_embeddings, train_probe


class PreparationTests(unittest.TestCase):
    def test_source_id_preserves_id_and_ignores_effect_settings(self):
        a = source_group("G61-40100-808-O2.1T10-20593.wav", "808")
        b = source_group("G61-40100-DS1-D10T0-20593.wav", "DS1")
        c = source_group("G61-40100-808-O2.1T10-99999.wav", "808")
        self.assertEqual(a, "G61-40100-1111-20593")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_split_is_order_independent_and_never_splits_a_source(self):
        groups = [str(i) for i in range(100)]
        result = group_splits(groups + groups)
        self.assertEqual(result, group_splits(list(reversed(groups))))
        self.assertEqual([list(result.values()).count(s) for s in ["train", "validation", "test"]], [72, 8, 20])

    def test_training_mode_preserves_frozen_bn_and_gradients_only_reach_head(self):
        encoder = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4))
        model = FrozenProbe(encoder, classes=3, embedding_dim=4).train()
        before = encoder[0].running_mean.clone()
        model(torch.randn(2, 4)).sum().backward()
        self.assertTrue(model.head.training)
        self.assertFalse(encoder.training)
        self.assertTrue(torch.equal(before, encoder[0].running_mean))
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        self.assertIsNotNone(model.head.weight.grad)

    def test_long_running_stages_are_disabled_by_default(self):
        for operation in (extract_embeddings, train_probe):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                operation({})

    def test_wav_integrity_rechecked_and_resampling_keeps_two_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.wav"
            sf.write(path, np.zeros(88201, dtype=np.float32), 44100)
            row = {"path": "sample.wav", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "label": "2", "sample_rate": "44100", "frames": "88201"}
            with patch("src.data.transfer.data_root", return_value=root):
                dataset = GuitarWaveforms([row])
            waveform, label, index = dataset[0]
            self.assertEqual(waveform.shape, (64000,))
            self.assertEqual((label, index), (2, 0))
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "changed since audit"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
