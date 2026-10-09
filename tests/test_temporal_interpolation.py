"""Scientific checks for temporal adaptation and preservation of original inputs."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data.dataset_ast import FxDatasetAST
from src.data.spectrogram_resize import adapt_spectrogram_time
from src.models.temporal_resize import TemporallyResizedPasst


class TemporalInterpolationTests(unittest.TestCase):
    def test_matches_htsat_interpolation_before_image_folding(self):
        from src.vendor.htsat.htsat import HTSAT_Swin_Transformer
        x = torch.randn(2, 1, 201, 64)
        actual = HTSAT_Swin_Transformer.reshape_wav2img(SimpleNamespace(spec_size=256, freq_ratio=4), x)
        resized = adapt_spectrogram_time(x, 1024)
        folded = resized.permute(0, 1, 3, 2).contiguous().reshape(2, 1, 64, 4, 256)
        folded = folded.permute(0, 1, 3, 2, 4).contiguous().reshape(2, 1, 256, 256)
        torch.testing.assert_close(actual, folded, rtol=0, atol=0)

    def test_preserves_frequency_columns_and_separate_samples(self):
        frequencies = torch.arange(128, dtype=torch.float32)
        x = frequencies.repeat(2, 198, 1)
        x[1] += 1000
        y = adapt_spectrogram_time(x, 1024)
        torch.testing.assert_close(y[0], frequencies.repeat(1024, 1), atol=0.001, rtol=0)
        torch.testing.assert_close(y[1], (frequencies + 1000).repeat(1024, 1), atol=0.002, rtol=0)

    def test_preserves_endpoints_and_interpolates_instead_of_padding(self):
        x = torch.linspace(1, 2, 198)[:, None].repeat(1, 128)
        y = adapt_spectrogram_time(x, 1024)
        torch.testing.assert_close(y[0], x[0], rtol=0, atol=0)
        torch.testing.assert_close(y[-1], x[-1], rtol=0, atol=0)
        self.assertTrue(bool((y[198:] > 0).all()))
        self.assertTrue(torch.equal(x, torch.linspace(1, 2, 198)[:, None].repeat(1, 128)))

    def test_shared_ast_audiomae_loader_retains_default_and_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            effect = Path(folder) / "808"
            effect.mkdir()
            array = np.linspace(-2, 2, 198 * 128, dtype=np.float32).reshape(198, 128)
            path = effect / "sample.npy"
            np.save(path, array)
            before = path.read_bytes()
            old = FxDatasetAST(folder)
            old.init_dataset()
            new = FxDatasetAST(folder, temporal_adaptation="bicubic")
            new.init_dataset()
            normalized = (torch.from_numpy(array) + 4.27) / (4.57 * 2)
            torch.testing.assert_close(old[0][0], F.pad(normalized, (0, 0, 0, 826)), rtol=0, atol=0)
            torch.testing.assert_close(new[0][0], adapt_spectrogram_time(normalized, 1024), rtol=0, atol=0)
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(old[0][3:], new[0][3:])

    def test_rejects_invalid_input_and_unsupported_modes(self):
        for target in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                adapt_spectrogram_time(torch.zeros(198, 128), target)
        for x in (torch.zeros(0, 128), torch.zeros(128), torch.ones(2, 128, dtype=torch.long), torch.full((2, 128), float("nan"))):
            with self.assertRaises(ValueError):
                adapt_spectrogram_time(x, 1024)
        with self.assertRaises(ValueError):
            adapt_spectrogram_time(torch.zeros(198, 128), 1024, mode="nearest")

    def test_passt_uses_correct_orientation_without_changing_frontend(self):
        class Mel(nn.Module):
            def forward(self, wave):
                return torch.arange(128, dtype=torch.float32).reshape(1, 128, 1).repeat(len(wave), 1, 200)
        class Net(nn.Module):
            def forward_features(self, image):
                self.image = image
                return torch.ones(len(image), 768), torch.ones(len(image), 768) * 3
        base = nn.Module()
        base.mel, base.net = Mel(), Net()
        resized = TemporallyResizedPasst(base)
        resized.train()
        result = resized(torch.ones(2, 64000))
        self.assertEqual(tuple(base.net.image.shape), (2, 1, 128, 998))
        self.assertFalse(base.training)
        self.assertTrue(torch.equal(result, torch.full((2, 768), 2.0)))
        torch.testing.assert_close(base.net.image[:, 0, :, 500], torch.arange(128, dtype=torch.float32).repeat(2, 1), atol=0.001, rtol=0)


if __name__ == "__main__":
    unittest.main()
