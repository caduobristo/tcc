"""Frozen PaSST/HTS-AT encoders with a new single-label classification head."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import contextlib
import io
import hashlib

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_ROOT = PROJECT_ROOT / "checkpoints/transfer_learning/pretrained"
WEIGHTS = {"passt": "passt-s-f128-p16-s10-ap.476-swa.pt", "htsat": "HTSAT_AudioSet_Saved_1.ckpt"}
WEIGHT_SHA256 = {"passt": "302903fa8c4aee817b11dc982da0b29aaf8d11a3e722420476d0a12c9db70c2c",
                 "htsat": "49804bf7767809be5d5bfe9dc25f401442e0d24f242dded6b6dabea56bb645b0"}


def load_frozen_encoder(model_name):
    if model_name not in WEIGHTS:
        raise ValueError("Choose passt or htsat")
    path = WEIGHTS_ROOT / WEIGHTS[model_name]
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weights missing: {path}; run scripts/prepare_transfer_models.py")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHT_SHA256[model_name]:
        raise ValueError(f"Checkpoint differs from the pinned official download: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if model_name == "passt":
        from hear21passt.models.passt import get_model
        from hear21passt.models.preprocess import AugmentMelSTFT
        with contextlib.redirect_stdout(io.StringIO()):
            net = get_model(arch="passt_s_swa_p16_128_ap476", pretrained=False,
                            n_classes=527, input_tdim=998, s_patchout_t=0, s_patchout_f=0, u_patchout=0)
        net.load_state_dict(state, strict=True)
        mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024,
                             freqm=0, timem=0, htk=False, fmin=0., fmax=None, norm=1,
                             fmin_aug_range=10, fmax_aug_range=2000)
        encoder = PasstEncoder(net, mel)
        # Upstream's debug flag is otherwise reset only by its classification forward.
        import hear21passt.models.passt as passt_module
        passt_module.first_RUN = False
    elif model_name == "htsat":
        from src.vendor.htsat.htsat import HTSAT_Swin_Transformer
        cfg = SimpleNamespace(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                              fmin=50, fmax=14000, enable_tscam=True, htsat_attn_heatmap=False,
                              loss_type="clip_bce", enable_repeat_mode=False)
        net = HTSAT_Swin_Transformer(config=cfg)
        pretrained = state["state_dict"]
        if not all(k.startswith("sed_model.") for k in pretrained):
            raise ValueError("Unexpected HTS-AT checkpoint namespace")
        net.load_state_dict({k.removeprefix("sed_model."): v for k, v in pretrained.items()}, strict=True)
        encoder = HtsatEncoder(net)
    else:
        raise ValueError("Choose passt or htsat")
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


class PasstEncoder(nn.Module):
    def __init__(self, net, mel):
        super().__init__()
        self.net, self.mel = net, mel

    def forward(self, waveform):
        # Official normalization is inside mel. Spectrogram layout: B,1,128,T.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrogram = self.mel(waveform.float()).unsqueeze(1)
        features = self.net.forward_features(spectrogram)
        return (features[0] + features[1]) / 2 if isinstance(features, tuple) else features


class HtsatEncoder(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, waveform):
        # Preserve official STFT, 64-band Log-Mel, BN and short-clip reshape.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            x = self.net.spectrogram_extractor(waveform.float())
            x = self.net.logmel_extractor(x)
            x = self.net.bn0(x.transpose(1, 3)).transpose(1, 3)
        x = self.net.reshape_wav2img(x)
        return self.net.forward_features(x)["latent_output"]


class FrozenProbe(nn.Module):
    """Training mode never changes the frozen encoder's BN/augmentation state."""
    def __init__(self, encoder, classes=13, embedding_dim=768):
        super().__init__()
        self.encoder = encoder.requires_grad_(False)
        self.encoder.eval()
        self.head = nn.Linear(embedding_dim, classes)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, waveform):
        with torch.no_grad():
            features = self.encoder(waveform)
        return self.head(features)
