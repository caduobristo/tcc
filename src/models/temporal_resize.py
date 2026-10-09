"""Opt-in PaSST temporal ablation, isolated from original encoder cache sources."""
import torch
from torch import nn

from src.data.spectrogram_resize import adapt_spectrogram_time


class TemporallyResizedPasst(nn.Module):
    """Keep the original frontend/weights; adapt its (B,F,T) output to 998 frames."""
    def __init__(self, encoder, target_frames=998):
        super().__init__()
        if target_frames != 998:
            raise ValueError("The pinned PaSST checkpoint is configured for 998 frames")
        self.encoder = encoder.requires_grad_(False).eval()
        self.target_frames = target_frames

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, waveform):
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            mel = self.encoder.mel(waveform.float())
            if mel.ndim != 3 or mel.shape[1] != 128:
                raise ValueError("PaSST frontend must produce (B,128,T)")
            mel = adapt_spectrogram_time(mel.transpose(1, 2), self.target_frames)
            image = mel.transpose(1, 2).unsqueeze(1)
        features = self.encoder.net.forward_features(image)
        return (features[0] + features[1]) / 2 if isinstance(features, tuple) else features


def load_temporally_resized_passt():
    from src.models.transfer import load_frozen_encoder
    return TemporallyResizedPasst(load_frozen_encoder("passt")).eval()
