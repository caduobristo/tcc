"""Temporal adaptation of numerical log-Mel tensors; never resize rendered PNGs."""
from numbers import Integral

import torch
from torch.nn import functional as F


def adapt_spectrogram_time(spectrogram, target_frames, *, mode="bicubic"):
    """Adapt (..., time, mel) without changing Mel bands or mixing batches.

    Bicubic uses the same align_corners=True setting as HTS-AT reshape_wav2img.
    Call on real, normalized frames, before any padding. ``pad`` retains the
    historical AST/AudioMAE behavior (right-pad or truncate).
    """
    if isinstance(target_frames, bool) or not isinstance(target_frames, Integral) or target_frames < 1:
        raise ValueError("target_frames must be a positive integer")
    if mode not in {"pad", "bicubic"}:
        raise ValueError("mode must be pad or bicubic")
    if spectrogram.ndim not in {2, 3, 4} or any(d == 0 for d in spectrogram.shape):
        raise ValueError("Expected a nonempty (T,F), (B,T,F) or (B,C,T,F) tensor")
    if not spectrogram.is_floating_point() or not torch.isfinite(spectrogram).all():
        raise ValueError("Spectrogram must contain finite floating-point values")
    frames, bands = spectrogram.shape[-2:]
    if frames == target_frames:
        return spectrogram
    if mode == "pad":
        return F.pad(spectrogram, (0, 0, 0, target_frames - frames)) if frames < target_frames else spectrogram[..., :target_frames, :]
    # Flatten leading dimensions into independent samples, keeping frequency width.
    original_shape = spectrogram.shape
    images = spectrogram.reshape(-1, 1, frames, bands).float()
    resized = F.interpolate(images, size=(target_frames, bands), mode="bicubic", align_corners=True)
    return resized.reshape(*original_shape[:-2], target_frames, bands).to(spectrogram.dtype)
