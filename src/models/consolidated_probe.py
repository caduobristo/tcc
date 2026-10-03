"""Inference with a consolidated frozen encoder, training scaler and linear head."""
from pathlib import Path
import json

import torch
from torch import nn

from src.data.transfer import EFFECTS, sha256
from src.models.transfer import load_frozen_encoder


class ConsolidatedProbe(nn.Module):
    def __init__(self, encoder, head, mean, scale, *, bfloat16_encoder=False):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.head = head
        self.register_buffer('mean',mean)
        self.register_buffer('scale',scale)
        self.bfloat16_encoder = bfloat16_encoder
        self.classes = tuple(EFFECTS)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward_embeddings(self, embeddings):
        return self.head((embeddings.float()-self.mean)/self.scale)

    def forward(self, waveform):
        if self.bfloat16_encoder and waveform.device.type != 'cuda':
            raise ValueError('This run used CUDA BF16 embeddings; use CUDA to preserve its precision recipe')
        with torch.no_grad(), torch.autocast(device_type=waveform.device.type,dtype=torch.bfloat16,
                                              enabled=self.bfloat16_encoder):
            features = self.encoder(waveform)
        return self.forward_embeddings(features)


def load_consolidated_probe(run_folder, device='cuda'):
    """Load local artifacts; input is mono waveform at 32 kHz with 64,000 samples."""
    folder = Path(run_folder)
    record = json.loads((folder/'run.json').read_text(encoding='utf-8'))
    if record['status'] != 'complete' or not record['encoder_frozen']:
        raise ValueError('Completed frozen-encoder run required')
    for name,key in (('best_head.pt','head_sha256'),('preprocessing.pt','preprocessing_sha256')):
        if sha256(folder/name) != record[key]:
            raise ValueError(f'Changed artifact: {name}')
    prep = torch.load(folder/'preprocessing.pt',map_location='cpu',weights_only=True)
    if (prep['mean'].shape != (768,) or prep['scale'].shape != (768,)
            or not bool(torch.isfinite(prep['mean']).all())
            or not bool(torch.isfinite(prep['scale']).all()) or not bool((prep['scale']>0).all())):
        raise ValueError('Invalid training-fitted scaler')
    classes = tuple(record.get('classes', EFFECTS))
    if len(classes) < 2 or classes != tuple(effect for effect in EFFECTS if effect in classes):
        raise ValueError('Invalid saved classifier class mapping')
    head = nn.Linear(768,len(classes))
    head.load_state_dict(torch.load(folder/'best_head.pt',map_location='cpu',weights_only=True),strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in head.parameters()):
        raise ValueError('Invalid classifier weights')
    probe = ConsolidatedProbe(load_frozen_encoder(record['model']),head,prep['mean'],prep['scale'],
                             bfloat16_encoder=record['cache_identity']['encoder_precision']=='bfloat16').to(device).eval()
    probe.classes = classes
    return probe
