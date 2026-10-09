"""Strict pretrained AST/AudioMAE loaders for the isolated temporal experiment.

AST tensor renaming reverses the official Transformers conversion, including
losslessly concatenating Q/K/V. Unused ImageNet and AudioSet heads are removed;
every parameter used for the 768-dimensional representation loads strictly.
"""
import json
import os
import pathlib
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


def sdpa_attention(self, x):
    batch, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels//self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
    return self.proj_drop(self.proj(x.transpose(1, 2).reshape(batch, tokens, channels)))


def enable_sdpa(model):
    """Use the same attention equation with PyTorch's memory-efficient kernel."""
    count = 0
    for module in model.modules():
        if all(hasattr(module, key) for key in ('qkv', 'num_heads', 'scale', 'proj', 'proj_drop')):
            module.forward = MethodType(sdpa_attention, module)
            count += 1
    if count != 12:
        raise ValueError(f'Expected twelve pretrained attention blocks, got {count}')
    return model


def ast_backbone_state(state):
    prefix = 'audio_spectrogram_transformer.'
    output, consumed = {}, set()
    aliases = {'embeddings.cls_token': 'cls_token', 'embeddings.distillation_token': 'dist_token',
               'embeddings.position_embeddings': 'pos_embed',
               'embeddings.patch_embeddings.projection.weight': 'patch_embed.proj.weight',
               'embeddings.patch_embeddings.projection.bias': 'patch_embed.proj.bias',
               'layernorm.weight': 'norm.weight', 'layernorm.bias': 'norm.bias'}
    for key, target in aliases.items():
        output['v.'+target] = state[prefix+key]
        consumed.add(prefix+key)
    replacements = {'layernorm_before': 'norm1', 'layernorm_after': 'norm2',
                    'attention.output.dense': 'attn.proj', 'intermediate.dense': 'mlp.fc1',
                    'output.dense': 'mlp.fc2'}
    for layer in range(12):
        start = f'{prefix}encoder.layer.{layer}.'
        for source, target in replacements.items():
            for suffix in ('weight', 'bias'):
                key = start+source+'.'+suffix
                output[f'v.blocks.{layer}.{target}.{suffix}'] = state[key]
                consumed.add(key)
        for suffix in ('weight', 'bias'):
            keys = [start+'attention.attention.'+name+'.'+suffix for name in ('query', 'key', 'value')]
            # HF release uses attention.attention, as the original converter does.
            if keys[0] not in state:
                keys = [start+'attention.self.'+name+'.'+suffix for name in ('query', 'key', 'value')]
            output[f'v.blocks.{layer}.attn.qkv.{suffix}'] = torch.cat([state[key] for key in keys], dim=0)
            consumed.update(keys)
    remaining = set(state)-consumed
    if remaining != {'classifier.layernorm.weight', 'classifier.layernorm.bias', 'classifier.dense.weight', 'classifier.dense.bias'}:
        raise ValueError(f'Unaccounted AST checkpoint tensors: {remaining}')
    return output


def load_legacy_encoder(name, *, use_sdpa=True):
    import timm
    from src.data.transfer import sha256
    if timm.__version__ != '0.4.5':
        raise RuntimeError('Use the isolated timm 0.4.5 environment for AST/AudioMAE')
    sources = json.loads((ROOT/'checkpoints/pretrained_models/temporal_sources.json').read_text())
    info = sources[name]
    path = ROOT/info['path']
    if sha256(path) != info['sha256']:
        raise ValueError('Pinned pretrained checkpoint changed')
    if name == 'ast':
        from safetensors.torch import load_file
        from src.models.ast_models import ASTModel
        original_patch = timm.models.vision_transformer.PatchEmbed
        try:
            model = ASTModel(label_dim=527, input_tdim=1024, imagenet_pretrain=False,
                             audioset_pretrain=False, verbose=False)
        finally:
            timm.models.vision_transformer.PatchEmbed = original_patch
        model.mlp_head = nn.Identity()
        model.v.head, model.v.head_dist = nn.Identity(), nn.Identity()
        state = ast_backbone_state(load_file(str(path)))
        model.load_state_dict(state, strict=True)
    elif name == 'audiomae':
        from src.models.audiomae_models import vit_base_patch16
        model = vit_base_patch16(num_classes=527, global_pool=True, drop_path_rate=0.1)
        # The known official Linux checkpoint contains a PosixPath in args.
        old = pathlib.PosixPath
        try:
            if os.name == 'nt':
                pathlib.PosixPath = pathlib.WindowsPath
            saved = torch.load(path, map_location='cpu', weights_only=False)
        finally:
            pathlib.PosixPath = old
        model.load_state_dict(saved['model'], strict=True)
        model.head = nn.Identity()
    else:
        raise ValueError('Choose ast or audiomae')
    model.requires_grad_(False).eval()
    return enable_sdpa(model) if use_sdpa else model
