import torch
import torch.nn as nn
from functools import partial
import timm.models.vision_transformer
from timm.models.layers import PatchEmbed

class AudioMAE_VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling """
    def __init__(self, global_pool=False, mask_2d=True, **kwargs):
        super().__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
        del self.norm  # remove the original norm
        self.mask_2d = mask_2d

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        if hasattr(self, 'head') and isinstance(self.head, nn.Module):
            x = self.head(x)
        return x

def vit_base_patch16(**kwargs):
    model = AudioMAE_VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    # Configure Patch Embedding specifically for 1024x128 spectrograms (AudioSet format)
    img_size = (1024, 128)
    in_chans = 1
    model.patch_embed = PatchEmbed(img_size, 16, in_chans, 768)
    
    # 1024//16 = 64, 128//16 = 8, 64 * 8 = 512 patches
    num_patches = 512
    model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 768), requires_grad=False)
    
    return model
