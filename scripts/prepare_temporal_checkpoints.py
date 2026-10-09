"""Prepare pinned, pretrained weights; never permit a random encoder fallback."""
import hashlib
import json
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HF_REF = 'f826b80d28226b62986cc218e5cec390b1096902'
AST_HASH = 'ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d'
AUDIO_MAE_HASH = '9704862465c8474ca47519912bdeca271f06b44fcc67f29266a8f57c1188ac12'
AST_URL = f'https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/resolve/{HF_REF}/model.safetensors'


def prepare():
    from src.data.transfer import sha256
    destination = ROOT/'checkpoints/pretrained_models'
    destination.mkdir(parents=True, exist_ok=True)
    ast = destination/'ast_audioset_10_10_04593.safetensors'
    if not ast.is_file():
        temporary = ast.with_suffix('.partial')
        digest = hashlib.sha256()
        size = 0
        with requests.get(AST_URL, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise ValueError('Checkpoint endpoint returned HTML')
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(4*1024*1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        if digest.hexdigest() != AST_HASH:
            raise ValueError('AST published SHA-256 mismatch; partial download preserved')
        temporary.rename(ast)
        print('AST_DOWNLOADED', size, flush=True)
    if sha256(ast) != AST_HASH:
        raise ValueError('AST checkpoint changed')
    audiomae = ROOT/'experiments/audiomae/ckpt/finetuned.pth'
    if not audiomae.is_file():
        raise FileNotFoundError('The official AudioMAE fine-tuned checkpoint is required')
    if sha256(audiomae) != AUDIO_MAE_HASH:
        raise ValueError('AudioMAE checkpoint differs from the pinned artifact used in this experiment')
    metadata = {
        'ast': {'path': ast.relative_to(ROOT).as_posix(), 'sha256': AST_HASH,
                'source': AST_URL, 'revision': HF_REF,
                'checkpoint': 'AudioSet 10/10 stride, weight averaging, 0.4593',
                'format': 'Hugging Face lossless tensor conversion of the original AST checkpoint',
                'conversion_source': 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/audio_spectrogram_transformer/convert_audio_spectrogram_transformer_original_to_pytorch.py',
                'original_dropbox_unavailable': True},
        'audiomae': {'path': audiomae.relative_to(ROOT).as_posix(), 'sha256': AUDIO_MAE_HASH,
                    'source': 'https://github.com/facebookresearch/AudioMAE',
                    'checkpoint': 'Official AudioSet fine-tuned checkpoint, epoch 99',
                    'colleague_checkpoint_byte_identity_not_proven': True},
        'random_weight_fallback': False,
    }
    (destination/'temporal_sources.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    prepare()
