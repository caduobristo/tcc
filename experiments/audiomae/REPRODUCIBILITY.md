# AudioMAE local reproduction

This directory contains a local reproduction setup for `Masked Autoencoders that Listen`.

## Layout

- `AudioMAE.pdf`: local copy of the paper.
- `repo`: official `facebookresearch/AudioMAE` clone pinned at `bd60e29651285f80d32a6405082835ad26e6f19f`.
- `.conda/envs/audiomae`: local Conda environment.
- `ckpt/pretrained.pth`: official AudioSet self-supervised checkpoint.
- `ckpt/finetuned.pth`: official AudioSet fine-tuned checkpoint.
- `data/smoke`: synthetic WAV used for smoke testing.
- `data/audioset`: expected location for AST-style AudioSet JSON/CSV files.
- `notebooks`: Jupyter notebooks for environment check, smoke test, and AudioSet evaluation template.
- `logs`: environment and smoke-test logs.

## Launch Jupyter

Run from PowerShell:

```powershell
$CONDA="C:\ProgramData\miniconda3\Scripts\conda.exe"
$ENV="C:\TCC\experiments\audiomae\.conda\envs\audiomae"
& $CONDA run -p $ENV jupyter lab --notebook-dir "C:\TCC\experiments\audiomae"
```

Use the `Python (AudioMAE)` kernel.

## Environment

The paper used a Linux Conda stack with PyTorch 1.7/CUDA 10.2 on V100 GPUs. This local setup uses Windows Conda with PyTorch 2.10/CUDA 13.0 because the machine has an RTX 5080.

Installed core versions:

- Python 3.10.20
- PyTorch 2.10.0+cu130
- torchvision 0.25.0+cu130
- torchaudio 2.10.0+cu130
- timm 0.3.2 with the repository patch applied
- JupyterLab
- ffmpeg

Compatibility edits applied to the local clone:

- `repo/util/misc.py`: replace removed `torch._six.inf` usage and add `torch_load_compat` for Linux checkpoints on Windows.
- `repo/main_finetune_as.py`: use `misc.torch_load_compat` for `--finetune`.
- `repo/main_finetune_esc.py`: use `misc.torch_load_compat` for `--finetune`.

## Verified smoke test

The smoke test generated `data/smoke/synthetic_10s_16k.wav`, converted it to a Kaldi-compatible fbank, loaded `ckpt/finetuned.pth`, and ran one forward pass on CUDA.

Observed result:

```text
fbank_shape: [1024, 128]
sample_shape: [1, 1, 1024, 128]
cuda_available: true
gpu: NVIDIA GeForce RTX 5080
checkpoint_epoch: 99
missing_keys: []
unexpected_keys: []
output_shape: [1, 527]
```

Full output is in `logs/smoke_test.txt`; installed packages are in `logs/pip-freeze.txt`.

## Full AudioSet evaluation

Full metric reproduction requires AudioSet audio and AST-style metadata files. Place these files under `data/audioset`:

- `train.json`
- `eval_19k.json`
- `class_labels_indices.csv`

Then run `notebooks/02_audioset_eval_template.ipynb` or the equivalent command:

```powershell
$PY="C:\TCC\experiments\audiomae\.conda\envs\audiomae\python.exe"
cd C:\TCC\experiments\audiomae\repo
& $PY main_finetune_as.py `
  --model vit_base_patch16 `
  --dataset audioset `
  --data_train C:\TCC\experiments\audiomae\data\audioset\train.json `
  --data_eval C:\TCC\experiments\audiomae\data\audioset\eval_19k.json `
  --label_csv C:\TCC\experiments\audiomae\data\audioset\class_labels_indices.csv `
  --finetune C:\TCC\experiments\audiomae\ckpt\finetuned.pth `
  --batch_size 16 `
  --eval
```

The paper/repository expectation is approximately `mAP = 0.4729` on the AudioSet eval set used by AST.
