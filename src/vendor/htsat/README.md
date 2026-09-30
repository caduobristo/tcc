# HTS-AT core

Source: https://github.com/RetroCirce/HTS-Audio-Transformer
Commit: `2e29471fce7e770edeaaa64a41908bf234844fc5`.

`htsat.py` and `layers.py` preserve the official architecture. The utility import is relative; `utils.py` contains only the two imported upstream functions to avoid unrelated training dependencies. License is preserved in `LICENSE`. The project wrapper in `src/models/transfer.py` selects the frozen latent output and handles waveform loading/preprocessing.
