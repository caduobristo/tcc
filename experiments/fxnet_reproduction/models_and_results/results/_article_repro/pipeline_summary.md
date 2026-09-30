# Article Reproduction Pipeline

## Visao Geral
- Ultima atualizacao: `2026-03-21T20:52:04-03:00`
- Inicio da execucao: `2026-03-21T19:08:04-03:00`
- Python: `C:\TCC\Reproduzir_artigo\.conda\envs\gfx-classifier\python.exe`
- Total de notebooks: **11**
- Concluidos: **3** | Falhas: **8** | Rodando: **0** | Pendentes: **0**
- Torch: `2.10.0+cu128` | CUDA: `True` | GPU: `NVIDIA GeForce RTX 5080`

## Metricas Por Modelo

Obs.: metricas em `%` quando os arquivos `.npy` de resultado existem.

| # | Notebook | Status | Epocas | Melhor epoca | Val (melhor) | Teste (na melhor val) | Val final | Teste final | Pasta resultado | Modelo salvo |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | train_fxnet_on_mono_cont.ipynb | failed | - | - | - | - | - | - | 20201024_fxnet_mono_cont | 20201024_fxnet_mono_cont_best |
| 2 | train_fxnet_on_poly_cont.ipynb | completed | 43 | 28 | 91.35% | 91.35% | 78.30% | 78.71% | 20201025_fxnet_poly_cont | 20201025_fxnet_poly_cont_best |
| 3 | train_fxnet_on_mono_disc.ipynb | completed | 38 | 23 | 97.01% | 97.05% | 96.19% | 96.51% | 20201210_fxnet_mono_disc_noTS9 | 20201210_fxnet_mono_disc_noTS9_best |
| 4 | train_fxnet_on_poly_disc.ipynb | completed | 46 | 31 | 98.66% | 98.73% | 97.64% | 97.59% | 20201211_fxnet_poly_disc_noTS9 | 20201211_fxnet_poly_disc_noTS9_best |
| 5 | train_setnetcond_on_mono_cont.ipynb | failed | - | - | - | - | - | - | 20201020_setnetcond_mono_cont | - |
| 6 | train_setnetcond_on_poly_cont.ipynb | failed | - | - | - | - | - | - | 20201022_setnetcond_poly_cont | - |
| 7 | train_setnetcond_on_mono_disc.ipynb | failed | - | - | - | - | - | - | 20210409_setnetcond_mono_disc | - |
| 8 | train_setnetcond_on_poly_disc.ipynb | failed | - | - | - | - | - | - | 20201018_setnetcond_poly_disc | - |
| 9 | train_multinet_on_mono_disc.ipynb | failed | - | - | - | - | - | - | 20201110_multinet_mono_disc | - |
| 10 | train_setnet_on_mono_disc.ipynb | failed | - | - | - | - | - | - | 20201111_setnet_mono_disc | - |
| 11 | train_fxnet_and_setnetcond_on_mono_disc.ipynb | failed | - | - | - | - | - | - | 20201001_fxnet_and_setnetcond_mono_disc_best | - |

## Falhas

| # | Notebook | Duracao (min) | Erro principal | Log |
|---:|---|---:|---|---|
| 1 | train_fxnet_on_mono_cont.ipynb | 33.8 | [2026-03-21T19:42:57-03:00] IndexError: list index out of range | train_fxnet_on_mono_cont_20260321_190907.log |
| 5 | train_setnetcond_on_mono_cont.ipynb | 0.1 | [2026-03-21T20:51:21-03:00] RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA__index_select) | train_setnetcond_on_mono_cont_20260321_205113.log |
| 6 | train_setnetcond_on_poly_cont.ipynb | 0.1 | [2026-03-21T20:51:28-03:00] RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA__index_select) | train_setnetcond_on_poly_cont_20260321_205121.log |
| 7 | train_setnetcond_on_mono_disc.ipynb | 0.1 | [2026-03-21T20:51:35-03:00] RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA__index_select) | train_setnetcond_on_mono_disc_20260321_205128.log |
| 8 | train_setnetcond_on_poly_disc.ipynb | 0.1 | [2026-03-21T20:51:42-03:00] RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA__index_select) | train_setnetcond_on_poly_disc_20260321_205135.log |
| 9 | train_multinet_on_mono_disc.ipynb | 0.1 | [2026-03-21T20:51:50-03:00] TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. | train_multinet_on_mono_disc_20260321_205142.log |
| 10 | train_setnet_on_mono_disc.ipynb | 0.1 | [2026-03-21T20:51:57-03:00] RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor | train_setnet_on_mono_disc_20260321_205150.log |
| 11 | train_fxnet_and_setnetcond_on_mono_disc.ipynb | 0.1 | [2026-03-21T20:52:04-03:00] RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA__index_select) | train_fxnet_and_setnetcond_on_mono_disc_20260321_205157.log |

## Linha Do Tempo

| # | Notebook | Status | Duracao (min) | Inicio | Fim | Log |
|---:|---|---|---:|---|---|---|
| 1 | train_fxnet_on_mono_cont.ipynb | failed | 33.8 | 2026-03-21T19:09:07-03:00 | 2026-03-21T19:42:57-03:00 | train_fxnet_on_mono_cont_20260321_190907.log |
| 2 | train_fxnet_on_poly_cont.ipynb | completed | 28.6 | 2026-03-21T19:42:57-03:00 | 2026-03-21T20:11:35-03:00 | train_fxnet_on_poly_cont_20260321_194257.log |
| 3 | train_fxnet_on_mono_disc.ipynb | completed | 22.5 | 2026-03-21T20:11:35-03:00 | 2026-03-21T20:34:07-03:00 | train_fxnet_on_mono_disc_20260321_201135.log |
| 4 | train_fxnet_on_poly_disc.ipynb | completed | 17.1 | 2026-03-21T20:34:07-03:00 | 2026-03-21T20:51:13-03:00 | train_fxnet_on_poly_disc_20260321_203407.log |
| 5 | train_setnetcond_on_mono_cont.ipynb | failed | 0.1 | 2026-03-21T20:51:13-03:00 | 2026-03-21T20:51:21-03:00 | train_setnetcond_on_mono_cont_20260321_205113.log |
| 6 | train_setnetcond_on_poly_cont.ipynb | failed | 0.1 | 2026-03-21T20:51:21-03:00 | 2026-03-21T20:51:28-03:00 | train_setnetcond_on_poly_cont_20260321_205121.log |
| 7 | train_setnetcond_on_mono_disc.ipynb | failed | 0.1 | 2026-03-21T20:51:28-03:00 | 2026-03-21T20:51:35-03:00 | train_setnetcond_on_mono_disc_20260321_205128.log |
| 8 | train_setnetcond_on_poly_disc.ipynb | failed | 0.1 | 2026-03-21T20:51:35-03:00 | 2026-03-21T20:51:42-03:00 | train_setnetcond_on_poly_disc_20260321_205135.log |
| 9 | train_multinet_on_mono_disc.ipynb | failed | 0.1 | 2026-03-21T20:51:42-03:00 | 2026-03-21T20:51:50-03:00 | train_multinet_on_mono_disc_20260321_205142.log |
| 10 | train_setnet_on_mono_disc.ipynb | failed | 0.1 | 2026-03-21T20:51:50-03:00 | 2026-03-21T20:51:57-03:00 | train_setnet_on_mono_disc_20260321_205150.log |
| 11 | train_fxnet_and_setnetcond_on_mono_disc.ipynb | failed | 0.1 | 2026-03-21T20:51:57-03:00 | 2026-03-21T20:52:04-03:00 | train_fxnet_and_setnetcond_on_mono_disc_20260321_205157.log |
