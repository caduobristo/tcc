# Reprodução e exploração com AudioMAE

## Origem e objetivo

Artigo: [Masked Autoencoders that Listen](https://arxiv.org/abs/2207.06405), de Huang et al.

Código de origem: [facebookresearch/AudioMAE](https://github.com/facebookresearch/AudioMAE), commit `bd60e29651285f80d32a6405082835ad26e6f19f`.

O código em `repo/` é um snapshot com ajustes de compatibilidade para PyTorch recente e checkpoints Linux no Windows, descritos em [REPRODUCIBILITY.md](REPRODUCIBILITY.md). A licença original CC BY-NC 4.0 está em [repo/LICENSE](repo/LICENSE); essa licença continua aplicável ao código importado.

O experimento valida a execução do AudioMAE e explora embeddings de guitarra. Ele apoia a etapa de reprodução e o uso de modelos pré-treinados previstos no TCC.

## Estrutura

- `repo/`: código de referência e adaptações locais.
- `scripts/`: inferência AudioSet e extração de embeddings no GUITAR-FX.
- `notebooks/`: verificação do ambiente, smoke test e template de avaliação AudioSet.
- `outputs/`: gráficos e tabelas resumidas das execuções históricas.
- `ckpt/`, `data/`, `.conda/`, `logs/` e `local_archive/`: arquivos locais ignorados pelo Git.

## Resultados e limites

O smoke test registrado carregou o checkpoint fine-tuned e gerou uma saída com 527 classes AudioSet. A avaliação completa do artigo no AudioSet ainda depende do dataset e seus metadados; o notebook correspondente é um template.

A exploração de embeddings em Mono Continuous utilizou 3.750 áudios e vetores de 768 dimensões. Veja [PCA](outputs/guitarfx_audiomae_embeddings_mono_continuous/pca_2d.png) e [t-SNE](outputs/guitarfx_audiomae_embeddings_mono_continuous/tsne_2d.png).

A inferência histórica nos quatro subconjuntos é documentada em [COMPARISON_WITH_FXNET.md](outputs/guitarfx_audiomae_all_datasets_fxnet_comparison/COMPARISON_WITH_FXNET.md). A cabeça AudioSet não prevê identidades de pedais: esses resultados não são acurácia de classificação GUITAR-FX e não são diretamente comparáveis à FxNet. A adaptação supervisionada da cabeça ou do encoder permanece uma etapa posterior.

## Executar

O ambiente local histórico usa Python 3.10, PyTorch/CUDA e `timm==0.3.2` com o patch do repositório. Consulte REPRODUCIBILITY.md e `repo/mae_env.yml` para a configuração. Não execute este experimento no ambiente antigo do PaSST sem verificar as versões.

Partindo da raiz do TCC, com o ambiente configurado:

```powershell
python experiments/audiomae/scripts/extract_audiomae_guitarfx_embeddings.py --help
python experiments/audiomae/scripts/run_audiomae_guitarfx_four_datasets.py --help
```

Os caminhos padrão são derivados da localização dos scripts: `ckpt/finetuned.pth` neste experimento e `../fxnet_reproduction/data/GUITAR-FX` para os áudios. É possível passar `--guitarfx-root`, `--checkpoint` e `--output-dir` para usar outros caminhos.

Abra os notebooks a partir de `experiments/audiomae/notebooks` ou `experiments/audiomae`. Os checkpoints oficiais devem ser obtidos conforme as instruções do repositório de origem; não são distribuídos neste TCC.

## Preservação local

Na reorganização de 29/09/2026, saídas embutidas dos notebooks foram guardadas em `local_archive/original_notebooks/`. Checkpoints e tabelas completas continuam no computador. O histórico Git do clone original foi preservado em `.git/experiment-source-history/audiomae.git` na raiz do TCC, sem publicá-lo.

Uma junção local de `C:\TCC\AudioMAE` para esta pasta mantém compatibilidade com prefixos Conda, kernels e caminhos históricos. Ela não faz parte do repositório publicado.
