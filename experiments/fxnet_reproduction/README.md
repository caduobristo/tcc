# Reprodução do artigo de Comunità, Stowell e Reiss

## Origem e objetivo

Artigo: [Guitar Effects Recognition and Parameter Estimation with Convolutional Neural Networks](https://arxiv.org/abs/2012.03216).

Código de referência: [mcomunita/gfx-classifier](https://github.com/mcomunita/gfx-classifier). O snapshot local foi importado do fork [rodrigomoliani/gfx-classifier](https://github.com/rodrigomoliani/gfx-classifier), commit `269e75d3417251406646621bee323c2192dc2483`, incluindo as alterações e scripts locais que ainda não tinham sido commitados.

A licença original BSD 3-Clause e os créditos estão preservados em [gfx-classifier/LICENSE](gfx-classifier/LICENSE). Os notebooks e módulos de referência não devem ser apresentados como código originalmente desenvolvido pelos autores do TCC.

Esta é a tentativa histórica de reprodução das redes FxNet, SetNet, SetNetCond e MultiNet. Ela é distinta da baseline consolidada em `../../notebooks/train_baseline.ipynb`.

## Estrutura

- `gfx-classifier/`: código de referência, notebooks e ferramentas de execução/avaliação.
- `models_and_results/results/`: relatórios e tabelas resumidas das execuções anteriores.
- `models_and_results/models/`, `data/`, `.conda/`, `.cache/` e `local_archive/`: arquivos locais ignorados pelo Git.

## Estado documentado

O [relatório de reprodução](models_and_results/results/_article_repro/pipeline_summary.md) registra 11 notebooks: três concluídos e oito com falhas. Os treinos discretos concluídos excluíram TS9, conforme os nomes dos resultados; seus números não devem ser comparados diretamente a benchmarks com 13 classes.

As falhas incluem incompatibilidades CPU/CUDA e um erro de índice na execução mono contínua. A reorganização preserva esse registro e não declara reprodução completa do artigo.

Também existem avaliações posteriores de checkpoints nos quatro subconjuntos em `models_and_results/results/_model_eval/`. Alguns relatórios usam `split=full`, que inclui o conjunto inteiro: esses números não equivalem a uma avaliação independente no teste. Consulte os JSONs de resumo antes de usar qualquer valor no TCC.

## Executar

Os notebooks esperam o diretório de trabalho `gfx-classifier/src/`, datasets em `../../data/GUITAR-FX` ou `../../data/GUITAR-FX-DIST` e modelos em `../../models_and_results/models`.

Esses são os caminhos dos notebooks históricos. A organização atual é descrita em [data/README.md](../../data/README.md): WAVs em `data/raw/guitar_fx_dist/` e features originais em `data/processed/guitar_fx_dist/baseline/`. O avaliador `evaluate_fxnet_model.py` já usa a localização central das features e mantém `--dataset-root` para substituí-la. Antes de retomar o pipeline histórico, restaure as features originais e ajuste as variáveis de caminho dos notebooks. Não use mel16/mel32 como substitutos: essas matrizes têm 198 quadros e não são compatíveis com a FxNet preservada.

O ambiente histórico foi preparado com Miniconda no Windows. As dependências originais estão em `gfx-classifier/src/requirements.txt`; versões antigas podem exigir adaptações para GPUs e versões recentes de Python/PyTorch. Os scripts `.cmd` e `.ps1` em `gfx-classifier/` usam o ambiente local `.conda/envs/gfx-classifier`.

```powershell
cd experiments/fxnet_reproduction/gfx-classifier
.\run_in_gfx_env.cmd python src/scripts/evaluate_fxnet_model.py --help
.\run_in_gfx_env.cmd python src/scripts/run_article_pipeline.py --help
```

Esses comandos exibem as opções. Treinar ou avaliar exige os datasets e checkpoints locais; não são distribuídos neste repositório.

## Preservação local

Os notebooks foram publicados sem saídas embutidas. Suas versões anteriores estão em `local_archive/original_notebooks/`. O histórico do clone original foi preservado em `.git/experiment-source-history/gfx-classifier.git` na raiz do TCC.

Uma junção local de `C:\TCC\Reproduzir_artigo` para esta pasta preserva os caminhos usados pelos ambientes Conda e pelas execuções anteriores. Ela não faz parte do repositório publicado.
