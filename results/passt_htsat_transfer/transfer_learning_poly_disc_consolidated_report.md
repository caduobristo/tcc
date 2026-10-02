# Transfer learning consolidado: PaSST e HTS-AT — Poly Discrete

Execução em 01/10/2026, com encoders congelados. Fine-tuning não foi realizado.

Sessão local: `results/transfer_learning/poly_disc/consolidation_20261001_214734_fa76e3`. Duração do protocolo: **10.00 minutos**, reutilizando os embeddings já extraídos.

83,160 áudios Poly Discrete/13 efeitos; 420 fontes; treino/validação/teste 59,796/6,534/16,830 por fonte (seed 42). WAVs a 32 kHz com frontend nativo, não os mel32 Kaldi.

## Resultado no teste

| Modelo | Acurácia (%) | F1 macro (%) | Precisão macro (%) | Recall macro (%) |
| --- | ---: | ---: | ---: | ---: |
| PaSST | 85,20 ± 0,12 | 83,24 ± 0,35 | 85,02 ± 0,33 | 83,20 ± 0,24 |
| HTS-AT | 92,73 ± 0,13 | 92,57 ± 0,11 | 93,18 ± 0,14 | 92,23 ± 0,08 |

Média ± desvio padrão amostral de cinco seeds novas (101, 202, 303, 404 e 505). Cada seed é avaliada nos mesmos 16,830 áudios; as cinco avaliações não são novos exemplos independentes.

## Seleção e execuções

Oito configurações por modelo foram comparadas usando F1 macro de validação. As duas melhores receberam mais duas seeds; a configuração foi escolhida pela média nas seeds 42, 7 e 21. A seleção foi registrada antes dos dez treinamentos finais, e todas as cabeças foram finalizadas antes de avaliar o teste.

AdamW: LR inicial 0,001, weight decay 0,01; teto de 200 épocas; scheduler e parada antecipada na validação. Protocolo local: `docs/transfer_learning_all_scenarios.md` (fora do Git). Consulte [a configuração](../configs/linear_probe/consolidation_poly_disc.json).

### PaSST

Configuração escolhida: `train_standardize_none_b128`. F1 macro de validação nas três seeds: **83,36 ± 0,39%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_none_b128 | 83,80 | 47 | 62 |
| train_standardize_train_balanced_b128 | 83,34 | 18 | 33 |
| none_train_balanced_b128 | 83,19 | 45 | 60 |
| train_standardize_train_balanced_b1024 | 82,84 | 42 | 57 |
| none_train_balanced_b1024 | 82,04 | 74 | 79 |
| train_standardize_none_b1024 | 81,74 | 42 | 57 |
| none_none_b128 | 81,06 | 28 | 32 |
| none_none_b1024 | 80,22 | 59 | 59 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_none_b128 | 83,36 ± 0,39 |
| train_standardize_train_balanced_b128 | 83,13 ± 0,23 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 60 / 70 | 32760 | 27.19 | 85,39 | 83,71 |
| 202 | 47 / 62 | 29016 | 23.26 | 85,12 | 83,24 |
| 303 | 54 / 69 | 32292 | 26.85 | 85,25 | 83,14 |
| 404 | 30 / 45 | 21060 | 17.51 | 85,10 | 82,75 |
| 505 | 35 / 39 | 18252 | 14.98 | 85,15 | 83,38 |

Checkpoint representativo escolhido pela **validação**, seed 101: `results/transfer_learning/poly_disc/consolidation_20261001_214734_fa76e3/passt/final/seed_101`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

### HTS-AT

Configuração escolhida: `train_standardize_none_b128`. F1 macro de validação nas três seeds: **92,22 ± 0,11%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_none_b128 | 92,10 | 37 | 45 |
| train_standardize_train_balanced_b128 | 91,78 | 43 | 53 |
| train_standardize_none_b1024 | 91,65 | 40 | 51 |
| train_standardize_train_balanced_b1024 | 90,96 | 35 | 36 |
| none_none_b128 | 90,03 | 83 | 84 |
| none_train_balanced_b128 | 89,05 | 69 | 78 |
| none_train_balanced_b1024 | 87,37 | 92 | 107 |
| none_none_b1024 | 87,27 | 107 | 108 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_none_b128 | 92,22 ± 0,11 |
| train_standardize_train_balanced_b128 | 91,80 ± 0,11 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 32 / 39 | 18252 | 15.05 | 92,73 | 92,56 |
| 202 | 31 / 38 | 17784 | 14.40 | 92,89 | 92,72 |
| 303 | 23 / 38 | 17784 | 14.21 | 92,57 | 92,41 |
| 404 | 24 / 39 | 18252 | 15.76 | 92,80 | 92,61 |
| 505 | 27 / 41 | 19188 | 16.68 | 92,66 | 92,57 |

Checkpoint representativo escolhido pela **validação**, seed 303: `results/transfer_learning/poly_disc/consolidation_20261001_214734_fa76e3/htsat/final/seed_303`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

## Desempenho por classe

| Efeito | Suporte por seed | PaSST F1 (%) | PaSST recall (%) | HTS-AT F1 (%) | HTS-AT recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 808 | 1700 | 53,31 ± 3,12 | 57,09 ± 7,39 | 75,73 ± 0,73 | 79,96 ± 4,32 |
| BD2 | 1700 | 97,77 ± 0,06 | 97,15 ± 0,12 | 99,28 ± 0,08 | 99,27 ± 0,05 |
| BMF | 1700 | 99,65 ± 0,03 | 99,42 ± 0,06 | 99,96 ± 0,02 | 99,92 ± 0,03 |
| DPL | 340 | 96,26 ± 0,30 | 96,24 ± 0,25 | 97,91 ± 0,26 | 98,06 ± 0,39 |
| DS1 | 1700 | 96,97 ± 0,06 | 96,87 ± 0,08 | 98,59 ± 0,27 | 98,51 ± 0,23 |
| FFC | 425 | 97,97 ± 0,06 | 98,92 ± 0,13 | 99,62 ± 0,13 | 99,95 ± 0,11 |
| MGS | 1700 | 92,14 ± 0,17 | 91,45 ± 0,38 | 96,40 ± 0,25 | 97,09 ± 0,51 |
| OD1 | 340 | 23,73 ± 3,59 | 15,47 ± 2,92 | 72,85 ± 0,71 | 65,76 ± 1,29 |
| RAT | 1700 | 95,48 ± 0,13 | 96,99 ± 0,05 | 98,85 ± 0,07 | 98,84 ± 0,21 |
| RBM | 1700 | 99,79 ± 0,02 | 99,88 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 |
| SD1 | 1700 | 83,25 ± 0,30 | 88,96 ± 0,64 | 91,57 ± 0,14 | 93,14 ± 0,64 |
| TS9 | 1700 | 46,63 ± 4,65 | 43,72 ± 7,77 | 72,89 ± 1,71 | 68,53 ± 4,79 |
| VTB | 425 | 99,23 ± 0,07 | 99,48 ± 0,31 | 99,79 ± 0,10 | 100,00 ± 0,00 |

## Comparação descritiva e limites


Não houve rodada preliminar de dez épocas neste cenário. A seleção usou somente validação, e o teste deste protocolo foi avaliado somente após o registro de todas as dez cabeças finais. Existem resultados históricos de outros modelos neste cenário; esta declaração não afirma um holdout externo ao projeto inteiro.

O DP é entre seeds de treinamento na mesma partição; não mede generalização entre novas fontes. Os cenários contínuo/discreto podem compartilhar fontes dentro da mesma família mono/poly. Aqui cada cenário tem sua própria cabeça e avaliação, sem treinar entre cenários. A comparação com o baseline histórico é descritiva: ele usa divisão por arquivo e outro frontend.

## Integridade e reuso

As dez cabeças foram recarregadas e suas predições reproduziram exatamente os arquivos salvos (16830 por cabeça). Hashes dos heads/scalers/caches, ausência de métricas de teste na seleção e ordem dos registros de seleção/treino/teste foram conferidos.

O resumo portátil está em [transfer_learning_poly_disc_consolidated_summary.json](transfer_learning_poly_disc_consolidated_summary.json). Dados, embeddings, checkpoints, predições por arquivo e gráficos permanecem locais e ignorados pelo Git.

Para consultar sem treinar: `notebooks/transfer_learning_all_scenarios.ipynb`. Para inferência com WAVs mono já reamostrados/cortados para 32 kHz/64.000 amostras, use `src.models.consolidated_probe.load_consolidated_probe`, passando o diretório da execução e dispositivo CUDA. Carregar a cabeça junto ao scaler salvo.

O carregador de inferência também foi conferido em 16 WAVs originais por modelo (primeiro batch do manifest): logits finitos e idênticos aos obtidos com o cache, com diferença máxima zero. Essa conferência valida o reuso do pipeline, não uma nova estimativa de desempenho.
