# Transfer learning consolidado: PaSST e HTS-AT — Mono Continuous

Execução em 01/10/2026, com encoders congelados. Fine-tuning não foi realizado.

Sessão local: `results/transfer_learning/mono_cont/consolidation_20261001_204120_f0174b`. Duração do protocolo: **12.83 minutos**, reutilizando os embeddings já extraídos.

130,000 áudios Mono Continuous/13 efeitos; 624 fontes; treino/validação/teste 93,575/10,012/26,413 por fonte (seed 42). WAVs a 32 kHz com frontend nativo, não os mel32 Kaldi.

## Resultado no teste

| Modelo | Acurácia (%) | F1 macro (%) | Precisão macro (%) | Recall macro (%) |
| --- | ---: | ---: | ---: | ---: |
| PaSST | 84,12 ± 0,06 | 83,95 ± 0,10 | 83,97 ± 0,10 | 84,03 ± 0,08 |
| HTS-AT | 91,67 ± 0,03 | 91,59 ± 0,03 | 91,58 ± 0,03 | 91,62 ± 0,02 |

Média ± desvio padrão amostral de cinco seeds novas (101, 202, 303, 404 e 505). Cada seed é avaliada nos mesmos 26,413 áudios; as cinco avaliações não são novos exemplos independentes.

## Seleção e execuções

Oito configurações por modelo foram comparadas usando F1 macro de validação. As duas melhores receberam mais duas seeds; a configuração foi escolhida pela média nas seeds 42, 7 e 21. A seleção foi registrada antes dos dez treinamentos finais, e todas as cabeças foram finalizadas antes de avaliar o teste.

AdamW: LR inicial 0,001, weight decay 0,01; teto de 200 épocas; scheduler e parada antecipada na validação. Protocolo local: `docs/transfer_learning_all_scenarios.md` (fora do Git). Consulte [a configuração](../configs/linear_probe/consolidation_mono_cont.json).

### PaSST

Configuração escolhida: `train_standardize_train_balanced_b128`. F1 macro de validação nas três seeds: **84,23 ± 0,03%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_train_balanced_b128 | 84,26 | 25 | 40 |
| train_standardize_none_b128 | 84,22 | 25 | 40 |
| none_none_b128 | 83,93 | 41 | 52 |
| none_train_balanced_b128 | 83,80 | 31 | 46 |
| train_standardize_train_balanced_b1024 | 83,76 | 41 | 41 |
| train_standardize_none_b1024 | 83,73 | 34 | 40 |
| none_none_b1024 | 83,35 | 68 | 83 |
| none_train_balanced_b1024 | 83,30 | 64 | 71 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_train_balanced_b128 | 84,23 ± 0,03 |
| train_standardize_none_b128 | 84,21 ± 0,09 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 43 / 50 | 36600 | 27.33 | 84,10 | 83,97 |
| 202 | 38 / 49 | 35868 | 27.33 | 84,17 | 84,03 |
| 303 | 32 / 47 | 34404 | 25.51 | 84,18 | 84,02 |
| 404 | 38 / 53 | 38796 | 27.07 | 84,13 | 83,97 |
| 505 | 20 / 35 | 25620 | 16.52 | 84,03 | 83,79 |

Checkpoint representativo escolhido pela **validação**, seed 101: `results/transfer_learning/mono_cont/consolidation_20261001_204120_f0174b/passt/final/seed_101`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

### HTS-AT

Configuração escolhida: `train_standardize_train_balanced_b128`. F1 macro de validação nas três seeds: **92,52 ± 0,05%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_train_balanced_b128 | 92,56 | 26 | 41 |
| train_standardize_none_b128 | 92,55 | 26 | 41 |
| train_standardize_train_balanced_b1024 | 92,26 | 55 | 57 |
| train_standardize_none_b1024 | 92,24 | 55 | 57 |
| none_none_b128 | 90,55 | 64 | 79 |
| none_train_balanced_b128 | 90,54 | 69 | 72 |
| none_train_balanced_b1024 | 89,03 | 82 | 82 |
| none_none_b1024 | 88,99 | 78 | 82 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_train_balanced_b128 | 92,52 ± 0,05 |
| train_standardize_none_b128 | 92,51 ± 0,06 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 41 / 43 | 31476 | 20.68 | 91,69 | 91,61 |
| 202 | 43 / 43 | 31476 | 21.93 | 91,66 | 91,59 |
| 303 | 35 / 45 | 32940 | 21.21 | 91,70 | 91,62 |
| 404 | 33 / 39 | 28548 | 18.09 | 91,67 | 91,60 |
| 505 | 30 / 45 | 32940 | 21.38 | 91,63 | 91,55 |

Checkpoint representativo escolhido pela **validação**, seed 505: `results/transfer_learning/mono_cont/consolidation_20261001_204120_f0174b/htsat/final/seed_505`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

## Desempenho por classe

| Efeito | Suporte por seed | PaSST F1 (%) | PaSST recall (%) | HTS-AT F1 (%) | HTS-AT recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 808 | 2092 | 52,69 ± 1,72 | 52,91 ± 3,77 | 74,05 ± 0,30 | 73,64 ± 0,89 |
| BD2 | 2066 | 95,42 ± 0,21 | 95,38 ± 0,39 | 97,93 ± 0,08 | 97,58 ± 0,25 |
| BMF | 2025 | 99,45 ± 0,18 | 99,62 ± 0,18 | 99,92 ± 0,01 | 99,96 ± 0,02 |
| DPL | 2008 | 97,27 ± 0,18 | 97,23 ± 0,19 | 98,04 ± 0,08 | 98,31 ± 0,10 |
| DS1 | 2145 | 96,09 ± 0,10 | 97,50 ± 0,29 | 98,06 ± 0,06 | 99,06 ± 0,04 |
| FFC | 2003 | 95,88 ± 0,07 | 95,63 ± 0,19 | 99,18 ± 0,05 | 99,21 ± 0,07 |
| MGS | 1999 | 94,11 ± 0,27 | 94,70 ± 0,37 | 92,57 ± 0,20 | 93,93 ± 0,31 |
| OD1 | 2037 | 61,06 ± 0,78 | 63,42 ± 1,90 | 82,72 ± 0,33 | 82,78 ± 0,88 |
| RAT | 2061 | 93,67 ± 0,18 | 93,89 ± 0,21 | 97,88 ± 0,07 | 97,26 ± 0,10 |
| RBM | 2031 | 99,53 ± 0,06 | 99,60 ± 0,13 | 99,81 ± 0,04 | 99,87 ± 0,03 |
| SD1 | 2002 | 58,17 ± 0,85 | 54,45 ± 2,08 | 77,43 ± 0,18 | 77,19 ± 0,30 |
| TS9 | 1947 | 48,77 ± 1,75 | 48,73 ± 3,63 | 73,95 ± 0,48 | 73,09 ± 1,26 |
| VTB | 1997 | 99,29 ± 0,09 | 99,34 ± 0,17 | 99,20 ± 0,06 | 99,19 ± 0,21 |

## Comparação descritiva e limites


Não houve rodada preliminar de dez épocas neste cenário. A seleção usou somente validação, e o teste deste protocolo foi avaliado somente após o registro de todas as dez cabeças finais. Existem resultados históricos de outros modelos neste cenário; esta declaração não afirma um holdout externo ao projeto inteiro.

O DP é entre seeds de treinamento na mesma partição; não mede generalização entre novas fontes. Os cenários contínuo/discreto podem compartilhar fontes dentro da mesma família mono/poly. Aqui cada cenário tem sua própria cabeça e avaliação, sem treinar entre cenários. A comparação com o baseline histórico é descritiva: ele usa divisão por arquivo e outro frontend.

## Integridade e reuso

As dez cabeças foram recarregadas e suas predições reproduziram exatamente os arquivos salvos (26413 por cabeça). Hashes dos heads/scalers/caches, ausência de métricas de teste na seleção e ordem dos registros de seleção/treino/teste foram conferidos.

O resumo portátil está em [transfer_learning_mono_cont_consolidated_summary.json](transfer_learning_mono_cont_consolidated_summary.json). Dados, embeddings, checkpoints, predições por arquivo e gráficos permanecem locais e ignorados pelo Git.

Para consultar sem treinar: `notebooks/transfer_learning_all_scenarios.ipynb`. Para inferência com WAVs mono já reamostrados/cortados para 32 kHz/64.000 amostras, use `src.models.consolidated_probe.load_consolidated_probe`, passando o diretório da execução e dispositivo CUDA. Carregar a cabeça junto ao scaler salvo.

O carregador de inferência também foi conferido em 16 WAVs originais por modelo (primeiro batch do manifest): logits finitos e idênticos aos obtidos com o cache, com diferença máxima zero. Essa conferência valida o reuso do pipeline, não uma nova estimativa de desempenho.
