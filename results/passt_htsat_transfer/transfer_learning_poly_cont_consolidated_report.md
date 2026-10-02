# Transfer learning consolidado: PaSST e HTS-AT — Poly Continuous

Execução em 02/10/2026, com encoders congelados. Fine-tuning não foi realizado.

Sessão local: `results/transfer_learning/poly_cont/consolidation_20261002_095702_4bcead`. Duração do protocolo: **10.64 minutos**, reutilizando os embeddings já extraídos.

130,000 áudios Poly Continuous/13 efeitos; 420 fontes; treino/validação/teste 93,402/10,064/26,534 por fonte (seed 42). WAVs a 32 kHz com frontend nativo, não os mel32 Kaldi.

## Resultado no teste

| Modelo | Acurácia (%) | F1 macro (%) | Precisão macro (%) | Recall macro (%) |
| --- | ---: | ---: | ---: | ---: |
| PaSST | 84,91 ± 0,26 | 84,84 ± 0,20 | 85,05 ± 0,31 | 84,92 ± 0,27 |
| HTS-AT | 93,88 ± 0,12 | 93,89 ± 0,12 | 93,91 ± 0,11 | 93,89 ± 0,12 |

Média ± desvio padrão amostral de cinco seeds novas (101, 202, 303, 404 e 505). Cada seed é avaliada nos mesmos 26,534 áudios; as cinco avaliações não são novos exemplos independentes.

## Seleção e execuções

Oito configurações por modelo foram comparadas usando F1 macro de validação. As duas melhores receberam mais duas seeds; a configuração foi escolhida pela média nas seeds 42, 7 e 21. A seleção foi registrada antes dos dez treinamentos finais, e todas as cabeças foram finalizadas antes de avaliar o teste.

AdamW: LR inicial 0,001, weight decay 0,01; teto de 200 épocas; scheduler e parada antecipada na validação. Protocolo local: `docs/transfer_learning_all_scenarios.md` (fora do Git). Consulte [a configuração](../configs/linear_probe/consolidation_poly_cont.json).

### PaSST

Configuração escolhida: `none_train_balanced_b128`. F1 macro de validação nas três seeds: **85,40 ± 0,20%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| none_train_balanced_b128 | 85,53 | 41 | 56 |
| none_none_b128 | 85,40 | 53 | 68 |
| none_train_balanced_b1024 | 85,09 | 84 | 88 |
| train_standardize_train_balanced_b1024 | 84,91 | 21 | 36 |
| train_standardize_none_b1024 | 84,89 | 21 | 36 |
| none_none_b1024 | 84,88 | 70 | 75 |
| train_standardize_none_b128 | 84,67 | 19 | 21 |
| train_standardize_train_balanced_b128 | 84,65 | 21 | 21 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| none_train_balanced_b128 | 85,40 ± 0,20 |
| none_none_b128 | 85,37 ± 0,14 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 32 / 47 | 34310 | 22.15 | 85,10 | 85,03 |
| 202 | 18 / 28 | 20440 | 13.57 | 85,11 | 84,99 |
| 303 | 37 / 38 | 27740 | 18.78 | 84,83 | 84,76 |
| 404 | 28 / 43 | 31390 | 22.27 | 84,48 | 84,54 |
| 505 | 42 / 44 | 32120 | 22.28 | 85,04 | 84,88 |

Checkpoint representativo escolhido pela **validação**, seed 101: `results/transfer_learning/poly_cont/consolidation_20261002_095702_4bcead/passt/final/seed_101`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

### HTS-AT

Configuração escolhida: `train_standardize_train_balanced_b128`. F1 macro de validação nas três seeds: **93,36 ± 0,06%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_train_balanced_b128 | 93,38 | 25 | 40 |
| train_standardize_none_b128 | 93,38 | 38 | 40 |
| train_standardize_none_b1024 | 93,15 | 66 | 66 |
| train_standardize_train_balanced_b1024 | 92,91 | 51 | 65 |
| none_none_b128 | 91,16 | 66 | 75 |
| none_train_balanced_b128 | 91,06 | 61 | 63 |
| none_train_balanced_b1024 | 90,04 | 106 | 110 |
| none_none_b1024 | 90,03 | 96 | 111 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_train_balanced_b128 | 93,36 ± 0,06 |
| train_standardize_none_b128 | 93,36 ± 0,04 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 38 / 39 | 28470 | 21.29 | 93,89 | 93,90 |
| 202 | 35 / 35 | 25550 | 18.57 | 93,77 | 93,78 |
| 303 | 47 / 62 | 45260 | 32.70 | 94,02 | 94,03 |
| 404 | 32 / 34 | 24820 | 17.49 | 93,76 | 93,76 |
| 505 | 46 / 46 | 33580 | 23.41 | 93,97 | 93,98 |

Checkpoint representativo escolhido pela **validação**, seed 303: `results/transfer_learning/poly_cont/consolidation_20261002_095702_4bcead/htsat/final/seed_303`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

## Desempenho por classe

| Efeito | Suporte por seed | PaSST F1 (%) | PaSST recall (%) | HTS-AT F1 (%) | HTS-AT recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 808 | 2074 | 49,72 ± 3,56 | 51,09 ± 7,85 | 74,34 ± 0,36 | 75,96 ± 1,45 |
| BD2 | 1995 | 98,10 ± 0,24 | 97,04 ± 0,64 | 99,27 ± 0,02 | 99,19 ± 0,04 |
| BMF | 2023 | 99,68 ± 0,05 | 99,37 ± 0,11 | 99,98 ± 0,00 | 99,95 ± 0,00 |
| DPL | 2025 | 98,65 ± 0,31 | 98,42 ± 0,65 | 99,44 ± 0,04 | 99,56 ± 0,03 |
| DS1 | 2090 | 98,30 ± 0,12 | 98,07 ± 0,30 | 99,24 ± 0,06 | 99,40 ± 0,04 |
| FFC | 2042 | 98,42 ± 0,14 | 99,03 ± 0,24 | 99,86 ± 0,03 | 99,97 ± 0,03 |
| MGS | 2075 | 90,96 ± 0,54 | 88,53 ± 1,71 | 96,89 ± 0,26 | 96,41 ± 0,63 |
| OD1 | 2021 | 65,80 ± 3,52 | 68,01 ± 9,96 | 90,72 ± 0,26 | 92,05 ± 0,36 |
| RAT | 2039 | 94,70 ± 0,24 | 97,56 ± 0,34 | 99,28 ± 0,04 | 99,01 ± 0,10 |
| RBM | 2023 | 99,86 ± 0,05 | 99,99 ± 0,02 | 99,98 ± 0,00 | 99,95 ± 0,00 |
| SD1 | 2039 | 62,86 ± 2,03 | 61,23 ± 7,87 | 88,43 ± 0,39 | 87,79 ± 0,88 |
| TS9 | 2024 | 46,52 ± 5,19 | 45,90 ± 8,95 | 73,17 ± 0,67 | 71,35 ± 1,62 |
| VTB | 2064 | 99,35 ± 0,25 | 99,79 ± 0,07 | 99,95 ± 0,00 | 100,00 ± 0,00 |

## Comparação descritiva e limites


Não houve rodada preliminar de dez épocas neste cenário. A seleção usou somente validação, e o teste deste protocolo foi avaliado somente após o registro de todas as dez cabeças finais. Existem resultados históricos de outros modelos neste cenário; esta declaração não afirma um holdout externo ao projeto inteiro.

O DP é entre seeds de treinamento na mesma partição; não mede generalização entre novas fontes. Os cenários contínuo/discreto podem compartilhar fontes dentro da mesma família mono/poly. Aqui cada cenário tem sua própria cabeça e avaliação, sem treinar entre cenários. A comparação com o baseline histórico é descritiva: ele usa divisão por arquivo e outro frontend.

## Integridade e reuso

As dez cabeças foram recarregadas e suas predições reproduziram exatamente os arquivos salvos (26534 por cabeça). Hashes dos heads/scalers/caches, ausência de métricas de teste na seleção e ordem dos registros de seleção/treino/teste foram conferidos.

O resumo portátil está em [transfer_learning_poly_cont_consolidated_summary.json](transfer_learning_poly_cont_consolidated_summary.json). Dados, embeddings, checkpoints, predições por arquivo e gráficos permanecem locais e ignorados pelo Git.

Para consultar sem treinar: `notebooks/transfer_learning_all_scenarios.ipynb`. Para inferência com WAVs mono já reamostrados/cortados para 32 kHz/64.000 amostras, use `src.models.consolidated_probe.load_consolidated_probe`, passando o diretório da execução e dispositivo CUDA. Carregar a cabeça junto ao scaler salvo.

O carregador de inferência também foi conferido em 16 WAVs originais por modelo (primeiro batch do manifest): logits finitos e idênticos aos obtidos com o cache, com diferença máxima zero. Essa conferência valida o reuso do pipeline, não uma nova estimativa de desempenho.
