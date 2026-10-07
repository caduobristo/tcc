# Transfer learning consolidado: PaSST e HTS-AT — Poly Continuous — sem TS9

Execução em 02/10/2026, com encoders congelados. Fine-tuning não foi realizado.

Sessão local: `results/transfer_learning/ablations/no_ts9/poly_cont/consolidation_20261002_130417_b0fb96`. Duração do protocolo: **9.72 minutos**, reutilizando os embeddings já extraídos.

120,000 áudios Poly Continuous/12 efeitos; 420 fontes; treino/validação/teste 86,230/9,260/24,510 por fonte (seed 42). WAVs a 32 kHz com frontend nativo, não os mel32 Kaldi.

## Resultado no teste

| Modelo | Acurácia (%) | F1 macro (%) | Precisão macro (%) | Recall macro (%) |
| --- | ---: | ---: | ---: | ---: |
| PaSST | 92,47 ± 0,10 | 92,37 ± 0,08 | 92,47 ± 0,16 | 92,45 ± 0,10 |
| HTS-AT | 97,69 ± 0,10 | 97,69 ± 0,10 | 97,70 ± 0,09 | 97,69 ± 0,09 |

Média ± desvio padrão amostral de cinco seeds novas (101, 202, 303, 404 e 505). Cada seed é avaliada nos mesmos 24,510 áudios; as cinco avaliações não são novos exemplos independentes.

## Seleção e execuções

Oito configurações por modelo foram comparadas usando F1 macro de validação. As duas melhores receberam mais duas seeds; a configuração foi escolhida pela média nas seeds 42, 7 e 21. A seleção foi registrada antes dos dez treinamentos finais, e todas as cabeças foram finalizadas antes de avaliar o teste.

AdamW: LR inicial 0,001, weight decay 0,01; teto de 200 épocas; scheduler e parada antecipada na validação. Consulte [a configuração](../../configs/linear_probe/consolidation_no_ts9_poly_cont.json).

Classes: `808, BD2, BMF, DPL, DS1, FFC, MGS, OD1, RAT, RBM, SD1, VTB`. Excluídos 10,000 WAVs de TS9 do treino, validação e teste. Todas as outras linhas mantêm sua partição original; não houve novo sorteio. Cabeças novas de 9,228 parâmetros, treinadas do início. Os caches originais foram reutilizados após validação; nenhuma extração ou fine-tuning foi necessário.

### PaSST

Configuração escolhida: `none_train_balanced_b128`. F1 macro de validação nas três seeds: **92,30 ± 0,05%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| none_train_balanced_b128 | 92,33 | 32 | 47 |
| none_none_b128 | 92,32 | 32 | 47 |
| train_standardize_none_b1024 | 92,08 | 28 | 43 |
| train_standardize_train_balanced_b1024 | 92,07 | 28 | 43 |
| train_standardize_none_b128 | 91,97 | 8 | 23 |
| train_standardize_train_balanced_b128 | 91,93 | 8 | 23 |
| none_none_b1024 | 91,80 | 60 | 62 |
| none_train_balanced_b1024 | 91,66 | 67 | 82 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| none_train_balanced_b128 | 92,30 ± 0,05 |
| none_none_b128 | 92,30 ± 0,05 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 28 / 43 | 28982 | 19.77 | 92,42 | 92,30 |
| 202 | 30 / 30 | 20220 | 14.70 | 92,41 | 92,33 |
| 303 | 28 / 43 | 28982 | 20.20 | 92,44 | 92,34 |
| 404 | 39 / 42 | 28308 | 19.38 | 92,43 | 92,38 |
| 505 | 37 / 52 | 35048 | 24.06 | 92,64 | 92,50 |

Checkpoint representativo escolhido pela **validação**, seed 101: `results/transfer_learning/ablations/no_ts9/poly_cont/consolidation_20261002_130417_b0fb96/passt/final/seed_101`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

### HTS-AT

Configuração escolhida: `train_standardize_none_b128`. F1 macro de validação nas três seeds: **97,42 ± 0,01%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_none_b128 | 97,42 | 28 | 43 |
| train_standardize_train_balanced_b128 | 97,42 | 28 | 43 |
| train_standardize_none_b1024 | 97,06 | 52 | 54 |
| train_standardize_train_balanced_b1024 | 97,06 | 52 | 54 |
| none_none_b128 | 95,54 | 63 | 73 |
| none_train_balanced_b128 | 95,50 | 71 | 71 |
| none_none_b1024 | 94,22 | 102 | 108 |
| none_train_balanced_b1024 | 94,21 | 102 | 108 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_none_b128 | 97,42 ± 0,01 |
| train_standardize_train_balanced_b128 | 97,42 ± 0,01 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 45 / 46 | 31004 | 20.57 | 97,80 | 97,79 |
| 202 | 33 / 40 | 26960 | 17.96 | 97,62 | 97,62 |
| 303 | 34 / 42 | 28308 | 19.77 | 97,68 | 97,68 |
| 404 | 29 / 44 | 29656 | 19.58 | 97,58 | 97,57 |
| 505 | 50 / 51 | 34374 | 23.58 | 97,78 | 97,77 |

Checkpoint representativo escolhido pela **validação**, seed 303: `results/transfer_learning/ablations/no_ts9/poly_cont/consolidation_20261002_130417_b0fb96/htsat/final/seed_303`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

## Desempenho por classe

| Efeito | Suporte por seed | PaSST F1 (%) | PaSST recall (%) | HTS-AT F1 (%) | HTS-AT recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 808 | 2074 | 99,14 ± 0,03 | 99,81 ± 0,09 | 98,55 ± 0,18 | 98,44 ± 0,19 |
| BD2 | 1995 | 98,24 ± 0,08 | 97,48 ± 0,22 | 99,32 ± 0,04 | 99,24 ± 0,07 |
| BMF | 2023 | 99,62 ± 0,14 | 99,26 ± 0,28 | 99,98 ± 0,00 | 99,95 ± 0,00 |
| DPL | 2025 | 98,72 ± 0,17 | 98,40 ± 0,57 | 99,43 ± 0,06 | 99,45 ± 0,08 |
| DS1 | 2090 | 98,33 ± 0,11 | 98,59 ± 0,20 | 99,14 ± 0,08 | 99,49 ± 0,06 |
| FFC | 2042 | 98,68 ± 0,10 | 98,78 ± 0,12 | 99,91 ± 0,01 | 99,99 ± 0,02 |
| MGS | 2075 | 91,10 ± 0,27 | 89,51 ± 1,17 | 97,19 ± 0,12 | 96,47 ± 0,17 |
| OD1 | 2021 | 68,08 ± 1,88 | 72,55 ± 5,96 | 91,08 ± 0,31 | 93,30 ± 0,73 |
| RAT | 2039 | 94,98 ± 0,16 | 97,37 ± 0,38 | 99,22 ± 0,05 | 98,94 ± 0,11 |
| RBM | 2023 | 99,80 ± 0,11 | 100,00 ± 0,00 | 99,98 ± 0,00 | 99,95 ± 0,00 |
| SD1 | 2039 | 62,37 ± 1,39 | 57,82 ± 4,56 | 88,51 ± 0,55 | 87,04 ± 1,40 |
| VTB | 2064 | 99,36 ± 0,07 | 99,80 ± 0,08 | 99,94 ± 0,01 | 100,00 ± 0,00 |

## Comparação descritiva e limites

Comparação com 13 classes: [transfer_learning_poly_cont_consolidated_report.md](transfer_learning_poly_cont_consolidated_report.md). Remover uma classe muda o conjunto de teste, o número de saídas, os pesos/estatísticas ajustados no treino e pode mudar a configuração selecionada. A diferença é descritiva; não representa melhoria no problema original de 13 classes. O cenário sem TS9 é uma ablação exploratória motivada por resultados já conhecidos.

**O teste já foi consultado anteriormente.** A escolha nesta consolidação usa somente validação, mas não equivale a um teste totalmente cego.

O DP é entre seeds de treinamento na mesma partição; não mede generalização entre novas fontes. Os cenários contínuo/discreto podem compartilhar fontes dentro da mesma família mono/poly. Aqui cada cenário tem sua própria cabeça e avaliação, sem treinar entre cenários. A comparação com o baseline histórico é descritiva: ele usa divisão por arquivo e outro frontend.

## Integridade e reuso

As dez cabeças foram recarregadas e suas predições reproduziram exatamente os arquivos salvos (24510 por cabeça). Hashes dos heads/scalers/caches, ausência de métricas de teste na seleção e ordem dos registros de seleção/treino/teste foram conferidos.

O resumo portátil está em [transfer_learning_no_ts9_poly_cont_consolidated_summary.json](transfer_learning_no_ts9_poly_cont_consolidated_summary.json). Dados, embeddings, checkpoints, predições por arquivo e gráficos permanecem locais e ignorados pelo Git.

Para consultar sem treinar: `notebooks/transfer_learning_no_ts9.ipynb`. Para inferência com WAVs mono já reamostrados/cortados para 32 kHz/64.000 amostras, use `src.models.consolidated_probe.load_consolidated_probe`, passando o diretório da execução e dispositivo CUDA. Carregar a cabeça junto ao scaler salvo e consultar `probe.classes` para interpretar as saídas.

O carregador de inferência também foi conferido em 16 WAVs selecionados por modelo: logits finitos e idênticos aos obtidos com o cache, com diferença máxima zero. Essa conferência valida o reuso do pipeline, não uma nova estimativa de desempenho.
