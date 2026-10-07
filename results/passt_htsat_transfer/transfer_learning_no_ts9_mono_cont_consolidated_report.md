# Transfer learning consolidado: PaSST e HTS-AT — Mono Continuous — sem TS9

Execução em 02/10/2026, com encoders congelados. Fine-tuning não foi realizado.

Sessão local: `results/transfer_learning/ablations/no_ts9/mono_cont/consolidation_20261002_124708_862a6a`. Duração do protocolo: **10.20 minutos**, reutilizando os embeddings já extraídos.

120,000 áudios Mono Continuous/12 efeitos; 624 fontes; treino/validação/teste 86,242/9,292/24,466 por fonte (seed 42). WAVs a 32 kHz com frontend nativo, não os mel32 Kaldi.

## Resultado no teste

| Modelo | Acurácia (%) | F1 macro (%) | Precisão macro (%) | Recall macro (%) |
| --- | ---: | ---: | ---: | ---: |
| PaSST | 90,67 ± 0,08 | 90,53 ± 0,07 | 90,50 ± 0,10 | 90,61 ± 0,08 |
| HTS-AT | 94,96 ± 0,07 | 94,91 ± 0,07 | 94,90 ± 0,07 | 94,93 ± 0,07 |

Média ± desvio padrão amostral de cinco seeds novas (101, 202, 303, 404 e 505). Cada seed é avaliada nos mesmos 24,466 áudios; as cinco avaliações não são novos exemplos independentes.

## Seleção e execuções

Oito configurações por modelo foram comparadas usando F1 macro de validação. As duas melhores receberam mais duas seeds; a configuração foi escolhida pela média nas seeds 42, 7 e 21. A seleção foi registrada antes dos dez treinamentos finais, e todas as cabeças foram finalizadas antes de avaliar o teste.

AdamW: LR inicial 0,001, weight decay 0,01; teto de 200 épocas; scheduler e parada antecipada na validação. Consulte [a configuração](../../configs/linear_probe/consolidation_no_ts9_mono_cont.json).

Classes: `808, BD2, BMF, DPL, DS1, FFC, MGS, OD1, RAT, RBM, SD1, VTB`. Excluídos 10,000 WAVs de TS9 do treino, validação e teste. Todas as outras linhas mantêm sua partição original; não houve novo sorteio. Cabeças novas de 9,228 parâmetros, treinadas do início. Os caches originais foram reutilizados após validação; nenhuma extração ou fine-tuning foi necessário.

### PaSST

Configuração escolhida: `train_standardize_train_balanced_b128`. F1 macro de validação nas três seeds: **90,60 ± 0,15%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_train_balanced_b128 | 90,75 | 32 | 47 |
| train_standardize_none_b128 | 90,63 | 42 | 47 |
| none_train_balanced_b128 | 90,58 | 30 | 45 |
| none_none_b128 | 90,57 | 44 | 45 |
| train_standardize_train_balanced_b1024 | 90,18 | 31 | 46 |
| train_standardize_none_b1024 | 90,17 | 46 | 46 |
| none_none_b1024 | 89,71 | 57 | 72 |
| none_train_balanced_b1024 | 89,70 | 57 | 72 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_train_balanced_b128 | 90,60 ± 0,15 |
| train_standardize_none_b128 | 90,55 ± 0,09 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 37 / 52 | 35048 | 24.17 | 90,62 | 90,50 |
| 202 | 37 / 52 | 35048 | 25.46 | 90,57 | 90,42 |
| 303 | 37 / 43 | 28982 | 20.06 | 90,67 | 90,54 |
| 404 | 39 / 54 | 36396 | 26.10 | 90,70 | 90,55 |
| 505 | 36 / 46 | 31004 | 21.27 | 90,78 | 90,63 |

Checkpoint representativo escolhido pela **validação**, seed 202: `results/transfer_learning/ablations/no_ts9/mono_cont/consolidation_20261002_124708_862a6a/passt/final/seed_202`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

### HTS-AT

Configuração escolhida: `train_standardize_none_b128`. F1 macro de validação nas três seeds: **95,18 ± 0,06%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_none_b128 | 95,14 | 28 | 37 |
| train_standardize_train_balanced_b128 | 95,11 | 28 | 37 |
| train_standardize_none_b1024 | 94,90 | 54 | 54 |
| train_standardize_train_balanced_b1024 | 94,89 | 39 | 46 |
| none_train_balanced_b128 | 93,42 | 69 | 71 |
| none_none_b128 | 93,40 | 69 | 71 |
| none_none_b1024 | 92,35 | 101 | 105 |
| none_train_balanced_b1024 | 92,34 | 101 | 105 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_none_b128 | 95,18 ± 0,06 |
| train_standardize_train_balanced_b128 | 95,17 ± 0,06 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 35 / 37 | 24938 | 17.14 | 94,91 | 94,87 |
| 202 | 24 / 34 | 22916 | 16.68 | 94,86 | 94,81 |
| 303 | 40 / 55 | 37070 | 25.72 | 94,96 | 94,91 |
| 404 | 37 / 52 | 35048 | 24.95 | 95,01 | 94,95 |
| 505 | 42 / 57 | 38418 | 26.61 | 95,04 | 94,99 |

Checkpoint representativo escolhido pela **validação**, seed 505: `results/transfer_learning/ablations/no_ts9/mono_cont/consolidation_20261002_124708_862a6a/htsat/final/seed_505`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

## Desempenho por classe

| Efeito | Suporte por seed | PaSST F1 (%) | PaSST recall (%) | HTS-AT F1 (%) | HTS-AT recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 808 | 2092 | 94,62 ± 0,14 | 95,83 ± 0,04 | 95,18 ± 0,21 | 95,38 ± 0,37 |
| BD2 | 2066 | 95,49 ± 0,13 | 95,51 ± 0,23 | 97,88 ± 0,12 | 97,37 ± 0,20 |
| BMF | 2025 | 99,35 ± 0,03 | 99,64 ± 0,02 | 99,97 ± 0,01 | 100,00 ± 0,00 |
| DPL | 2008 | 97,30 ± 0,12 | 97,19 ± 0,11 | 98,08 ± 0,05 | 98,25 ± 0,10 |
| DS1 | 2145 | 96,13 ± 0,01 | 97,56 ± 0,15 | 98,11 ± 0,11 | 99,07 ± 0,17 |
| FFC | 2003 | 95,95 ± 0,12 | 95,66 ± 0,22 | 99,19 ± 0,05 | 99,20 ± 0,14 |
| MGS | 1999 | 94,65 ± 0,14 | 94,43 ± 0,14 | 92,65 ± 0,11 | 94,09 ± 0,39 |
| OD1 | 2037 | 61,48 ± 1,26 | 63,09 ± 2,82 | 83,08 ± 0,56 | 82,88 ± 1,77 |
| RAT | 2061 | 93,89 ± 0,06 | 93,82 ± 0,17 | 97,89 ± 0,02 | 97,17 ± 0,14 |
| RBM | 2031 | 99,57 ± 0,04 | 99,63 ± 0,08 | 99,87 ± 0,02 | 99,90 ± 0,00 |
| SD1 | 2002 | 58,47 ± 0,85 | 55,34 ± 2,23 | 77,71 ± 0,11 | 76,52 ± 1,08 |
| VTB | 1997 | 99,44 ± 0,06 | 99,60 ± 0,08 | 99,26 ± 0,06 | 99,29 ± 0,14 |

## Comparação descritiva e limites

Comparação com 13 classes: [transfer_learning_mono_cont_consolidated_report.md](transfer_learning_mono_cont_consolidated_report.md). Remover uma classe muda o conjunto de teste, o número de saídas, os pesos/estatísticas ajustados no treino e pode mudar a configuração selecionada. A diferença é descritiva; não representa melhoria no problema original de 13 classes. O cenário sem TS9 é uma ablação exploratória motivada por resultados já conhecidos.

**O teste já foi consultado anteriormente.** A escolha nesta consolidação usa somente validação, mas não equivale a um teste totalmente cego.

O DP é entre seeds de treinamento na mesma partição; não mede generalização entre novas fontes. Os cenários contínuo/discreto podem compartilhar fontes dentro da mesma família mono/poly. Aqui cada cenário tem sua própria cabeça e avaliação, sem treinar entre cenários. A comparação com o baseline histórico é descritiva: ele usa divisão por arquivo e outro frontend.

## Integridade e reuso

As dez cabeças foram recarregadas e suas predições reproduziram exatamente os arquivos salvos (24466 por cabeça). Hashes dos heads/scalers/caches, ausência de métricas de teste na seleção e ordem dos registros de seleção/treino/teste foram conferidos.

O resumo portátil está em [transfer_learning_no_ts9_mono_cont_consolidated_summary.json](transfer_learning_no_ts9_mono_cont_consolidated_summary.json). Dados, embeddings, checkpoints, predições por arquivo e gráficos permanecem locais e ignorados pelo Git.

Para consultar sem treinar: `notebooks/transfer_learning_no_ts9.ipynb`. Para inferência com WAVs mono já reamostrados/cortados para 32 kHz/64.000 amostras, use `src.models.consolidated_probe.load_consolidated_probe`, passando o diretório da execução e dispositivo CUDA. Carregar a cabeça junto ao scaler salvo e consultar `probe.classes` para interpretar as saídas.

O carregador de inferência também foi conferido em 16 WAVs selecionados por modelo: logits finitos e idênticos aos obtidos com o cache, com diferença máxima zero. Essa conferência valida o reuso do pipeline, não uma nova estimativa de desempenho.
