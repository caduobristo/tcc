# Transfer learning consolidado: PaSST e HTS-AT — Mono Discrete — sem TS9

Execução em 02/10/2026, com encoders congelados. Fine-tuning não foi realizado.

Sessão local: `results/transfer_learning/ablations/no_ts9/mono_disc/consolidation_20261002_123615_4c06be`. Duração do protocolo: **10.75 minutos**, reutilizando os embeddings já extraídos.

111,072 áudios Mono Discrete/12 efeitos; 624 fontes; treino/validação/teste 79,922/8,722/22,428 por fonte (seed 42). WAVs a 32 kHz com frontend nativo, não os mel32 Kaldi.

## Resultado no teste

| Modelo | Acurácia (%) | F1 macro (%) | Precisão macro (%) | Recall macro (%) |
| --- | ---: | ---: | ---: | ---: |
| PaSST | 91,11 ± 0,37 | 88,39 ± 0,28 | 88,23 ± 0,07 | 90,29 ± 0,12 |
| HTS-AT | 95,32 ± 0,08 | 93,88 ± 0,09 | 94,27 ± 0,20 | 93,57 ± 0,07 |

Média ± desvio padrão amostral de cinco seeds novas (101, 202, 303, 404 e 505). Cada seed é avaliada nos mesmos 22,428 áudios; as cinco avaliações não são novos exemplos independentes.

## Seleção e execuções

Oito configurações por modelo foram comparadas usando F1 macro de validação. As duas melhores receberam mais duas seeds; a configuração foi escolhida pela média nas seeds 42, 7 e 21. A seleção foi registrada antes dos dez treinamentos finais, e todas as cabeças foram finalizadas antes de avaliar o teste.

AdamW: LR inicial 0,001, weight decay 0,01; teto de 200 épocas; scheduler e parada antecipada na validação. Consulte [a configuração](../../configs/linear_probe/consolidation_no_ts9_mono_disc.json).

Classes: `808, BD2, BMF, DPL, DS1, FFC, MGS, OD1, RAT, RBM, SD1, VTB`. Excluídos 12,480 WAVs de TS9 do treino, validação e teste. Todas as outras linhas mantêm sua partição original; não houve novo sorteio. Cabeças novas de 9,228 parâmetros, treinadas do início. Os caches originais foram reutilizados após validação; nenhuma extração ou fine-tuning foi necessário.

### PaSST

Configuração escolhida: `train_standardize_train_balanced_b128`. F1 macro de validação nas três seeds: **87,45 ± 0,16%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_train_balanced_b128 | 87,57 | 56 | 60 |
| train_standardize_none_b128 | 87,34 | 27 | 39 |
| none_train_balanced_b128 | 87,20 | 48 | 63 |
| none_none_b128 | 87,18 | 45 | 57 |
| train_standardize_none_b1024 | 86,99 | 62 | 77 |
| train_standardize_train_balanced_b1024 | 86,74 | 65 | 69 |
| none_train_balanced_b1024 | 85,45 | 41 | 56 |
| none_none_b1024 | 84,83 | 23 | 38 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_train_balanced_b128 | 87,45 ± 0,16 |
| train_standardize_none_b128 | 87,29 ± 0,18 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 46 / 55 | 34375 | 23.24 | 91,23 | 88,46 |
| 202 | 62 / 62 | 38750 | 27.03 | 91,00 | 88,33 |
| 303 | 43 / 51 | 31875 | 23.05 | 90,61 | 87,98 |
| 404 | 40 / 46 | 28750 | 19.49 | 91,63 | 88,76 |
| 505 | 51 / 66 | 41250 | 28.35 | 91,11 | 88,42 |

Checkpoint representativo escolhido pela **validação**, seed 404: `results/transfer_learning/ablations/no_ts9/mono_disc/consolidation_20261002_123615_4c06be/passt/final/seed_404`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

### HTS-AT

Configuração escolhida: `train_standardize_none_b128`. F1 macro de validação nas três seeds: **94,69 ± 0,07%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_none_b128 | 94,62 | 38 | 45 |
| train_standardize_train_balanced_b128 | 94,26 | 36 | 51 |
| train_standardize_none_b1024 | 93,95 | 51 | 62 |
| train_standardize_train_balanced_b1024 | 93,29 | 36 | 51 |
| none_none_b128 | 90,70 | 55 | 59 |
| none_train_balanced_b128 | 89,93 | 32 | 47 |
| none_none_b1024 | 89,37 | 97 | 105 |
| none_train_balanced_b1024 | 86,75 | 79 | 94 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_none_b128 | 94,69 ± 0,07 |
| train_standardize_train_balanced_b128 | 94,13 ± 0,12 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 48 / 54 | 33750 | 23.33 | 95,29 | 93,87 |
| 202 | 46 / 54 | 33750 | 25.37 | 95,36 | 93,97 |
| 303 | 43 / 58 | 36250 | 26.54 | 95,34 | 93,88 |
| 404 | 45 / 60 | 37500 | 28.42 | 95,41 | 93,96 |
| 505 | 41 / 56 | 35000 | 24.75 | 95,20 | 93,74 |

Checkpoint representativo escolhido pela **validação**, seed 505: `results/transfer_learning/ablations/no_ts9/mono_disc/consolidation_20261002_123615_4c06be/htsat/final/seed_505`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

## Desempenho por classe

| Efeito | Suporte por seed | PaSST F1 (%) | PaSST recall (%) | HTS-AT F1 (%) | HTS-AT recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 808 | 2520 | 95,47 ± 0,12 | 96,18 ± 0,09 | 95,65 ± 0,07 | 95,71 ± 0,16 |
| BD2 | 2520 | 97,97 ± 0,03 | 97,26 ± 0,09 | 98,92 ± 0,06 | 98,48 ± 0,06 |
| BMF | 2520 | 99,08 ± 0,04 | 99,04 ± 0,08 | 99,91 ± 0,03 | 99,93 ± 0,03 |
| DPL | 504 | 89,48 ± 0,46 | 94,33 ± 0,62 | 92,88 ± 0,42 | 91,59 ± 0,55 |
| DS1 | 2520 | 95,90 ± 0,08 | 97,06 ± 0,18 | 97,62 ± 0,11 | 98,00 ± 0,12 |
| FFC | 630 | 94,20 ± 0,51 | 97,90 ± 0,07 | 99,29 ± 0,10 | 99,84 ± 0,00 |
| MGS | 2520 | 93,94 ± 0,07 | 92,27 ± 0,25 | 91,86 ± 0,20 | 92,78 ± 1,03 |
| OD1 | 504 | 36,08 ± 0,65 | 61,90 ± 3,99 | 68,83 ± 0,43 | 64,33 ± 1,30 |
| RAT | 2520 | 92,39 ± 0,10 | 91,89 ± 0,34 | 97,54 ± 0,10 | 97,54 ± 0,10 |
| RBM | 2520 | 99,17 ± 0,04 | 99,37 ± 0,09 | 99,85 ± 0,03 | 99,92 ± 0,00 |
| SD1 | 2520 | 67,78 ± 2,84 | 57,40 ± 4,35 | 84,61 ± 0,34 | 84,87 ± 1,37 |
| VTB | 630 | 99,19 ± 0,09 | 98,89 ± 0,00 | 99,67 ± 0,07 | 99,84 ± 0,00 |

## Comparação descritiva e limites

Comparação com 13 classes: [transfer_learning_consolidated_report.md](transfer_learning_consolidated_report.md). Remover uma classe muda o conjunto de teste, o número de saídas, os pesos/estatísticas ajustados no treino e pode mudar a configuração selecionada. A diferença é descritiva; não representa melhoria no problema original de 13 classes. O cenário sem TS9 é uma ablação exploratória motivada por resultados já conhecidos.

**O teste já foi consultado anteriormente.** A escolha nesta consolidação usa somente validação, mas não equivale a um teste totalmente cego.

O DP é entre seeds de treinamento na mesma partição; não mede generalização entre novas fontes. Os cenários contínuo/discreto podem compartilhar fontes dentro da mesma família mono/poly. Aqui cada cenário tem sua própria cabeça e avaliação, sem treinar entre cenários. A comparação com o baseline histórico é descritiva: ele usa divisão por arquivo e outro frontend.

## Integridade e reuso

As dez cabeças foram recarregadas e suas predições reproduziram exatamente os arquivos salvos (22428 por cabeça). Hashes dos heads/scalers/caches, ausência de métricas de teste na seleção e ordem dos registros de seleção/treino/teste foram conferidos.

O resumo portátil está em [transfer_learning_no_ts9_mono_disc_consolidated_summary.json](transfer_learning_no_ts9_mono_disc_consolidated_summary.json). Dados, embeddings, checkpoints, predições por arquivo e gráficos permanecem locais e ignorados pelo Git.

Para consultar sem treinar: `notebooks/transfer_learning_no_ts9.ipynb`. Para inferência com WAVs mono já reamostrados/cortados para 32 kHz/64.000 amostras, use `src.models.consolidated_probe.load_consolidated_probe`, passando o diretório da execução e dispositivo CUDA. Carregar a cabeça junto ao scaler salvo e consultar `probe.classes` para interpretar as saídas.

O carregador de inferência também foi conferido em 16 WAVs selecionados por modelo: logits finitos e idênticos aos obtidos com o cache, com diferença máxima zero. Essa conferência valida o reuso do pipeline, não uma nova estimativa de desempenho.
