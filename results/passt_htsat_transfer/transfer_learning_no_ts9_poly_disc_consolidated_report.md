# Transfer learning consolidado: PaSST e HTS-AT — Poly Discrete — sem TS9

Execução em 02/10/2026, com encoders congelados. Fine-tuning não foi realizado.

Sessão local: `results/transfer_learning/ablations/no_ts9/poly_disc/consolidation_20261002_125724_75d459`. Duração do protocolo: **6.81 minutos**, reutilizando os embeddings já extraídos.

74,760 áudios Poly Discrete/12 efeitos; 420 fontes; treino/validação/teste 53,756/5,874/15,130 por fonte (seed 42). WAVs a 32 kHz com frontend nativo, não os mel32 Kaldi.

## Resultado no teste

| Modelo | Acurácia (%) | F1 macro (%) | Precisão macro (%) | Recall macro (%) |
| --- | ---: | ---: | ---: | ---: |
| PaSST | 93,10 ± 0,35 | 90,31 ± 0,32 | 89,89 ± 0,45 | 91,04 ± 0,31 |
| HTS-AT | 97,73 ± 0,15 | 96,45 ± 0,23 | 96,58 ± 0,29 | 96,35 ± 0,20 |

Média ± desvio padrão amostral de cinco seeds novas (101, 202, 303, 404 e 505). Cada seed é avaliada nos mesmos 15,130 áudios; as cinco avaliações não são novos exemplos independentes.

## Seleção e execuções

Oito configurações por modelo foram comparadas usando F1 macro de validação. As duas melhores receberam mais duas seeds; a configuração foi escolhida pela média nas seeds 42, 7 e 21. A seleção foi registrada antes dos dez treinamentos finais, e todas as cabeças foram finalizadas antes de avaliar o teste.

AdamW: LR inicial 0,001, weight decay 0,01; teto de 200 épocas; scheduler e parada antecipada na validação. Consulte [a configuração](../../configs/linear_probe/consolidation_no_ts9_poly_disc.json).

Classes: `808, BD2, BMF, DPL, DS1, FFC, MGS, OD1, RAT, RBM, SD1, VTB`. Excluídos 8,400 WAVs de TS9 do treino, validação e teste. Todas as outras linhas mantêm sua partição original; não houve novo sorteio. Cabeças novas de 9,228 parâmetros, treinadas do início. Os caches originais foram reutilizados após validação; nenhuma extração ou fine-tuning foi necessário.

### PaSST

Configuração escolhida: `none_train_balanced_b128`. F1 macro de validação nas três seeds: **90,27 ± 0,22%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| none_train_balanced_b128 | 90,53 | 42 | 57 |
| train_standardize_none_b128 | 90,24 | 38 | 53 |
| train_standardize_train_balanced_b128 | 90,15 | 39 | 43 |
| train_standardize_train_balanced_b1024 | 89,51 | 49 | 51 |
| none_none_b128 | 89,34 | 32 | 47 |
| none_train_balanced_b1024 | 88,69 | 59 | 74 |
| train_standardize_none_b1024 | 88,09 | 29 | 39 |
| none_none_b1024 | 87,18 | 66 | 69 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| none_train_balanced_b128 | 90,27 ± 0,22 |
| train_standardize_none_b128 | 90,05 ± 0,19 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 51 / 66 | 27720 | 19.64 | 93,17 | 90,62 |
| 202 | 46 / 50 | 21000 | 15.23 | 92,70 | 89,92 |
| 303 | 36 / 51 | 21420 | 14.79 | 93,01 | 90,17 |
| 404 | 42 / 57 | 23940 | 16.51 | 92,95 | 90,16 |
| 505 | 49 / 53 | 22260 | 16.58 | 93,65 | 90,67 |

Checkpoint representativo escolhido pela **validação**, seed 505: `results/transfer_learning/ablations/no_ts9/poly_disc/consolidation_20261002_125724_75d459/passt/final/seed_505`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

### HTS-AT

Configuração escolhida: `train_standardize_none_b128`. F1 macro de validação nas três seeds: **96,12 ± 0,11%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_none_b128 | 96,03 | 32 | 32 |
| train_standardize_train_balanced_b128 | 95,81 | 23 | 38 |
| train_standardize_none_b1024 | 95,60 | 62 | 62 |
| train_standardize_train_balanced_b1024 | 95,01 | 51 | 54 |
| none_none_b128 | 93,20 | 57 | 69 |
| none_train_balanced_b128 | 92,67 | 51 | 66 |
| none_none_b1024 | 90,40 | 84 | 99 |
| none_train_balanced_b1024 | 89,63 | 75 | 90 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_none_b128 | 96,12 ± 0,11 |
| train_standardize_train_balanced_b128 | 95,65 ± 0,16 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 39 / 54 | 22680 | 15.57 | 97,81 | 96,57 |
| 202 | 25 / 40 | 16800 | 11.51 | 97,76 | 96,42 |
| 303 | 43 / 58 | 24360 | 17.98 | 97,85 | 96,62 |
| 404 | 32 / 47 | 19740 | 13.94 | 97,48 | 96,05 |
| 505 | 36 / 39 | 16380 | 11.26 | 97,75 | 96,59 |

Checkpoint representativo escolhido pela **validação**, seed 303: `results/transfer_learning/ablations/no_ts9/poly_disc/consolidation_20261002_125724_75d459/htsat/final/seed_303`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

## Desempenho por classe

| Efeito | Suporte por seed | PaSST F1 (%) | PaSST recall (%) | HTS-AT F1 (%) | HTS-AT recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 808 | 1700 | 99,14 ± 0,03 | 99,20 ± 0,10 | 98,72 ± 0,09 | 98,68 ± 0,23 |
| BD2 | 1700 | 97,56 ± 0,12 | 96,44 ± 0,47 | 99,28 ± 0,05 | 99,38 ± 0,09 |
| BMF | 1700 | 99,37 ± 0,12 | 99,01 ± 0,20 | 99,95 ± 0,01 | 99,89 ± 0,03 |
| DPL | 340 | 93,91 ± 1,12 | 96,88 ± 0,16 | 98,10 ± 0,23 | 98,88 ± 0,38 |
| DS1 | 1700 | 96,74 ± 0,24 | 96,31 ± 0,39 | 98,70 ± 0,09 | 98,35 ± 0,40 |
| FFC | 425 | 97,70 ± 0,20 | 98,87 ± 0,69 | 99,53 ± 0,08 | 99,86 ± 0,13 |
| MGS | 1700 | 90,22 ± 0,42 | 88,08 ± 0,49 | 97,14 ± 0,18 | 96,67 ± 0,60 |
| OD1 | 340 | 38,15 ± 1,28 | 46,65 ± 5,98 | 74,73 ± 2,00 | 71,71 ± 1,35 |
| RAT | 1700 | 94,75 ± 0,40 | 96,64 ± 0,81 | 98,92 ± 0,07 | 98,85 ± 0,18 |
| RBM | 1700 | 99,47 ± 0,07 | 99,74 ± 0,09 | 100,00 ± 0,00 | 100,00 ± 0,00 |
| SD1 | 1700 | 77,44 ± 1,37 | 74,74 ± 3,37 | 92,60 ± 0,59 | 93,89 ± 0,76 |
| VTB | 425 | 99,25 ± 0,10 | 99,95 ± 0,11 | 99,72 ± 0,10 | 100,00 ± 0,00 |

## Comparação descritiva e limites

Comparação com 13 classes: [transfer_learning_poly_disc_consolidated_report.md](transfer_learning_poly_disc_consolidated_report.md). Remover uma classe muda o conjunto de teste, o número de saídas, os pesos/estatísticas ajustados no treino e pode mudar a configuração selecionada. A diferença é descritiva; não representa melhoria no problema original de 13 classes. O cenário sem TS9 é uma ablação exploratória motivada por resultados já conhecidos.

**O teste já foi consultado anteriormente.** A escolha nesta consolidação usa somente validação, mas não equivale a um teste totalmente cego.

O DP é entre seeds de treinamento na mesma partição; não mede generalização entre novas fontes. Os cenários contínuo/discreto podem compartilhar fontes dentro da mesma família mono/poly. Aqui cada cenário tem sua própria cabeça e avaliação, sem treinar entre cenários. A comparação com o baseline histórico é descritiva: ele usa divisão por arquivo e outro frontend.

## Integridade e reuso

As dez cabeças foram recarregadas e suas predições reproduziram exatamente os arquivos salvos (15130 por cabeça). Hashes dos heads/scalers/caches, ausência de métricas de teste na seleção e ordem dos registros de seleção/treino/teste foram conferidos.

O resumo portátil está em [transfer_learning_no_ts9_poly_disc_consolidated_summary.json](transfer_learning_no_ts9_poly_disc_consolidated_summary.json). Dados, embeddings, checkpoints, predições por arquivo e gráficos permanecem locais e ignorados pelo Git.

Para consultar sem treinar: `notebooks/transfer_learning_no_ts9.ipynb`. Para inferência com WAVs mono já reamostrados/cortados para 32 kHz/64.000 amostras, use `src.models.consolidated_probe.load_consolidated_probe`, passando o diretório da execução e dispositivo CUDA. Carregar a cabeça junto ao scaler salvo e consultar `probe.classes` para interpretar as saídas.

O carregador de inferência também foi conferido em 16 WAVs selecionados por modelo: logits finitos e idênticos aos obtidos com o cache, com diferença máxima zero. Essa conferência valida o reuso do pipeline, não uma nova estimativa de desempenho.
