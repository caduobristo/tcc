# Transfer learning consolidado: PaSST e HTS-AT

Execução em 01/10/2026, com encoders congelados. Fine-tuning não foi realizado.

Sessão local: `results/transfer_learning/mono_disc/consolidation_20261001_093003_ec7d69`. Duração do protocolo: **11.42 minutos**, reutilizando os embeddings já extraídos.

123.552 áudios Mono Discrete/13 efeitos; 624 fontes; treino/validação/teste 88.902/9.702/24.948 por fonte (seed 42). WAVs a 32 kHz com frontend nativo, não os mel32 Kaldi.

## Resultado no teste

| Modelo | Acurácia (%) | F1 macro (%) | Precisão macro (%) | Recall macro (%) |
| --- | ---: | ---: | ---: | ---: |
| PaSST | 82,83 ± 0,21 | 82,22 ± 0,18 | 82,13 ± 0,17 | 84,11 ± 0,14 |
| HTS-AT | 90,86 ± 0,09 | 90,49 ± 0,09 | 90,79 ± 0,18 | 90,26 ± 0,12 |

Média ± desvio padrão amostral de cinco seeds novas (101, 202, 303, 404 e 505). Cada seed é avaliada nos mesmos 24.948 áudios; não são 124.740 exemplos independentes.

## Seleção e execuções

Oito configurações por modelo foram comparadas usando F1 macro de validação. As duas melhores receberam mais duas seeds; a configuração foi escolhida pela média nas seeds 42, 7 e 21. A seleção foi registrada antes dos dez treinamentos finais, e todas as cabeças foram finalizadas antes de avaliar o teste.

AdamW: LR inicial 0,001, weight decay 0,01; teto de 200 épocas; scheduler e parada antecipada na validação. Consulte [o protocolo](../docs/transfer_learning_consolidation.md) e [a configuração](../configs/linear_probe/consolidation_mono_disc.json).

### PaSST

Configuração escolhida: `train_standardize_train_balanced_b128`. F1 macro de validação nas três seeds: **81,66 ± 0,03%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_train_balanced_b128 | 81,68 | 67 | 78 |
| train_standardize_none_b128 | 81,41 | 42 | 57 |
| train_standardize_train_balanced_b1024 | 80,94 | 59 | 74 |
| none_train_balanced_b1024 | 80,28 | 69 | 84 |
| none_train_balanced_b128 | 80,20 | 20 | 35 |
| train_standardize_none_b1024 | 80,14 | 56 | 61 |
| none_none_b128 | 79,41 | 25 | 26 |
| none_none_b1024 | 79,25 | 68 | 71 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_train_balanced_b128 | 81,66 ± 0,03 |
| train_standardize_none_b128 | 81,51 ± 0,12 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 49 / 64 | 44480 | 31.74 | 82,97 | 82,36 |
| 202 | 35 / 50 | 34750 | 24.09 | 82,68 | 82,06 |
| 303 | 41 / 55 | 38225 | 25.86 | 82,55 | 82,01 |
| 404 | 41 / 48 | 33360 | 23.02 | 83,06 | 82,39 |
| 505 | 46 / 47 | 32665 | 22.48 | 82,89 | 82,30 |

Checkpoint representativo escolhido pela **validação**, seed 202: `results/transfer_learning/mono_disc/consolidation_20261001_093003_ec7d69/passt/final/seed_202`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

### HTS-AT

Configuração escolhida: `train_standardize_none_b128`. F1 macro de validação nas três seeds: **91,41 ± 0,06%**.

| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |
| --- | ---: | ---: | ---: |
| train_standardize_none_b128 | 91,44 | 63 | 63 |
| train_standardize_none_b1024 | 90,90 | 59 | 61 |
| train_standardize_train_balanced_b128 | 90,56 | 33 | 35 |
| train_standardize_train_balanced_b1024 | 90,17 | 46 | 48 |
| none_none_b128 | 88,28 | 70 | 85 |
| none_train_balanced_b128 | 87,33 | 73 | 79 |
| none_none_b1024 | 85,72 | 92 | 95 |
| none_train_balanced_b1024 | 85,19 | 84 | 89 |

| Finalista | F1 macro validação médio ± DP (%) |
| --- | ---: |
| train_standardize_none_b128 | 91,41 ± 0,06 |
| train_standardize_none_b1024 | 90,99 ± 0,13 |

| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 47 / 58 | 40310 | 27.61 | 90,92 | 90,54 |
| 202 | 36 / 43 | 29885 | 19.86 | 90,82 | 90,44 |
| 303 | 44 / 59 | 41005 | 27.88 | 90,97 | 90,55 |
| 404 | 38 / 40 | 27800 | 18.09 | 90,83 | 90,56 |
| 505 | 41 / 56 | 38920 | 27.19 | 90,74 | 90,35 |

Checkpoint representativo escolhido pela **validação**, seed 303: `results/transfer_learning/mono_disc/consolidation_20261001_093003_ec7d69/htsat/final/seed_303`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.

Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.

## Desempenho por classe

| Efeito | Suporte por seed | PaSST F1 (%) | PaSST recall (%) | HTS-AT F1 (%) | HTS-AT recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 808 | 2520 | 51,89 ± 1,55 | 49,59 ± 3,00 | 73,87 ± 0,60 | 75,25 ± 1,57 |
| BD2 | 2520 | 97,91 ± 0,02 | 97,20 ± 0,05 | 98,82 ± 0,02 | 98,40 ± 0,08 |
| BMF | 2520 | 99,03 ± 0,04 | 99,23 ± 0,05 | 99,85 ± 0,02 | 99,85 ± 0,02 |
| DPL | 504 | 89,34 ± 0,50 | 94,60 ± 0,55 | 92,98 ± 0,29 | 91,90 ± 0,72 |
| DS1 | 2520 | 95,93 ± 0,12 | 97,24 ± 0,10 | 97,23 ± 0,07 | 97,83 ± 0,15 |
| FFC | 630 | 94,15 ± 0,54 | 97,87 ± 0,18 | 99,34 ± 0,11 | 99,84 ± 0,00 |
| MGS | 2520 | 93,56 ± 0,15 | 92,06 ± 0,60 | 91,72 ± 0,26 | 92,50 ± 0,85 |
| OD1 | 504 | 34,93 ± 0,34 | 62,78 ± 1,35 | 67,20 ± 0,84 | 63,21 ± 1,53 |
| RAT | 2520 | 92,45 ± 0,30 | 92,05 ± 0,31 | 97,51 ± 0,10 | 97,70 ± 0,19 |
| RBM | 2520 | 99,18 ± 0,05 | 99,34 ± 0,11 | 99,83 ± 0,02 | 99,92 ± 0,00 |
| SD1 | 2520 | 67,09 ± 0,97 | 56,69 ± 1,32 | 84,12 ± 0,29 | 85,11 ± 0,26 |
| TS9 | 2520 | 54,30 ± 1,46 | 55,98 ± 3,18 | 74,29 ± 0,18 | 71,94 ± 0,96 |
| VTB | 630 | 99,14 ± 0,13 | 98,83 ± 0,09 | 99,59 ± 0,07 | 99,90 ± 0,09 |

## Comparação descritiva e limites

- passt: F1 macro inicial 75,52%; agora 82,22%, diferença de 6.71 pontos percentuais. Recall OD1 inicial 0,00%; agora 62,78%.
- htsat: F1 macro inicial 73,27%; agora 90,49%, diferença de 17.22 pontos percentuais. Recall OD1 inicial 14,09%; agora 63,21%.

A primeira rodada tinha uma seed, dez épocas, seleção por acurácia e não demonstrou convergência. A consolidação muda duração, critério e possivelmente scaler/batch/pesos; não atribuir todo ganho a um único ajuste.

**O teste já foi consultado anteriormente.** A escolha nesta consolidação usa somente validação, mas não equivale a um teste totalmente cego. O DP é entre seeds de treinamento na mesma partição; não mede generalização entre novas fontes. Esta etapa não reproduz exatamente o AST e não estabelece superioridade geral de uma arquitetura.

## Integridade e reuso

As dez cabeças foram recarregadas e suas predições reproduziram exatamente os arquivos salvos (24948 por cabeça). Hashes dos heads/scalers/caches, ausência de métricas de teste na seleção e ordem dos registros de seleção/treino/teste foram conferidos.

O resumo portátil está em [transfer_learning_consolidated_summary.json](transfer_learning_consolidated_summary.json). Dados, embeddings, checkpoints, predições por arquivo e gráficos permanecem locais e ignorados pelo Git.

Para consultar sem treinar: `notebooks/consolidate_transfer_learning.ipynb`. Para inferência com WAVs mono já reamostrados/cortados para 32 kHz/64.000 amostras, use `src.models.consolidated_probe.load_consolidated_probe`, passando o diretório da execução e dispositivo CUDA. A precisão BF16 do encoder é preservada; saída são logits das 13 classes na ordem de `src.data.transfer.EFFECTS`.

O carregador de inferência também foi conferido em 16 WAVs originais por modelo (primeiro batch do manifest): logits finitos e idênticos aos obtidos com o cache, com diferença máxima zero. Essa conferência valida o reuso do pipeline, não uma nova estimativa de desempenho.
