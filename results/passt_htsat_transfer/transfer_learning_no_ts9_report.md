# Transfer learning sem TS9: PaSST e HTS-AT

Ablação exploratória em 02/10/2026. Retiramos somente TS9, mantendo os outros 12 efeitos, as partições por fonte e os encoders congelados. Cada cenário recebeu 34 treinamentos novos, com cinco seeds finais por modelo. Fine-tuning não foi realizado.

## Resultados finais

Média ± desvio padrão amostral entre as seeds 101, 202, 303, 404 e 505. A seleção de configuração/checkpoint usa somente F1 macro de validação.

| Cenário | Modelo | Acurácia sem TS9 (%) | F1 macro sem TS9 (%) | Precisão macro (%) | Recall macro (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| mono disc | PaSST | 91,11 ± 0,37 | 88,39 ± 0,28 | 88,23 ± 0,07 | 90,29 ± 0,12 |
| mono disc | HTS-AT | 95,32 ± 0,08 | 93,88 ± 0,09 | 94,27 ± 0,20 | 93,57 ± 0,07 |
| mono cont | PaSST | 90,67 ± 0,08 | 90,53 ± 0,07 | 90,50 ± 0,10 | 90,61 ± 0,08 |
| mono cont | HTS-AT | 94,96 ± 0,07 | 94,91 ± 0,07 | 94,90 ± 0,07 | 94,93 ± 0,07 |
| poly disc | PaSST | 93,10 ± 0,35 | 90,31 ± 0,32 | 89,89 ± 0,45 | 91,04 ± 0,31 |
| poly disc | HTS-AT | 97,73 ± 0,15 | 96,45 ± 0,23 | 96,58 ± 0,29 | 96,35 ± 0,20 |
| poly cont | PaSST | 92,47 ± 0,10 | 92,37 ± 0,08 | 92,47 ± 0,16 | 92,45 ± 0,10 |
| poly cont | HTS-AT | 97,69 ± 0,10 | 97,69 ± 0,10 | 97,70 ± 0,09 | 97,69 ± 0,09 |

## Comparação com os resultados anteriores

Esta tabela compara problemas com 13 e 12 classes e conjuntos de teste de tamanhos diferentes. A diferença não significa melhoria no problema original.

| Cenário | Modelo | Acurácia 13 classes (%) | Acurácia sem TS9 (%) | Diferença (p.p.) | F1 macro 13 classes (%) | F1 macro sem TS9 (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mono disc | PaSST | 82,83 ± 0,21 | 91,11 ± 0,37 | +8.28 | 82,22 ± 0,18 | 88,39 ± 0,28 |
| mono disc | HTS-AT | 90,86 ± 0,09 | 95,32 ± 0,08 | +4.46 | 90,49 ± 0,09 | 93,88 ± 0,09 |
| mono cont | PaSST | 84,12 ± 0,06 | 90,67 ± 0,08 | +6.55 | 83,95 ± 0,10 | 90,53 ± 0,07 |
| mono cont | HTS-AT | 91,67 ± 0,03 | 94,96 ± 0,07 | +3.29 | 91,59 ± 0,03 | 94,91 ± 0,07 |
| poly disc | PaSST | 85,20 ± 0,12 | 93,10 ± 0,35 | +7.90 | 83,24 ± 0,35 | 90,31 ± 0,32 |
| poly disc | HTS-AT | 92,73 ± 0,13 | 97,73 ± 0,15 | +5.00 | 92,57 ± 0,11 | 96,45 ± 0,23 |
| poly cont | PaSST | 84,91 ± 0,26 | 92,47 ± 0,10 | +7.55 | 84,84 ± 0,20 | 92,37 ± 0,08 |
| poly cont | HTS-AT | 93,88 ± 0,12 | 97,69 ± 0,10 | +3.81 | 93,89 ± 0,12 | 97,69 ± 0,10 |

## Comparação nos mesmos WAVs restantes

Recarregamos também as 40 cabeças antigas e reproduzimos todas as predições originais. A tabela abaixo avalia essas cabeças de 13 saídas somente nos WAVs das 12 classes restantes. Uma predição TS9 continua contando como erro; não removemos logits nem corrigimos previsões antigas. F1 macro é calculado sobre as 12 classes verdadeiras. As cabeças novas usam exatamente esses mesmos WAVs de teste.

| Cenário | Modelo | Acurácia cabeça antiga nos mesmos WAVs (%) | Acurácia cabeça nova (%) | F1 macro antiga nas 12 classes (%) | F1 macro nova (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| mono disc | PaSST | 85,85 ± 0,38 | 91,11 ± 0,37 | 85,86 ± 0,28 | 88,39 ± 0,28 |
| mono disc | HTS-AT | 92,98 ± 0,17 | 95,32 ± 0,08 | 92,87 ± 0,11 | 93,88 ± 0,09 |
| mono cont | PaSST | 86,94 ± 0,22 | 90,67 ± 0,08 | 88,23 ± 0,16 | 90,53 ± 0,07 |
| mono cont | HTS-AT | 93,15 ± 0,12 | 94,96 ± 0,07 | 93,94 ± 0,10 | 94,91 ± 0,07 |
| poly disc | PaSST | 89,86 ± 0,82 | 93,10 ± 0,35 | 87,88 ± 0,57 | 90,31 ± 0,32 |
| poly disc | HTS-AT | 95,45 ± 0,47 | 97,73 ± 0,15 | 95,32 ± 0,20 | 96,45 ± 0,23 |
| poly cont | PaSST | 88,14 ± 0,77 | 92,47 ± 0,10 | 89,47 ± 0,68 | 92,37 ± 0,08 |
| poly cont | HTS-AT | 95,74 ± 0,15 | 97,69 ± 0,10 | 96,59 ± 0,11 | 97,69 ± 0,10 |

## Classe 808, configurações e tempos

- mono disc, PaSST: recall de 808 nos mesmos WAVs, cabeça antiga **49,59%**; cabeça sem TS9 **96,18 ± 0,09%**. Configuração nova `{'normalization': 'train_standardize', 'class_weighting': 'train_balanced', 'batch_size': 128}`. Treinos finais: seed 101: 55 épocas/23.24 s, seed 202: 62 épocas/27.03 s, seed 303: 51 épocas/23.05 s, seed 404: 46 épocas/19.49 s, seed 505: 66 épocas/28.35 s.
- mono disc, HTS-AT: recall de 808 nos mesmos WAVs, cabeça antiga **75,25%**; cabeça sem TS9 **95,71 ± 0,16%**. Configuração nova `{'normalization': 'train_standardize', 'class_weighting': 'none', 'batch_size': 128}`. Treinos finais: seed 101: 54 épocas/23.33 s, seed 202: 54 épocas/25.37 s, seed 303: 58 épocas/26.54 s, seed 404: 60 épocas/28.42 s, seed 505: 56 épocas/24.75 s.
- mono cont, PaSST: recall de 808 nos mesmos WAVs, cabeça antiga **52,91%**; cabeça sem TS9 **95,83 ± 0,04%**. Configuração nova `{'normalization': 'train_standardize', 'class_weighting': 'train_balanced', 'batch_size': 128}`. Treinos finais: seed 101: 52 épocas/24.17 s, seed 202: 52 épocas/25.46 s, seed 303: 43 épocas/20.06 s, seed 404: 54 épocas/26.10 s, seed 505: 46 épocas/21.27 s.
- mono cont, HTS-AT: recall de 808 nos mesmos WAVs, cabeça antiga **73,64%**; cabeça sem TS9 **95,38 ± 0,37%**. Configuração nova `{'normalization': 'train_standardize', 'class_weighting': 'none', 'batch_size': 128}`. Treinos finais: seed 101: 37 épocas/17.14 s, seed 202: 34 épocas/16.68 s, seed 303: 55 épocas/25.72 s, seed 404: 52 épocas/24.95 s, seed 505: 57 épocas/26.61 s.
- poly disc, PaSST: recall de 808 nos mesmos WAVs, cabeça antiga **57,09%**; cabeça sem TS9 **99,20 ± 0,10%**. Configuração nova `{'normalization': 'none', 'class_weighting': 'train_balanced', 'batch_size': 128}`. Treinos finais: seed 101: 66 épocas/19.64 s, seed 202: 50 épocas/15.23 s, seed 303: 51 épocas/14.79 s, seed 404: 57 épocas/16.51 s, seed 505: 53 épocas/16.58 s.
- poly disc, HTS-AT: recall de 808 nos mesmos WAVs, cabeça antiga **79,96%**; cabeça sem TS9 **98,68 ± 0,23%**. Configuração nova `{'normalization': 'train_standardize', 'class_weighting': 'none', 'batch_size': 128}`. Treinos finais: seed 101: 54 épocas/15.57 s, seed 202: 40 épocas/11.51 s, seed 303: 58 épocas/17.98 s, seed 404: 47 épocas/13.94 s, seed 505: 39 épocas/11.26 s.
- poly cont, PaSST: recall de 808 nos mesmos WAVs, cabeça antiga **51,09%**; cabeça sem TS9 **99,81 ± 0,09%**. Configuração nova `{'normalization': 'none', 'class_weighting': 'train_balanced', 'batch_size': 128}`. Treinos finais: seed 101: 43 épocas/19.77 s, seed 202: 30 épocas/14.70 s, seed 303: 43 épocas/20.20 s, seed 404: 42 épocas/19.38 s, seed 505: 52 épocas/24.06 s.
- poly cont, HTS-AT: recall de 808 nos mesmos WAVs, cabeça antiga **75,96%**; cabeça sem TS9 **98,44 ± 0,19%**. Configuração nova `{'normalization': 'train_standardize', 'class_weighting': 'none', 'batch_size': 128}`. Treinos finais: seed 101: 46 épocas/20.57 s, seed 202: 40 épocas/17.96 s, seed 303: 42 épocas/19.77 s, seed 404: 44 épocas/19.58 s, seed 505: 51 épocas/23.58 s.

136 treinamentos: 8 candidatos + 4 confirmações de finalistas + 5 seeds finais por modelo/cenário. Consolidação total: **37.48 minutos**; soma dos tempos registrados dos treinos: **36.69 minutos**. Os tempos excluem extração anterior, edição do código e verificação/reportagem posterior. Não houve nova rodada inicial de dez épocas.

## Integridade e limites

Retiramos TS9 do treino, validação e teste, remapeando VTB do índice 12 para 11. Encoders, WAVs, caches, partições por fonte e arquivos originais foram preservados. Scalers e pesos de classes foram reajustados somente nos dados de treino restantes. A busca foi repetida com o mesmo orçamento e seleção por validação; suas configurações podem diferir das escolhidas para 13 classes.

As 40 cabeças novas reproduziram todas as predições de teste. O caminho WAV → encoder → scaler → cabeça foi conferido em 16 WAVs por modelo/cenário, com logits exatos. Poly Continuous mantém os pares duplicados MGS na mesma partição do experimento original.

O teste já foi consultado e motivou esta ablação. Os resultados são exploratórios, com DP entre treinamentos na mesma partição; não são uma avaliação externa cega nem medida de incerteza entre novas fontes. O experimento com 13 classes continua sendo a referência principal do TCC.

## Consulta e reprodução

- Notebook: `notebooks/transfer_learning_no_ts9.ipynb`, execução desabilitada por padrão.
- Executar: `python scripts/run_transfer_no_ts9.py --run` (requer manifests e os oito caches originais validados).
- Gerar comparação sem treinar: `python scripts/report_transfer_no_ts9.py`.
- Para inferência, carregar a cabeça com `load_consolidated_probe` e interpretar as saídas em `probe.classes`.

Relatórios individuais:

- [mono_disc](transfer_learning_no_ts9_mono_disc_consolidated_report.md).
- [mono_cont](transfer_learning_no_ts9_mono_cont_consolidated_report.md).
- [poly_disc](transfer_learning_no_ts9_poly_disc_consolidated_report.md).
- [poly_cont](transfer_learning_no_ts9_poly_cont_consolidated_report.md).

Resumo estruturado: [transfer_learning_no_ts9_summary.json](transfer_learning_no_ts9_summary.json). Dados, pesos, caches, gráficos, logs e documentos de contexto permanecem locais.
