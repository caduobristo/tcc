# Matrizes de confusão: PaSST e HTS-AT

Figuras científicas para a [issue 5](https://github.com/caduobristo/tcc/issues/5), geradas a partir das predições de teste já concluídas. Não houve extração, treinamento ou fine-tuning.

Cada célula contém a média dos percentuais nas seeds 101, 202, 303, 404 e 505. As linhas representam as classes verdadeiras; as colunas, as classes previstas. Cada linha soma 100% antes do arredondamento; a diagonal corresponde ao recall médio da classe.

As cinco seeds usam os mesmos WAVs de teste. Os suportes são registrados por seed: não representam cinco vezes mais exemplos independentes. Acurácia, F1, precisão e recall macro são calculados separadamente por seed e depois agregados; o F1 médio não é calculado a partir da matriz média.

As matrizes inteiras foram reconstruídas com os arquivos de predições, comparadas com os registros originais e usadas para conferir todas as métricas agregadas. Hashes de heads, scalers, predições e resumos foram conferidos. `summary.json` registra percentuais, suportes por seed e fontes, sem publicar predições por WAV.

## Figuras

| Cenário | PaSST, 13 classes | HTS-AT, 13 classes | PaSST, sem TS9 | HTS-AT, sem TS9 |
| --- | --- | --- | --- | --- |
| Mono Discrete | [PASST](passt_mono_disc_13_classes.png) | [HTSAT](htsat_mono_disc_13_classes.png) | [PASST](passt_mono_disc_no_ts9.png) | [HTSAT](htsat_mono_disc_no_ts9.png) |
| Mono Continuous | [PASST](passt_mono_cont_13_classes.png) | [HTSAT](htsat_mono_cont_13_classes.png) | [PASST](passt_mono_cont_no_ts9.png) | [HTSAT](htsat_mono_cont_no_ts9.png) |
| Poly Discrete | [PASST](passt_poly_disc_13_classes.png) | [HTSAT](htsat_poly_disc_13_classes.png) | [PASST](passt_poly_disc_no_ts9.png) | [HTSAT](htsat_poly_disc_no_ts9.png) |
| Poly Continuous | [PASST](passt_poly_cont_13_classes.png) | [HTSAT](htsat_poly_cont_13_classes.png) | [PASST](passt_poly_cont_no_ts9.png) | [HTSAT](htsat_poly_cont_no_ts9.png) |

## Limites e reprodução

A ablação sem TS9 remove essa classe de treino, validação e teste, reutiliza os encoders/embeddings e refaz a seleção e o treinamento das cabeças. As partições dos demais WAVs são mantidas. Os resultados de 12 classes não substituem os de 13 classes; a análise é exploratória, com teste previamente consultado.

Para regenerar as figuras na raiz, com os resultados locais completos: `python scripts/render_transfer_confusion_matrices.py`. Predições, caches, pesos, logs e documentos de contexto continuam locais. Apenas estas figuras selecionadas e seus metadados agregados são publicados.

Relatórios: [13 classes](../transfer_learning_all_scenarios_report.md), [sem TS9](../transfer_learning_no_ts9_report.md).
