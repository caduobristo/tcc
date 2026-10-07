# Transfer learning congelado nos quatro cenários GUITAR-FX-DIST

PaSST e HTS-AT usam seus encoders pré-treinados congelados e uma cabeça linear de 9.997 parâmetros. WAVs oficiais de dois segundos, reamostrados para 32 kHz, com o frontend nativo de cada modelo. Os mel16/mel32 Kaldi não alimentam estes experimentos.

## Resultados no teste

Média ± desvio padrão amostral entre cinco sementes (101, 202, 303, 404 e 505), mantendo a mesma partição por fonte em cada cenário.

| Cenário | Modelo | Acurácia (%) | F1 macro (%) | Precisão macro (%) | Recall macro (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Mono Discrete | PaSST | 82,83 ± 0,21 | 82,22 ± 0,18 | 82,13 ± 0,17 | 84,11 ± 0,14 |
| Mono Discrete | HTS-AT | 90,86 ± 0,09 | 90,49 ± 0,09 | 90,79 ± 0,18 | 90,26 ± 0,12 |
| Mono Continuous | PaSST | 84,12 ± 0,06 | 83,95 ± 0,10 | 83,97 ± 0,10 | 84,03 ± 0,08 |
| Mono Continuous | HTS-AT | 91,67 ± 0,03 | 91,59 ± 0,03 | 91,58 ± 0,03 | 91,62 ± 0,02 |
| Poly Discrete | PaSST | 85,20 ± 0,12 | 83,24 ± 0,35 | 85,02 ± 0,33 | 83,20 ± 0,24 |
| Poly Discrete | HTS-AT | 92,73 ± 0,13 | 92,57 ± 0,11 | 93,18 ± 0,14 | 92,23 ± 0,08 |
| Poly Continuous | PaSST | 84,91 ± 0,26 | 84,84 ± 0,20 | 85,05 ± 0,31 | 84,92 ± 0,27 |
| Poly Continuous | HTS-AT | 93,88 ± 0,12 | 93,89 ± 0,12 | 93,91 ± 0,11 | 93,89 ± 0,12 |

## Dados, tempos e seleção

| Cenário | Áudios selecionados | Fontes | Treino / validação / teste | Extração PaSST / HTS-AT (min) | Consolidação (min) |
| --- | ---: | ---: | --- | ---: | ---: |
| Mono Discrete | 123,552 | 624 | 88,902 / 9,702 / 24,948 | 2.38 / 2.30 | 11.42 |
| Mono Continuous | 130,000 | 624 | 93,575 / 10,012 / 26,413 | 3.22 / 3.35 | 12.83 |
| Poly Discrete | 83,160 | 420 | 59,796 / 6,534 / 16,830 | 2.10 / 2.21 | 10.00 |
| Poly Continuous | 130,000 | 420 | 93,402 / 10,064 / 26,534 | 2.56 / 2.54 | 10.64 |

Cada cenário tem 34 treinamentos de cabeça: oito candidatos por modelo, duas finalistas com mais duas seeds e cinco novas seeds finais por modelo. Busca batch 128/1.024 × scaler ausente/padronização no treino × pesos ausentes/balanceados no treino. AdamW LR 0,001, weight decay 0,01, teto de 200 épocas; scheduler e parada antecipada por F1 macro de validação. As dez cabeças finais de cada cenário são concluídas antes do seu teste.

Extração é medida separadamente; consolidação inclui seleção, confirmação, validação e teste sobre vetores. Os tempos não incluem downloads, configuração do ambiente ou verificações posteriores. Hardware: RTX 5080 de 16 GB. Houve preparação de dados e compressão local em paralelo; os tempos representam esta execução e não um benchmark de desempenho sob carga controlada. Consulte os relatórios individuais para épocas e segundos de cada seed.

A extração HTS-AT de Poly Continuous foi interrompida na execução anterior. O arquivo parcial foi preservado e essa extração foi refeita, antes de iniciar o treinamento das cabeças do cenário; o cache PaSST completo foi reutilizado. A tabela registra a extração concluída e não inclui a tentativa interrompida nem o intervalo entre as execuções.

## Relatórios individuais

- [Mono Discrete](transfer_learning_consolidated_report.md): busca, sementes, métricas por classe e artefatos verificados.
- [Mono Continuous](transfer_learning_mono_cont_consolidated_report.md): busca, sementes, métricas por classe e artefatos verificados.
- [Poly Discrete](transfer_learning_poly_disc_consolidated_report.md): busca, sementes, métricas por classe e artefatos verificados.
- [Poly Continuous](transfer_learning_poly_cont_consolidated_report.md): busca, sementes, métricas por classe e artefatos verificados.

## Integridade, fontes e limites

Todos os volumes oficiais usados foram verificados pelos MD5 publicados; os WAVs extraídos passaram por CRC, leitura completa, formato/valores finitos e SHA-256. Nenhuma fonte sonora nem conteúdo WAV idêntico atravessa treino/validação/teste dentro de cada cenário. Recarregamos as 40 cabeças finais/scalers e reproduzimos todas as predições salvas. O caminho WAV → encoder → scaler → cabeça também foi comparado em 16 áudios por modelo/cenário, sem diferença nos logits.

Poly Continuous contém 12 pares de WAVs idênticos na classe MGS: dez pares no treino e dois na validação, nenhum no teste. Foram preservados conforme os metadados oficiais, com cada par na mesma fonte e partição; não há vazamento por esses duplicados. Os outros três cenários não contêm duplicados WAV selecionados. O protocolo de Poly Continuous usa os 130.000 arquivos, sem deduplicação de conteúdo.

Os três downloads novos usaram uma área temporária; após MD5 e extração/CRC completos, os volumes temporários foram descartados para caber no disco. WAVs selecionados, metadados, checksums publicados, índice de membros/CRC, manifests e recibos da preparação foram preservados localmente. Os ZIPs anteriormente baixados pelo usuário e os volumes de Mono Discrete foram preservados.

Fontes oficiais: [Mono Discrete](https://zenodo.org/records/4298000), [Mono Continuous](https://zenodo.org/records/4296040), [Poly Discrete](https://zenodo.org/records/4298025), [Poly Continuous](https://zenodo.org/records/4298017).

Mono Discrete teve teste consultado na rodada preliminar. Nos outros três cenários, este protocolo não fez rodada preliminar nem usou métricas de teste na seleção. Há resultados históricos de outros modelos nestes conjuntos, portanto não são fontes externas inéditas ao projeto. O DP entre seeds mede variação do treinamento, não variação entre partições. Contínuo/discreto podem compartilhar fontes dentro da família mono/poly; os resultados são de treinamentos separados, sem afirmar transferência entre cenários.

O baseline FxNet histórico usa divisão por arquivo e outro frontend. Comparações com ele são descritivas e não equivalem a uma comparação controlada. Não houve fine-tuning nem treino dos encoders.

Código, protocolos, notebooks e resumos estão no Git. Dados, embeddings, checkpoints, logs completos e documentação de contexto permanecem locais. Consulta: `notebooks/transfer_learning_all_scenarios.ipynb`, com execução desabilitada por padrão. Inferência: `src.models.consolidated_probe.load_consolidated_probe`, carregando obrigatoriamente o scaler salvo junto à cabeça.
