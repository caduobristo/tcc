# Experimentos do TCC

Esta pasta reúne reproduções de artigos e explorações que apoiam o pipeline principal do TCC. Cada experimento registra sua origem, seu objetivo e os limites dos resultados.

| Experimento | Objetivo | Estado registrado |
| --- | --- | --- |
| [FxNet / Comunità](fxnet_reproduction/README.md) | Reproduzir classificação e estimação de parâmetros com o código de referência | Execuções parciais, falhas registradas e avaliações de checkpoints |
| [AudioMAE](audiomae/README.md) | Validar o modelo pré-treinado e explorar representações de timbre | Smoke test, inferência AudioSet e embeddings; sem classificador de pedais adaptado |
| [PaSST](passt/README.md) | Explorar embeddings no GEC-GIM | Visualização e vizinhos próximos em uma amostra pequena |

A baseline consolidada do projeto continua em `../notebooks/train_baseline.ipynb`, com implementação em `../src/` e resultados em `../results/`. A reprodução histórica da FxNet e essa baseline são registros distintos.

## Política de arquivos

São versionados códigos, notebooks sem saídas embutidas, licenças, documentação, gráficos e tabelas resumidas. Ambientes Conda, caches, datasets, checkpoints, arrays de embeddings e predições por arquivo permanecem locais e são ignorados pelo Git.

As reproduções importadas preservam os códigos e licenças de origem. Não são submódulos: o clone deste TCC contém os snapshots necessários. Mudanças de compatibilidade e scripts locais estão descritos em cada experimento.

Os relatórios históricos preservam datas, caminhos e falhas das execuções originais. Uma reorganização de arquivos não representa uma nova execução ou validação completa dos artigos.
