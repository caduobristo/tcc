# Inventário e integridade dos dados — 30/09/2026

**Atualização após a auditoria inicial:** nesta mesma data, foram restaurados os WAVs oficiais de Mono Discrete para preparar os novos experimentos PaSST/HTS-AT. O inventário histórico abaixo mantém as lacunas encontradas inicialmente; a nova disponibilidade e a validação estão detalhadas na seção seguinte.

## Restauração para PaSST e HTS-AT

Foram baixados os 13 volumes oficiais do [Mono Discrete, Zenodo 4298000](https://zenodo.org/records/4298000), somando **26.770.255.392 bytes** sem o README. Todos passaram no MD5 publicado. O 7-Zip terminou a extração e a verificação de CRC sem erros: **164.736 WAVs e 14 CSVs**, 29.075.403.768 bytes no total. Os volumes ficam em `data/archives/guitar_fx_dist/official/Mono_Discrete/`; os áudios, em `data/raw/guitar_fx_dist/Mono_Discrete/Audio/`. As features originais do baseline também estão nesse arquivo oficial, mas não foram extraídas para o caminho do baseline.

O índice compartilhado dos novos modelos seleciona **123.552 WAVs**, excluindo MT2 e ambas as pastas NoFX. Todos os arquivos selecionados foram decodificados e tiveram formato, duração, valores finitos e SHA-256 conferidos: mono, 44.100 Hz, 88.201 amostras. O loader reamostra para 32 kHz e mantém exatamente dois segundos. Não há WAVs selecionados byte a byte duplicados.

As 624 gravações originais foram divididas com seed 42, preservando no mesmo conjunto todos os efeitos/parâmetros de cada fonte:

| Partição | Fontes | WAVs |
| --- | ---: | ---: |
| Treino | 449 | 88.902 |
| Validação | 49 | 9.702 |
| Teste | 126 | 24.948 |

Há 13 classes em cada partição e nenhuma fonte ou conteúdo WAV idêntico atravessa conjuntos. Índice e relatório ficam em `data/manifests/transfer_learning/mono_disc/samples.csv` e `samples.json`; a restauração está registrada em `data/audits/transfer_learning/raw_restore.json`. Esses artefatos são locais e ignorados pelo Git. Os frontends nativos dos checkpoints substituem, nestes dois novos experimentos, os mel32 Kaldi recebidos da dupla. Não foram produzidos embeddings completos nem iniciados treinamentos. Veja o [guia de preparação](transfer_learning_setup.md).

Os WAVs de Mono Continuous, Poly Continuous e Poly Discrete, além de GEC-GIM, permanecem ausentes. Os ZIPs mel16/mel32 existentes continuam intactos.

## O que está disponível

Foram encontrados nove ZIPs na antiga pasta `dados/`: oito variantes de espectrogramas do GUITAR-FX-DIST e um pacote de CSVs. Eles foram movidos, sem recompactação, para `data/archives/guitar_fx_dist/`. A pasta antiga é uma junção de compatibilidade. O pacote do Drive em Downloads foi copiado para `data/archives/import_bundles/`; contém outra cópia byte a byte de `csvs.zip`. O original em Downloads foi preservado.

Os nove ZIPs somam **113.486.868.204 bytes (113,49 GB; 105,69 GiB)**. Cada variante cobre os quatro cenários. Foram conferidos todos os **1.115.408 NPYs**, e não apenas uma amostra.

| Cenário | NPYs por variante, incluindo NoFX | Efeitos, com MT2 | Efeitos, sem MT2 e sem NoFX | NoFX original + pré-processado |
| --- | ---: | ---: | ---: | ---: |
| Mono Continuous | 141.248 | 140.000 | 130.000 | 624 + 624 |
| Mono Discrete | 164.736 | 163.488 | 123.552 | 624 + 624 |
| Poly Continuous | 140.840 | 140.000 | 130.000 | 420 + 420 |
| Poly Discrete | 110.880 | 110.040 | 83.160 | 420 + 420 |

Classes presentes: `808`, `BD2`, `BMF`, `DPL`, `DS1`, `FFC`, `MGS`, `MT2`, `OD1`, `RAT`, `RBM`, `SD1`, `TS9` e `VTB`. Também existem `_NoFX_mono`, `_NoFX_mono_preprocessed`, `_NoFX_poly` e `_NoFX_poly_preprocessed` nos respectivos cenários. Esses números descrevem os ZIPs atuais, não uma contagem reexecutada do dataset histórico do baseline.

Ambas as variantes usam **float32, shape (198, 128)**. Para cada cenário, os conjuntos de nomes coincidem entre mel16 e mel32, mas **nenhuma amostra com o mesmo nome tem conteúdo byte a byte idêntico nas duas variantes**. Não são cópias redundantes que possam ser apagadas. O usuário confirmou que vieram do Drive da dupla e que 16/32 representam a taxa de amostragem: **16 kHz e 32 kHz**. O notebook de geração na branch `feature/transfer-learning`, commit `02563bb`, confirma as duas configurações. A receita inclui mono, reamostragem, waveform multiplicada por 32768 e Kaldi Fbank com 128 bandas, Hanning, passo de 10 ms e dither 0. Versões exatas do ambiente gerador e hashes fornecidos pelo produtor ainda não estão disponíveis.

## Uso nos experimentos e lacunas

| Experimento / fonte | Dados usados segundo o código e os registros | Situação local atual |
| --- | --- | --- |
| Baseline consolidado FxNet | GUITAR-FX-DIST, quatro cenários, features `mel_22050_1024_512`, 13 classes, MT2 excluído | Resultados e quatro checkpoints presentes; features originais ausentes |
| Reprodução FxNet/SetNetCond/MultiNet | GUITAR-FX (nome local do GUITAR-FX-DIST), features originais; discretos concluídos com TS9 e MT2 excluídos | Código, checkpoints e saídas presentes; dados de entrada ausentes |
| AudioMAE, embeddings | WAVs GUITAR-FX Mono Continuous; 250 por classe, 14 efeitos + NoFX; 3.750 embeddings de 768 dimensões | Embeddings, metadados e checkpoint presentes; WAVs ausentes |
| AudioMAE, cabeça AudioSet | WAVs dos quatro cenários GUITAR-FX; inferência de categorias AudioSet | Predições e relatórios presentes; WAVs ausentes; isso não é treinamento em AudioSet local |
| PaSST exploratório | GEC-GIM segundo README/datasheet; issue chama GEC-PIM; nomes `ag_G_<efeito>...wav` | 109 embeddings de 768 dimensões presentes; WAVs ausentes; Chorus tem 9, demais 10 classes têm 10 |
| IDMT-SMT-GUITAR | Estudado para geração sintética | Nenhum pacote ou WAV identificado nas localizações inspecionadas |
| IDMT-SMT-AUDIO-EFFECTS | Referência e fonte dos sons limpos do GUITAR-FX-DIST | Dataset independente não encontrado; derivação no GUITAR-FX não significa que a coleção IDMT esteja baixada |
| GEPE-GIM | Documentado para estimação de parâmetros | Dataset não encontrado; nenhuma execução local confirmada |
| AudioSet, avaliação do artigo AudioMAE | Notebook template de avaliação com áudios/metadados AudioSet | Dataset não encontrado; checkpoints pré-treinados não substituem o dataset |
| AST na branch de transfer learning | GUITAR-FX Mono Discrete/mel_16; backbone ImageNet + AudioSet e cabeça para 13 classes | Notebook registra carregamento de 123.552 amostras e inicialização do modelo; não há épocas, métricas finais ou checkpoint AST versionado |
| ZIPs mel16/mel32 atuais | Log-Mel a 16/32 kHz recebidos da dupla; notebook gerador identificado | mel_16 é a variante selecionada pelo AST; identidade byte a byte com os arquivos usados pelo colega não é comprovada sem hashes de origem |

As evidências principais são `notebooks/train_baseline.ipynb`, `src/data/dataset.py`, `src/models/fxnet.py`, `results/baseline_results_summary.json`, os relatórios por cenário, os JSONs/CSVs dos experimentos AudioMAE e FxNet e `experiments/passt/results_10samples/embeddings.npz`.

O notebook salvo do baseline aponta historicamente para `../../datasets/GUITAR-FX-DIST/<cenario>`. Seu último cenário executado foi Poly Continuous, com 130.000 amostras, 13 classes, 93.600 no treino, 10.400 na validação e 26.000 no teste. O código e os resultados confirmam também execuções dos outros três cenários. O split implementado é **72% treino / 8% validação / 20% teste**, com seed 42; não é 80/10/10. A escolha aleatória é por arquivo, sem agrupamento explícito por fonte sonora.

A FxNet produz 6.264 entradas para a camada densa com matrizes de 87 × 128. A entrada de 198 × 128 gera 16.008 entradas e falha nessa camada. Isso foi confirmado com um forward em CPU, sem treinamento. Portanto, os ZIPs atuais não reproduzem diretamente o baseline. A configuração do notebook agora usa `data/processed/guitar_fx_dist/baseline/`, recusa ausência e shape incompatível, e o notebook anterior à alteração foi preservado em `data/manifests/backups/`.

## Integridade e inconsistências

- Oito ZIPs de features: leitura completa, CRC de cada membro, SHA-256 do arquivo e de cada NPY; zero membros corrompidos, zero shapes/dtypes inesperados e zero arrays com NaN/Inf.
- 56 CSVs: CRC, decodificação, cabeçalhos, rótulos e parâmetros numéricos finitos entre 0 e 1 verificados. Nenhum valor inválido ou conflito entre linhas do mesmo filename foi encontrado.
- Os CSVs originais têm **11.366 repetições exatas de linhas**. Foram removidas apenas nas cópias derivadas; as fontes permanecem intactas.
- **31 nomes únicos dos CSVs não têm NPY correspondente em nenhuma variante**: Mono Continuous/DPL (1), Poly Continuous/808 (1), FFC (27), RBM (1), VTB (1). Isso não prova que a coleção de NPYs esteja incompleta: pode ser metadado excedente. Não foram inventadas ou regeneradas amostras para preencher a diferença.
- Todos os NPYs de efeitos têm metadados. Os CSVs derivados em `metadata/guitar_fx_dist/validated/` contêm **553.528 linhas**, uma por arquivo de efeito disponível; as pastas NoFX não vêm com CSV de parâmetros.
- Poly Continuous tem **12 pares de arrays idênticos por variante**, todos em MGS, com nomes/parâmetros diferentes. Foram preservados e registrados. Precisam ser considerados na separação treino/teste para evitar cópias idênticas nos dois conjuntos.
- Os manifests permitem identificar cada arquivo, seu CRC e seu SHA-256. A lista nominal dos 31 casos e dos pares repetidos está em `data/audits/metadata_consistency.json`.
- Foram inventariados adicionalmente **153 artefatos** de execução: SHA-256, parse dos JSONs, leitura/shape/tamanho/valores finitos dos NPYs numéricos, CRC dos NPZs e dos 14 checkpoints no formato ZIP do PyTorch. Não foi feita nova avaliação dos modelos; o campo de caminhos de um NPZ é object/pickle e teve somente seu cabeçalho e CRC verificados, sem desserialização. Nenhum erro estrutural foi encontrado nessa auditoria.

Os relatórios completos estão em `data/audits/integrity_report.json`, `metadata_consistency.json` e `artifact_inventory.json`, todos locais. **Integridade binária boa não significa metadados originais consistentes nem equivalência com os dados do baseline.** Não existem checksums fornecidos pelo produtor para comparar os ZIPs do Drive com uma referência externa.

## Organização e próximos dados necessários

Os datasets de entrada têm um destino central; checkpoints, embeddings, predições, gráficos e métricas continuam junto dos experimentos para preservar a relação com suas execuções. A regra Git que ignorava genericamente qualquer pasta `dataset/` também ocultava o módulo Python da reprodução FxNet. Ela foi restringida para manter os dados fora do Git e permitir versionar esse código.

Para repetir o baseline: restaurar as features originais dos quatro cenários com metadados correspondentes, conferir hashes/origem e preservar a taxonomia de 13 classes. Para reextrair embeddings ou ajustar o encoder AudioMAE: restaurar os WAVs dos cenários GUITAR-FX-DIST; as features atuais não permitem recuperar o áudio original. Os embeddings AudioMAE já salvos, com seus rótulos e filenames, permitem estudar classificadores sobre vetores sem restaurar os WAVs. Para repetir a extração PaSST: restaurar os WAVs GEC-GIM correspondentes aos filenames dos embeddings e esclarecer o nome GEC-PIM no registro. Para usar mel16/mel32 em novos modelos: partir da receita na branch de transfer learning, registrar as versões do ambiente e conferir especialmente escala, normalização e expectativas do checkpoint. O AST presente nessa branch seleciona 16 kHz, não 32 kHz.

MFCCs, novos dados sintéticos e coleções completas IDMT/GEPE-GIM/AudioSet não foram identificados como datasets de entrada prontos. Isso é disponibilidade local, não uma exigência de baixar todos antes do próximo experimento.

## Escopo e fontes

A busca percorreu o checkout real de `C:\TCC` sem contar junções duas vezes ou confundir ambientes/bibliotecas com dados, e inspecionou candidatos em Downloads, Desktop, Documents, OneDrive, Projetos e UTFPR. Não é uma busca em todo o disco, outros usuários, unidades desconectadas ou na nuvem. O projeto TCC do ChatGPT não trouxe conversas pertinentes na listagem disponível; os resultados históricos aqui são sustentados pelos arquivos e registros do repositório, sem presumir contexto adicional carregado.

Fontes primárias: [GUITAR-FX-DIST Mono Continuous](https://zenodo.org/records/4296040), [Mono Discrete](https://zenodo.org/records/4298000), [Poly Continuous](https://zenodo.org/records/4298017), [Poly Discrete](https://zenodo.org/records/4298025), [código e links dos datasets GEC/GEPE-GIM](https://github.com/kevingerkens/gitfx), [IDMT-SMT-GUITAR](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html), [IDMT-SMT-AUDIO-EFFECTS](https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html).

Registros do TCC: [datasheets, issue 2](https://github.com/caduobristo/tcc/issues/2), [AudioMAE](https://github.com/caduobristo/tcc/issues/1#issuecomment-4376134470), [PaSST](https://github.com/caduobristo/tcc/issues/1#issuecomment-4546846885). As fontes externas identificam as coleções e os relatos; não certificam os arquivos processados recebidos pelo Drive.
