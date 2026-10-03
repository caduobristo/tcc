# Transformers de áudio para classificação de efeitos de guitarra

ICSXG0-S71 TCC | Semestre 1 de 2026

Trabalho de Conclusão de Curso (TCC) em andamento para o curso de graduação em Engenharia da Computação na Universidade Tecnologia Federal do Paraná - UTFPR - Campus Curitiba.

Alunos: Carlos Eduardo Obristo e Rodrigo Moliani Braga

Orientador: Gustavo Benvenutti Borba

<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/a262350d-b957-4065-a927-9e0d3f97e732" />

---

## 1. Introdução e Visão Geral

Este repositório documenta o desenvolvimento de um sistema para reconhecimento automático de efeitos de guitarra a partir de sinais de áudio, com ênfase na aplicação de técnicas modernas de aprendizado de máquina.

O projeto parte da constatação de que efeitos como *overdrive*, *distortion* e *fuzz* desempenham papel fundamental na definição do timbre da guitarra elétrica. Apesar de sua relevância prática em performance musical e produção sonora, observa-se uma lacuna na literatura no que se refere à identificação automática desses efeitos por meio de métodos computacionais.

Diante desse contexto, o objetivo do trabalho é investigar diferentes abordagens para o problema, com foco na análise comparativa entre representações de áudio e arquiteturas de redes neurais ao longo do desenvolvimento do TCC.

---

## 2. Objetivos

### 2.1 Objetivo Geral

Desenvolver um pipeline computacional capaz de classificar efeitos de guitarra a partir de sinais de áudio.

### 2.2 Objetivos Específicos

- Reproduzir experimentos relevantes da literatura
- Comparar diferentes representações de áudio (Mel-spectrogram, MFCC, entre outras)
- Avaliar arquiteturas de redes neurais para tarefas de classificação
- Explorar cenários mais realistas, como sinais contendo múltiplos instrumentos
- Investigar relações de similaridade tímbrica aprendidas pelos modelos

---

## 3. Abordagem Inicial: Redes Convolucionais

A abordagem inicial adotada neste projeto segue a linha tradicional da literatura em classificação de áudio, baseada no uso de Redes Neurais Convolucionais (CNNs).

Nesse contexto, o sinal de áudio é previamente transformado em representações bidimensionais, como espectrogramas, permitindo que o problema seja tratado de forma análoga a tarefas de visão computacional.

Embora eficaz, essa abordagem apresenta limitações relevantes:

- Predominância na captura de padrões locais
- Dificuldade na modelagem de dependências de longo alcance no domínio temporal
- Necessidade de ajustes arquiteturais específicos para diferentes tarefas

Essas limitações motivaram a investigação de arquiteturas mais recentes e expressivas.

---

## 4. Evolução da Abordagem: Transformers para Áudio

A partir da revisão bibliográfica, o projeto evoluiu para a incorporação de modelos baseados em Transformers, que vêm apresentando resultados melhores em tarefas de processamento de áudio.

A principal mudança conceitual consiste na forma de representação do sinal: o espectrograma deixa de ser tratado exclusivamente como uma imagem e passa a ser interpretado como uma sequência de unidades (patches), possibilitando a aplicação de mecanismos de self-attention.

Essa abordagem permite uma modelagem mais eficiente de relações globais no sinal, superando limitações inerentes às CNNs.

---

## 5. Aplicação de Transformers em Áudio

A aplicação de Transformers em tarefas de áudio segue um pipeline estruturado, descrito a seguir:

- Conversão do sinal de áudio em espectrograma (por exemplo, log-Mel)
- Segmentação do espectrograma em patches (tipicamente 16×16)
- Projeção de cada patch em um vetor de características (embedding)
- Adição de codificação posicional (positional encoding)
- Processamento da sequência por um Transformer Encoder
- Utilização de um token especial (CLS) para a tarefa de classificação

Esse processo possibilita:

- Captura de dependências globais nos domínios de tempo e frequência
- Modelagem de interações complexas entre diferentes regiões do espectrograma

---

## 6. Arquiteturas Investigadas

Durante o desenvolvimento do projeto, diferentes variações de Transformers aplicados a áudio foram analisadas:

### [6.1 AST (Audio Spectrogram Transformer)](https://arxiv.org/abs/2104.01778)
- Arquitetura baseada exclusivamente em mecanismos de atenção  
- Derivada do [Vision Transformer](https://arxiv.org/abs/2010.11929)  
- Alta capacidade de modelagem de contexto global  

### [6.2 PaSST (Patchout Spectrogram Transformer)](https://arxiv.org/abs/2110.05069)
- Extensão do AST com foco em eficiência computacional  
- Introdução do método Patchout para redução da sequência de entrada  
- Melhoria na generalização e redução de custo computacional  

### [6.3 HTS-AT (Hierarchical Token-Semantic Audio Transformer)](https://arxiv.org/abs/2202.00874)
- Arquitetura hierárquica com redução progressiva da dimensionalidade  
- Uso de atenção local (window attention)  
- Suporte à detecção temporal de eventos  

### [6.4 AudioMAE (Masked Autoencoders)](https://arxiv.org/abs/2207.06405)
- Abordagem baseada em aprendizado auto-supervisionado  
- Reconstrução de patches mascarados do espectrograma  
- Redução da dependência de dados rotulados  

---
## 7. Documentação dos Datasets

Os datasets utilizados neste trabalho são documentados seguindo o framework [Datasheets for Datasets](https://arxiv.org/abs/1803.09010), proposto por Gebru et al., com adaptações para o domínio de áudio musical e efeitos de guitarra.

Os datasheets completos estão disponíveis na issue dedicada: [Datasheets for Datasets](https://github.com/caduobristo/tcc/issues/2)

### Datasets analisados

* **IDMT-SMT-GUITAR:**
  Dataset de guitarra limpa com múltiplas técnicas performáticas e estruturas musicais, utilizado como base para geração sintética de novos dados.

* **IDMT-SMT-AUDIO-EFFECTS:**
  Dataset organizado por categorias de efeitos de áudio, contendo gravações processadas e metadados estruturados via XML.

* **GEC-GIM:**
  Dataset para classificação de efeitos de guitarra em sinais mistos contendo múltiplos instrumentos.

* **GEPE-GIM:**
  Dataset voltado à estimação contínua de parâmetros de efeitos de guitarra em mixagens instrumentais.

---

## 8. Estado Atual do Projeto

Até o momento, o projeto encontra-se nas seguintes etapas:

- Revisão bibliográfica consolidada, com foco em arquiteturas modernas para classificação de áudio baseadas em CNNs e Transformers  
- Estudo detalhado de modelos como AST, PaSST, HTS-AT e AudioMAE, incluindo suas estratégias de treinamento e representação  
- Definição do pipeline experimental, contemplando pré-processamento, modelagem e avaliação  
- Reprodução de experimentos da literatura, validando resultados reportados e consolidando o ambiente experimental  
- Implementação de transfer learning com PaSST e HTS-AT congelados e treinamento de cabeças lineares para as 13 classes de efeitos; protocolos e resultados na seção 13

Como próximos passos, destacam-se:

- Realização de **fine-tuning parcial ou total** dos modelos, visando adaptação mais profunda ao domínio do problema  
- Investigação da viabilidade de **treinamento de modelos baseados em Transformers do zero**, considerando disponibilidade de dados e custo computacional  
- Expansão e organização do dataset, incluindo possíveis estratégias de geração de dados sintéticos  

---

## 9. Resultados Preliminares

Foram realizados testes preliminares com arquiteturas baseadas em Transformers para áudio, com foco nos modelos PaSST (Patchout Spectrogram Transformer) e AudioMAE (Masked Autoencoders for Audio).

Os experimentos conduzidos até o momento, incluindo configurações utilizadas, etapas de reprodução e resultados obtidos, foram documentados em uma issue separada do repositório: https://github.com/caduobristo/tcc/issues/1

Esses testes têm como objetivo avaliar a viabilidade das arquiteturas investigadas para tarefas de classificação de efeitos de guitarra e servir como base para os próximos experimentos de fine-tuning e adaptação ao domínio do projeto.

---

## 10. Contribuições Esperadas

Espera-se que o projeto resulte em:

- Um pipeline reprodutível para classificação de efeitos de guitarra
- Análise do impacto de diferentes representações de áudio no desempenho dos modelos
- Comparação entre abordagens baseadas em CNNs e Transformers
- Geração de insights sobre modelagem computacional de timbre musical

---

## 11. Experimentos de reprodução

As tentativas de reprodução e explorações iniciais estão organizadas em [experiments/](experiments/README.md):

- [FxNet e redes de estimação de parâmetros](experiments/fxnet_reproduction/README.md): snapshot do código de Comunità et al., adaptações locais e registros de execuções concluídas e com falhas.
- [AudioMAE](experiments/audiomae/README.md): validação do modelo pré-treinado, inferência AudioSet e exploração de embeddings de guitarra.
- [PaSST](experiments/passt/README.md): extração e análise exploratória de embeddings.

A baseline consolidada possui seu próprio [notebook de treinamento](notebooks/train_baseline.ipynb) e [relatório de resultados](results/baseline/baseline_results_report.md). Ela deve ser distinguida das reproduções históricas acima.

Os experimentos incluem código, documentação, licenças e resultados resumidos. Datasets, ambientes, checkpoints das reproduções e artefatos volumosos devem ser preparados localmente, conforme o README de cada experimento.

## 12. Organização dos dados locais

Dados, metadados, caches e auditorias permanecem locais em `data/` e `results/transfer_learning/`. O protocolo local `docs/transfer_learning_consolidation.md` registra os caminhos e pré-requisitos do cenário Mono Discrete. Os ZIPs mel16/mel32 são variantes processadas distintas das features originais da FxNet e dos frontends nativos dos novos modelos.

## 13. Transfer learning com encoder congelado

PaSST e HTS-AT foram consolidados nos quatro cenários GUITAR-FX-DIST: **Mono Discrete, Mono Continuous, Poly Discrete e Poly Continuous**, em 01–02/10/2026. Cada cenário tem sua própria cabeça linear de 9.997 parâmetros; os encoders pré-treinados permanecem congelados. Os WAVs oficiais são reamostrados para 32 kHz e passam pelo frontend nativo de cada checkpoint. Os ZIPs mel16/mel32 Kaldi da dupla não foram usados nesta etapa.

O [relatório dos quatro cenários](results/passt_htsat_transfer/transfer_learning_all_scenarios_report.md) e o [resumo JSON](results/passt_htsat_transfer/transfer_learning_all_scenarios_summary.json) reúnem acurácia, F1, precisão e recall macro, tempos, integridade e links para as métricas por classe. O [notebook de consulta](notebooks/transfer_learning_all_scenarios.ipynb) apresenta as oito combinações modelo/cenário sem iniciar novo treinamento; `RUN_ADDITIONAL_TRANSFER=False` é o padrão. O [relatório original Mono Discrete](results/passt_htsat_transfer/transfer_learning_consolidated_report.md) e os notebooks [PaSST](notebooks/train_passt.ipynb), [HTS-AT](notebooks/train_htsat.ipynb) e [consolidação Mono Discrete](notebooks/consolidate_transfer_learning.ipynb) foram preservados.

O protocolo usa oito configurações por modelo, confirmação das duas finalistas por validação e cinco seeds novas para a avaliação final: 34 treinamentos de cabeça por cenário, **102 novos nos três cenários adicionais**. Scaler e pesos de classe são calculados somente no treino. AdamW, limite de 200 épocas e parada antecipada por F1 macro de validação; configurações em [configs/linear_probe/](configs/linear_probe/). Os desvios medem variação entre seeds na mesma divisão por fonte. Mono Discrete teve o teste consultado na rodada preliminar; os outros três não tiveram rodada preliminar neste protocolo. Não houve fine-tuning.

### Reprodução dos três cenários adicionais

Em uma cópia nova, prepare Python 3.11, 7-Zip e o ambiente CUDA usado pelo projeto. O script `scripts/setup_transfer_environment.ps1` aceita `-Standalone -BasePython <executável Python 3.11>` para instalar o ambiente sem depender da reprodução FxNet local. Depois, na raiz do projeto:

```powershell
.\.venv-transfer\Scripts\python.exe scripts/prepare_transfer_models.py
.\.venv-transfer\Scripts\python.exe scripts/run_additional_transfer_learning.py --run --prepare-data
```

O executor prepara Mono Continuous, Poly Discrete e Poly Continuous em sequência, valida MD5/CRC dos dados e SHA-256 dos WAVs/caches, preserva as partições por fonte e salva os resultados. Ele reutiliza cenários concluídos. Os volumes novos de download ficam na área temporária e são descartados após extração verificada; arquivos previamente existentes são preservados. A preparação confere espaço disponível e mantém reserva de 12 GiB. Os três cenários novos restauram as 13 classes usadas, sem WAVs MT2/NoFX nem features baseline.

Para inferência, use `src.models.consolidated_probe.load_consolidated_probe`, carregando obrigatoriamente a cabeça e seu scaler. Código, protocolos, notebooks e resumos são versionados; áudios, embeddings, pesos, predições individuais e documentação de contexto permanecem locais.

## 14. Transfer learning sem TS9

Uma ablação adicional repete PaSST e HTS-AT nos quatro cenários com **12 classes**, retirando somente TS9 de treino, validação e teste. Os WAVs restantes conservam exatamente as partições por fonte do experimento original. Os oito caches de embeddings são reutilizados após validação; cada cabeça de 768 → 12 (9.228 parâmetros) é inicializada e treinada novamente, sem modificar o encoder.

O orçamento de busca e as cinco seeds finais são os mesmos do protocolo com 13 classes. Scalers/pesos de classes são ajustados somente no treino restante, e a configuração é selecionada novamente por F1 macro de validação. O [relatório sem TS9](results/passt_htsat_transfer/transfer_learning_no_ts9_report.md) e o [resumo JSON](results/passt_htsat_transfer/transfer_learning_no_ts9_summary.json) incluem os resultados finais, comparação com 13 classes e avaliação das cabeças antigas nos mesmos WAVs de teste das 12 classes restantes. Previsões TS9 das cabeças antigas continuam contando como erro.

Média ± DP nas cinco seeds finais, em porcentagem. Execução concluída em 02/10/2026; verificações finais em 03/10/2026.

| Cenário | PaSST acurácia | PaSST F1 macro | HTS-AT acurácia | HTS-AT F1 macro |
| --- | ---: | ---: | ---: | ---: |
| Mono Discrete | 91,11 ± 0,37 | 88,39 ± 0,28 | 95,32 ± 0,08 | 93,88 ± 0,09 |
| Mono Continuous | 90,67 ± 0,08 | 90,53 ± 0,07 | 94,96 ± 0,07 | 94,91 ± 0,07 |
| Poly Discrete | 93,10 ± 0,35 | 90,31 ± 0,32 | 97,73 ± 0,15 | 96,45 ± 0,23 |
| Poly Continuous | 92,47 ± 0,10 | 92,37 ± 0,08 | 97,69 ± 0,10 | 97,69 ± 0,10 |

O [notebook de consulta](notebooks/transfer_learning_no_ts9.ipynb) mantém `RUN_ABLATION=False`. A ablação é exploratória, motivada por resultados já consultados; remover uma classe altera o problema e não demonstra melhoria na tarefa original com 13 classes. Os resultados originais são preservados como referência principal.

```powershell
# Exibe o plano sem treinar
& .\.venv-transfer\Scripts\python.exe scripts/run_transfer_no_ts9.py
# Executa a ablação; requer os manifests e os oito caches originais
& .\.venv-transfer\Scripts\python.exe scripts/run_transfer_no_ts9.py --run
# Recarrega cabeças antigas e gera a comparação, sem treinamento
& .\.venv-transfer\Scripts\python.exe scripts/report_transfer_no_ts9.py
```

Cabeças, scalers, predições e gráficos ficam localmente em `results/transfer_learning/ablations/no_ts9/`. Para inferência, carregar a cabeça junto ao scaler e consultar `probe.classes` para interpretar a ordem das saídas. Nenhum áudio TS9 é apagado.
