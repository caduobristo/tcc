# Transformers de áudio para classificação de efeitos de guitarra

CSX43-S71 TCC | Semestre 2 de 2026

Trabalho de Conclusão de Curso (TCC) em andamento para o curso de graduação em Engenharia da Computação na Universidade Tecnologia Federal do Paraná - UTFPR - Campus Curitiba.

Alunos: Carlos Eduardo Obristo e Rodrigo Moliani Braga

Orientador: Gustavo Benvenutti Borba

<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/a262350d-b957-4065-a927-9e0d3f97e732" />


## 1. Introdução e Visão Geral

Este repositório documenta o desenvolvimento de um sistema para reconhecimento automático de efeitos de guitarra a partir de sinais de áudio, com ênfase na aplicação de técnicas modernas de aprendizado de máquina.

O projeto parte da constatação de que efeitos como *overdrive*, *distortion* e *fuzz* desempenham papel fundamental na definição do timbre da guitarra elétrica. Apesar de sua relevância prática em performance musical e produção sonora, observa-se uma lacuna na literatura no que se refere à identificação automática desses efeitos por meio de métodos computacionais.

Diante desse contexto, o objetivo do trabalho é investigar diferentes abordagens para o problema, com foco na análise comparativa entre representações de áudio e arquiteturas de redes neurais ao longo do desenvolvimento do TCC.


## 2. Objetivos

### 2.1 Objetivo Geral

Desenvolver um pipeline computacional capaz de classificar efeitos de guitarra a partir de sinais de áudio.

### 2.2 Objetivos Específicos

- Reproduzir experimentos relevantes da literatura
- Comparar diferentes representações de áudio (Mel-spectrogram, MFCC, entre outras)
- Avaliar arquiteturas de redes neurais para tarefas de classificação
- Explorar cenários mais realistas, como sinais contendo múltiplos instrumentos
- Investigar relações de similaridade tímbrica aprendidas pelos modelos

## 3. Abordagem Inicial: Redes Convolucionais

A abordagem inicial adotada neste projeto segue a linha tradicional da literatura em classificação de áudio, baseada no uso de Redes Neurais Convolucionais (CNNs).

Nesse contexto, o sinal de áudio é previamente transformado em representações bidimensionais, como espectrogramas, permitindo que o problema seja tratado de forma análoga a tarefas de visão computacional.

Embora eficaz, essa abordagem apresenta limitações relevantes:

- Predominância na captura de padrões locais
- Dificuldade na modelagem de dependências de longo alcance no domínio temporal
- Necessidade de ajustes arquiteturais específicos para diferentes tarefas

Essas limitações motivaram a investigação de arquiteturas mais recentes e expressivas.

## 4. Evolução da Abordagem: Transformers para Áudio

A partir da revisão bibliográfica, o projeto evoluiu para a incorporação de modelos baseados em Transformers, que vêm apresentando resultados melhores em tarefas de processamento de áudio.

A principal mudança conceitual consiste na forma de representação do sinal: o espectrograma deixa de ser tratado exclusivamente como uma imagem e passa a ser interpretado como uma sequência de unidades (patches), possibilitando a aplicação de mecanismos de self-attention.

Essa abordagem permite uma modelagem mais eficiente de relações globais no sinal, superando limitações inerentes às CNNs.

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

## 7. Documentação dos Datasets

Os datasets utilizados neste trabalho são documentados seguindo o framework [Datasheets for Datasets](https://arxiv.org/abs/1803.09010), proposto por Gebru et al., com adaptações para o domínio de áudio musical e efeitos de guitarra. Os datasheets completos estão disponíveis na [Issue #2](https://github.com/caduobristo/tcc/issues/2).

### Datasets analisados

* **IDMT-SMT-GUITAR:** Guitarra limpa com múltiplas técnicas performáticas, utilizado como base para geração sintética de novos dados.
* **IDMT-SMT-AUDIO-EFFECTS:** Gravações processadas organizadas por categorias de efeitos.
* **GEC-GIM:** Sinais mistos contendo múltiplos instrumentos para classificação de efeitos.
* **GEPE-GIM:** Sinais mistos voltados à estimação contínua de parâmetros de efeitos em mixagens.

## 8. Reprodução da Baseline e Experimentos Iniciais

As fases iniciais da pesquisa consistiram na validação do ambiente e na definição de uma linha de base sólida, replicando trabalhos consolidados da literatura. Esses experimentos e testes exploratórios estão organizados em [experiments/](experiments/README.md):

- **FxNet e redes de convolução:** Um snapshot do código de Comunità et al. com adaptações locais, servindo de fundação metodológica ([detalhes](experiments/fxnet_reproduction/README.md)).
- **Modelos Pré-treinados:** Validações iniciais do modelo AudioMAE (inferência AudioSet e exploração de embeddings) e do modelo PaSST (extração e análise exploratória de embeddings).

A baseline consolidada, cujo [notebook de treinamento](notebooks/train_baseline.ipynb) e [relatório de resultados](results/baseline/baseline_results_report.md) guiam o projeto, estabelece a referência de desempenho das CNNs tradicionais, sobre a qual as arquiteturas Transformer são comparadas.

## 9. Resultados de Transfer Learning (Linear Probing)

Após o estabelecimento da baseline, a investigação avançou para a aplicação de *Transfer Learning* extraindo os embeddings gerados pelos *encoders* congelados dos modelos PaSST, AST, AudioMAE e HTS-AT (cujos detalhes técnicos estão documentados na [Issue #5](https://github.com/caduobristo/tcc/issues/5)). O treinamento se restringiu a uma cabeça linear (*Linear Probing*) para realizar a classificação sobre as 13 classes de efeitos de guitarra.

A tabela abaixo resume o **F1 macro** obtido pelos modelos nesta primeira abordagem (antes da adaptação temporal nos modelos que não a possuem nativamente). O HTS-AT obteve os melhores resultados de forma consistente, o que serviu de grande indício, pois sua arquitetura já realiza uma interpolação temporal de forma nativa:

| Dataset | PaSST | AST | AudioMAE | HTS-AT |
| :--- | ---: | ---: | ---: | ---: |
| Mono Discrete | 82,22% | 83,81% | 73,37% | 90,49% |
| Mono Continuous | 83,95% | 87,03% | 77,13% | 91,59% |
| Poly Discrete | 83,24% | 87,94% | 70,70% | 92,57% |
| Poly Continuous | 84,84% | 90,31% | 77,10% | 93,89% |

Nesta etapa, observou-se que modelos como o AudioMAE apresentaram maiores dificuldades com as resoluções temporais padrão dos datasets em comparação ao HTS-AT, sugerindo a eficácia do tratamento temporal dos dados.

## 10. Impacto da Adaptação Temporal

A partir dos resultados e descobertas do *Linear Probing* do HTS-AT, elaborou-se a hipótese de que a compatibilização do tamanho do espectrograma (alongamento temporal por interpolação bicúbica, inspirada no próprio HTS-AT) aos outros modelos pré-treinados poderia aprimorar a extração das features.

Os achados exploratórios desse experimento (debatidos na [Issue #6](https://github.com/caduobristo/tcc/issues/6)) validaram a hipótese com ganhos expressivos:

- **O alongamento melhorou o desempenho dos três modelos testados:** Ocorreu ganho de acurácia e F1 macro médios nos 12 pares de modelo/dataset avaliados (o HTS-AT não foi submetido a esta mudança pois a técnica já é o seu padrão).
- **AST consolidou os melhores resultados:** Apresentou o maior F1 macro entre os três modelos adaptados em todos os cenários testados.
- **AudioMAE teve o salto mais expressivo:** O modelo se beneficiou drasticamente da interpolação temporal. Em Poly Discrete, passou de 70,70% (na abordagem inicial) para 90,67% (um ganho de quase 20 pontos percentuais).

**F1 macro final** obtido pelas variantes com alongamento temporal (média de 5 inicializações):

| Dataset | PaSST | AST | AudioMAE |
| :--- | ---: | ---: | ---: |
| Mono Discrete | 84,86% | 90,74% | 87,55% |
| Mono Continuous | 85,34% | 91,78% | 88,88% |
| Poly Discrete | 86,19% | 92,89% | 90,67% |
| Poly Continuous | 87,38% | 93,40% | 92,00% |

Apesar da eficácia do pós-processamento temporal para aproximar o desempenho das arquiteturas, certas classes de efeitos contíguos (ex: distinção entre os pedais de overdrive TS9 e 808) continuaram apresentando erros sistemáticos.

## 11. Conclusões e Próximos Passos

Os experimentos realizados até agora viabilizaram a construção de um pipeline reprodutível para classificação de efeitos e revelaram o potencial das arquiteturas de atenção sobre as convolucionais tradicionais. A descoberta do impacto da interpolação temporal foi um marco, demonstrando que a forma como o sinal acústico é mapeado para a resolução nativa do *encoder* pode ser tão decisiva quanto a escolha da arquitetura.

Com base nos sucessos e gargalos mapeados (como os limites do *Linear Probing* para desambiguação de pedais similares com *encoders* congelados), as próximas etapas da pesquisa envolvem:

- **Investigação da adaptação temporal:** Analisar e documentar mais a fundo as razões pelas quais o alongamento do espectrograma gera ganhos tão expressivos e altera o comportamento interno das camadas de atenção.
- **Fine-tuning estratégico:** Realização de *fine-tuning* (parcial ou total) direcionado prioritariamente aos modelos que apresentaram os melhores resultados e maior capacidade de adaptação, otimizando o uso dos recursos computacionais da pesquisa.
- **Desambiguação de classes:** Estruturação de técnicas adicionais e ampliação do dataset sintético para contornar desbalanceamentos e as confusões persistentes entre pedais de características muito próximas (como TS9 e 808).

