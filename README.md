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

Como próximos passos, destacam-se:

- Aplicação de **transfer learning** a partir de modelos pré-treinados (ex: AudioSet), adaptando-os ao domínio específico de efeitos de guitarra  
- Substituição da camada de classificação original por uma **rede fully connected** ajustada ao novo conjunto de classes (timbres/efeitos), em substituição às categorias genéricas utilizadas nos datasets originais  
- Treinamento supervisionado dessa nova camada de classificação, mantendo o backbone do Transformer congelado (feature extractor)  
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
