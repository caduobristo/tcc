# Estudo da branch de transfer learning

Revisão de 30/09/2026 da implementação AST em `feature/transfer-learning`, originalmente no commit `02563bb`, após integrar a organização da `main` (`9d0f2b2`). A integração preservou os cinco arquivos específicos da branch. Os conflitos em `.gitignore` e `notebooks/train_baseline.ipynb` foram resolvidos com as versões da `main`, que contêm as exclusões de dados e os caminhos padronizados do baseline.

## O que foi implementado e executado

| Arquivo | Função e evidência |
| --- | --- |
| `notebooks/generate_features.ipynb` | Seleciona um dos quatro cenários e gera Log-Mel a 16 ou 32 kHz. Não contém saídas de uma execução completa. |
| `src/data/feature_extraction.py` | Converte WAV para mono, reamostra, multiplica a waveform por 32768 e calcula Kaldi Fbank com 128 bandas, janela Hanning e passo de 10 ms. Salva `.npy` por efeito; usa processos paralelos via joblib. |
| `src/data/dataset_ast.py` | Carrega arrays, associa classes/parâmetros, normaliza e preenche ou corta para 1024 quadros. Pode guardar cada item processado em RAM. |
| `src/models/ast_models.py` | Adaptação do AST de Yuan Gong, com encoder pré-treinado ImageNet/AudioSet e nova cabeça linear para os efeitos. |
| `notebooks/train_ast.ipynb` | Configura Mono Discrete a 16 kHz, 13 classes, MT2 e NoFX mono excluídos; prevê 10 épocas treinando somente a cabeça, batch 4 e AdamW com LR 0,001. |

O notebook AST registra **123.552 amostras, 13 classes, device DirectML e inicialização do modelo pré-treinado** na máquina do colega. A célula do laço de treinamento não tem execution count nem saídas. Não há checkpoint AST, métricas finais ou avaliação AST de teste nos arquivos locais inspecionados. Portanto, a evidência disponível confirma preparação e inicialização, sem confirmar um treinamento concluído. Treinar somente a cabeça é linear probing; o notebook não contém uma etapa posterior de ajuste do encoder.

Os ZIPs atuais têm contagens e representação compatíveis com essa seleção, mas não há hashes do produtor para provar que sejam exatamente os mesmos arquivos utilizados naquela execução. A identificação 16/32 kHz foi confirmada pelo usuário e pela configuração do notebook. **O AST desta branch seleciona 16 kHz.** O notebook reserva 32 kHz para outros modelos, mas a taxa de amostragem sozinha não certifica compatibilidade com um checkpoint.

## Problemas confirmados antes de treinar

### Caminhos e leitura dos dados

O gerador usa `../dataset/GUITAR-FX-DIST` e grava `mel_16/<cenario>/<efeito>/<arquivo>.npy`. O notebook AST usa `../datasets/GUITAR-FX` para Mono Discrete e `../../datasets/GUITAR-FX-DIST` para os outros cenários. Esses caminhos diferem entre si e da organização documentada em [`data/README.md`](../data/README.md).

O materializador da organização atual grava `data/processed/guitar_fx_dist/<variante>/<cenario>/Features/<efeito>/mel_198x128/<arquivo>.npy`, ou um subconjunto explicitamente identificado. O loader AST procura arrays diretamente na pasta do efeito. Em uma verificação com o subconjunto já materializado, havia 32 arrays; o loader identificou 13 classes após as exclusões, mas **indexou zero amostras**. Esse subconjunto serve para verificação e não constitui o conjunto completo de treinamento.

É necessário adaptar o loader e os notebooks aos caminhos centrais ou à leitura direta dos ZIPs, usando `src/data/paths.py` e `src/data/archive.py`. Os parâmetros devem vir dos CSVs derivados validados. O gerador da branch não copia os CSVs para sua saída; quando não os encontra, o loader usa `-1` para parâmetros. Isso não altera diretamente a classificação atual, que ignora esses parâmetros, mas impede usá-los como alvos confiáveis de regressão.

### Escala e normalização do AST

A [implementação oficial do AST](https://github.com/YuanGongND/ast/blob/master/src/dataloader.py) subtrai a média da waveform, calcula Fbank sem multiplicar por 32768, faz padding/corte e depois normaliza. A branch multiplica a waveform por 32768 e normaliza os arrays **antes** do padding. Logo, as duas rotinas não produzem a mesma entrada para o checkpoint.

Uma verificação com ruído sintético de dois segundos, seed 42, usando o mesmo Kaldi Fbank instalado, produziu `(198, 128)` nas duas taxas. A multiplicação elevou o Log-Mel em média em 20,632 unidades a 16 kHz e 20,794 a 32 kHz. O deslocamento ideal para energia é `2*ln(32768) = 20,794`; pisos numéricos explicam por que não é uniforme em todas as bandas. Uma amostra real mel16 foi lida sem extrair o ZIP e carregada pelo dataset AST para verificar a forma e o padding. Essa única amostra não estima estatísticas do dataset inteiro.

Com os valores atuais de média/desvio (`-4,27`/`4,57`), a branch acrescenta padding de valor 0 no espaço já normalizado. Aplicar padding 0 antes dessa mesma normalização produziria aproximadamente 0,467. **80,66% dos quadros de cada entrada são padding**: 826 dos 1024, para arrays de 198 quadros.

Antes de comparar resultados, definir e registrar a receita correspondente ao checkpoint e as estatísticas de normalização. Os WAVs não estão disponíveis localmente para regenerar e conferir a equivalência. Subtrair uma constante dos ZIPs inteiros sem verificar pisos e a receita de origem não comprovaria equivalência. Os dados originais foram preservados.

### Ambiente, pesos e memória

`ast_models.py` exige `timm==0.4.5` e importa `wget`. O `requirements.txt` do projeto não lista essas dependências nem `joblib` explicitamente. Os [requisitos oficiais históricos do AST](https://github.com/YuanGongND/ast/blob/master/requirements.txt) usam versões antigas de PyTorch; a compatibilidade com o ambiente moderno deste computador precisa ser verificada em ambiente próprio.

O checkpoint exigido em `checkpoints/pretrained_models/audioset_10_10_0.4593.pth` não foi encontrado. No ambiente local FxNet inspecionado, PyTorch/Torchaudio são 2.10, `timm`, `wget` e `torchcodec` não estão instalados. Uma tentativa de ler um WAV sintético pelo `process_single_audio` falhou com a exigência de TorchCodec, inclusive após o fallback `backend="soundfile"`. Esse resultado é específico desse ambiente, não uma prova de falha na máquina do colega.

O cache é preenchido sob demanda com tensores **já ampliados para 1024 × 128**. Para o cenário salvo, treino e validação acumulam 98.841 amostras: aproximadamente **48,26 GiB só de tensores**, antes de metadados, modelo e demais gastos. A máquina local dispõe de aproximadamente 30,88 GiB de RAM física. O default `CACHE_IN_RAM=True` não é adequado para o conjunto completo aqui. Avaliar leitura sem cache ou cache de representações menores, e o comprimento de entrada como escolha experimental registrada.

## Protocolo experimental pendente

- **Split:** `DataSplit` implementa aproximadamente 72% treino, 8% validação e 20% teste, apesar da docstring 80/10/10. Para 123.552 amostras: 88.956 / 9.885 / 24.711. O sorteio é por arquivo; não agrupa fonte sonora, preset ou execuções relacionadas. A auditoria identificou também pares de arrays idênticos em Poly Continuous, descritos em [`docs/data_inventory.md`](data_inventory.md). É necessário considerar agrupamento ao definir comparações sem vazamento.
- **Classes poly:** a exclusão do notebook contém apenas NoFX mono. Nos cenários poly, as duas pastas NoFX poly seriam incluídas como classes distintas, levando a 15 classes com MT2 excluído. Definir um protocolo comum entre cenários e modelos.
- **Avaliação:** o test loader é criado e não utilizado. Não há cálculo/persistência de precisão, recall, F1, confusão, histórico ou resultado final AST. A seleção pela validação deve ser seguida da avaliação do melhor checkpoint no teste reservado.
- **Falhas de entrada:** o dataset devolve `None` em erros e o collate descarta esses itens. O gerador pula arquivos existentes sem verificá-los, coleta erros como strings e o notebook imprime sucesso mesmo após erro ou ausência da pasta `Audio`. Registrar cobertura real e impedir sucesso aparente com dados ausentes ou inválidos.

## Verificações e próximo passo

Foram feitos leitura dos cinco arquivos, inspeção dos outputs dos notebooks, consulta ao código oficial AST, verificação de dependências/pesos e pequenas verificações em CPU com waveform sintética e uma amostra mel16 real. As nove verificações da organização de dados passaram após a integração da `main`. Não houve treinamento AST, download de pesos, alteração dos ZIPs nem troca dos ambientes existentes. Os achados não medem a qualidade de classificação do AST.

A branch é uma base útil para o experimento AST, mas ainda precisa integrar o acesso aos dados padronizados, alinhar representação/normalização, configurar o ambiente e checkpoint, ajustar cache/classes e completar o protocolo de teste. Esses pontos devem anteceder um treinamento completo. A integridade binária dos dados disponíveis e a compatibilidade científica com o modelo são verificações diferentes; os resultados da primeira estão registrados no inventário e nos relatórios locais ignorados pelo Git.
