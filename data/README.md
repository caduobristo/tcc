# Dados locais do TCC

O inventário detalhado está em [docs/data_inventory.md](../docs/data_inventory.md). O [catalog.json](catalog.json) registra disponibilidade, origem e hashes dos arquivos compactados. Datasets, metadados extraídos, modelos e manifestações por arquivo não são publicados no Git.

## Estrutura comum

```text
data/
  archives/guitar_fx_dist/             # 8 ZIPs de features + csvs.zip, nomes originais
  archives/import_bundles/             # pacote do Drive que contém outra cópia de csvs.zip
  raw/<dataset>/                      # WAVs originais, quando disponíveis
  processed/guitar_fx_dist/baseline/   # features originais da FxNet, ainda ausentes
  processed/guitar_fx_dist/mel16/      # Log-Mel de áudio a 16 kHz
  processed/guitar_fx_dist/mel32/      # Log-Mel de áudio a 32 kHz
  metadata/guitar_fx_dist/<cenario>/<efeito>/             # CSVs originais preservados
  metadata/guitar_fx_dist/validated/<cenario>/<efeito>/   # CSVs derivados e alinhados
  manifests/                          # hashes de todos os membros dos ZIPs e backups
  audits/                             # relatórios completos de integridade e consistência
```

`raw/` e `processed/.../baseline/` são destinos definidos, não datasets já baixados. Não foram criadas pastas vazias para representar disponibilidade. `dados/` permanece como junção local para `archives/guitar_fx_dist/`, preservando os nove nomes de arquivos antigos. As junções antigas dos ambientes também foram mantidas.

Os identificadores são `guitar_fx_dist`, `gec_gim`, `gepe_gim`, `idmt_smt_guitar`, `idmt_smt_audio_effects` e `audioset`. Os cenários do GUITAR-FX preservam os nomes `Mono_Continuous`, `Mono_Discrete`, `Poly_Continuous` e `Poly_Discrete`, assim como os nomes de efeitos e amostras.

Por padrão, os caminhos são relativos à raiz deste repositório. Para utilizar outra unidade, configure `TCC_DATA_ROOT` apontando para a pasta equivalente a `data/`. Não é necessário alterar o diretório pessoal ou mover ambientes Conda.

## Representações diferentes

O usuário confirmou em 30/09/2026 que mel16 e mel32 correspondem a áudio reamostrado para **16 kHz e 32 kHz**, respectivamente. Os ZIPs vieram do Drive da dupla. O notebook `notebooks/generate_features.ipynb` e o módulo `src/data/feature_extraction.py` da branch `feature/transfer-learning`, inspecionados no commit `02563bb`, registram a geração: conversão para mono, reamostragem, multiplicação da waveform por 32768 e Kaldi Fbank, com 128 bandas, janela Hanning, passo de 10 ms e dither 0. Todos os NPYs são `float32`, com shape `(198, 128)`. Os nomes e as quantidades de arquivos coincidem entre as variantes, mas seus conteúdos diferem. A confirmação de origem e o código não substituem checksums do produtor ou validação da escala contra o checkpoint escolhido.

O notebook de geração indica 16 kHz para AST/AudioMAE e 32 kHz para PaSST/HTS-AT. Portanto, a escolha da variante depende do modelo e do pré-processamento esperado por seu checkpoint; não há motivo para adotar 32 kHz para todos os experimentos. A compatibilidade completa exige conferir escala e normalização, além da taxa de amostragem.

A FxNet preservada recebe 87 quadros e 128 bandas. As features históricas `mel_22050_1024_512` devem ser restauradas em:

```text
data/processed/guitar_fx_dist/baseline/<cenario>/Features/<efeito>/
  proc_settings.csv
  mel_22050_1024_512/<amostra>.npy
```

O notebook do baseline já procura esse caminho e falha explicitamente quando as features não existem ou têm shape incompatível. Os ZIPs atuais não foram vinculados a esse caminho nem convertidos artificialmente para a forma esperada pela FxNet.

Os scripts AudioMAE procuram WAVs em `data/raw/guitar_fx_dist/<cenario>/Audio/<efeito>/`; o PaSST procura áudios diretamente em `data/raw/gec_gim/` e aceita `--dataset` para outro arranjo. O avaliador histórico FxNet usa o destino central das features. Os outros notebooks de reprodução preservam os caminhos históricos de referência, que precisam ser ajustados ao restaurar os dados.

## Acesso sem extrair todos os ZIPs

```python
from src.data.archive import GuitarFxArchive

with GuitarFxArchive("mono_cont", "mel32") as dataset:
    print(dataset.class_counts())  # padrão: exclui MT2 e as duas pastas NoFX
    effect, sample = next(dataset.iter_samples())
    mel = dataset.read_feature(effect, sample)  # ndarray float32 (198, 128)
    settings = dataset.read_settings(effect, sample)
```

Esse leitor verifica o CRC da amostra e recusa arrays com shape inesperado, valores não finitos ou conteúdo que exija pickle. Os CSVs usados são os derivados validados. Não há rótulo de pedal nos CSVs das pastas NoFX; `read_settings` não está disponível para elas. Este leitor fornece os dados; não adapta automaticamente a arquitetura do baseline.

## Auditoria e extração seletiva

Com Python e NumPy instalados, a partir da raiz do projeto:

```powershell
python scripts/audit_datasets.py audit --workers 2
python scripts/audit_datasets.py reconcile
python scripts/audit_datasets.py materialize --scenario mono_cont --variant mel32 --max-per-class 2
```

`audit` lê todas as matrizes, confere CRC, shape, dtype, valores finitos, metadados e SHA-256 dos ZIPs e de cada NPY. Pode demorar e ler mais de 200 GB somando hashes e conteúdo. O retorno é 1 quando há inconsistências nos metadados originais, mesmo que a integridade binária seja boa: consulte separadamente `binary_integrity_ok`, `metadata_source_consistent` e os detalhes de `metadata_consistency.json`. `reconcile` usa os manifests já auditados para produzir os CSVs derivados; não é uma nova leitura completa dos dados nem substitui `audit` após trocar arquivos.

`materialize` preserva as 14 classes de efeitos e ambas as pastas NoFX. Sem `--max-per-class`, extrai um cenário completo para `processed/guitar_fx_dist/<variante>/<cenario>/Features/`. Com limite, cria um subconjunto explicitamente identificado em `<variante>/subsets/first_<N>_per_class/<cenario>/Features/`. Dentro de cada efeito, os NPYs ficam em `mel_198x128/`. Nunca se chama essa representação de `mel_22050_1024_512`.

A extração verifica espaço livre, conserva uma reserva de 10 GiB e recusa sobrescrever arquivos com conteúdo diferente. Os ZIPs originais são mantidos. Foi materializado somente um subconjunto de verificação com 32 matrizes (2 por pasta) de Mono Continuous/mel32; isso não é o cenário completo nem um split de treinamento.

## Limites da validação

Os hashes locais permitem detectar alterações futuras. Sem os hashes do produtor do Drive, não comprovam identidade byte a byte com os arquivos que originaram as execuções do colega ou com as features originais do Zenodo. A branch de transfer learning registra uso de mel_16 pelo AST, mas isso não prova um treinamento concluído. Modelos e saídas continuam nos respectivos experimentos; são artefatos de execução, não datasets de entrada.
