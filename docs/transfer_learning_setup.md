# Preparação de PaSST e HTS-AT: encoder congelado

O primeiro experimento é **GUITAR-FX-DIST / Mono Discrete**, 13 efeitos, excluindo MT2 e todas as pastas NoFX. Os notebooks `notebooks/train_passt.ipynb` e `notebooks/train_htsat.ipynb` seguem as etapas do notebook AST: configuração, carregamento, modelo pré-treinado, cabeça classificadora e avaliação. Nesta etapa não há fine-tuning do encoder.

## Estado e dados

A restauração dos WAVs oficiais do Mono Discrete foi concluída. O pacote completo tem 26,77 GB, dividido em 12 volumes `.z01`–`.z12` e um `.zip`. Todos passaram no MD5 publicado no [Zenodo 4298000](https://zenodo.org/records/4298000); o 7-Zip extraiu 164.736 WAVs e 14 CSVs sem erro de CRC. Os volumes ficam em `data/archives/guitar_fx_dist/official/Mono_Discrete/`; a extração preserva apenas WAVs e CSVs em `data/raw/guitar_fx_dist/`. Os outros três cenários são suportados pelos scripts, mas não foram baixados nesta preparação. Seus ZIPs mel16/mel32 existentes continuam preservados.

O índice compartilhado fica em `data/manifests/transfer_learning/mono_disc/samples.csv`. Sua criação verifica integralmente os WAVs selecionados: formato, duração, valores finitos e SHA-256; confere os nomes contra os CSVs derivados validados. Arquivos oficiais podem ter 88.201 amostras, incluindo o último ponto do intervalo, em vez de 88.200. O loader reamostra 44,1 para 32 kHz e corta esse ponto adicional para produzir exatamente 64.000 amostras (2 segundos). Não multiplica a waveform por 32768.

As partições são aproximadamente 72/8/20 **por áudio de origem**, seed 42. O ID reconstruído é, por exemplo, `G61-40100-1111-20593` para todos os efeitos/configurações derivados dessa gravação. Inclui o file ID final, além de instrumento e nota. Todas as versões da mesma fonte permanecem no mesmo conjunto. A indexação recusa conteúdo WAV idêntico atravessando partições e exige as 13 classes em cada conjunto. As partições diferem das divisões aleatórias por arquivo do AST histórico; seus resultados anteriores não são comparações diretas neste novo protocolo.

Os **123.552 WAVs selecionados** passaram na auditoria completa: 624 fontes, nenhuma duplicata de conteúdo WAV e nenhum vazamento entre partições. Há 88.902 exemplos/449 fontes de treino, 9.702/49 de validação e 24.948/126 de teste. Todos têm 44.100 Hz, mono e 88.201 amostras. O manifest possui SHA-256 `ddeb86c177deab02e364585be60680866b8cbc94926840dc669d3c17e02c816b`. O loader confere o SHA-256 de cada WAV novamente durante sua utilização.

## Representação e modelos

Os `mel32` recebidos da dupla não são usados como entrada nativa destes checkpoints. A taxa coincide, mas a receita Kaldi, a escala e o número de bandas não bastam para garantir a representação esperada. Trabalhamos com os WAVs e o frontend próprio de cada modelo:

| Modelo | Pesos selecionados | Processamento e vetor usado |
| --- | --- | --- |
| PaSST | `passt_s_swa_p16_128_ap476`, AudioSet | `hear21passt` 0.0.26, frontend oficial de 128 bandas, 32 kHz, janela 800, FFT 1024, hop 320, normalização interna. Vetor médio dos tokens de classificação/distilação, 768 dimensões, antes da cabeça AudioSet. |
| HTS-AT | `HTSAT_AudioSet_Saved_1.ckpt`, AudioSet | Core oficial no commit `2e29471fce7e770edeaaa64a41908bf234844fc5`, frontend de 64 bandas, 32 kHz, FFT/janela 1024, hop 320, 50–14.000 Hz e BatchNorm do checkpoint. `latent_output` de 768 dimensões anterior à camada token-semantic/classificação. |

O frontend fica em float32; a inferência do encoder usa bfloat16 em CUDA. Todos os parâmetros do encoder ficam congelados e o módulo permanece em `eval`, incluindo BatchNorm e augmentações. O PaSST aceita o trecho curto e recorta suas posições temporais; Patchout e máscaras estão desativados para este linear probing. O HTS-AT usa sua rotina oficial para interpolar/reorganizar o espectrograma curto para a grade do Swin (`infer_mode=False`, sem repetir a waveform). Esse tratamento de duração é parte explícita do protocolo; não é padding a 10 segundos nem reprodução integral do treino original em AudioSet.

Pesos locais, lidos com `weights_only=True`, possuem SHA-256 fixado e precisam corresponder a todas as chaves/tamanhos do modelo (`strict=True`). Nunca se permite usar pesos aleatórios no lugar dos pré-treinados. As fontes e hashes também ficam em `checkpoints/transfer_learning/pretrained/sources.json`, ignorado pelo Git. O core HTS-AT foi incluído em `src/vendor/htsat/` com a licença MIT; as alterações são import relativo, utilitários necessários e limpeza de espaços finais. Código oficial: [PaSST/HEAR](https://github.com/kkoutini/passt_hear21), [HTS-AT](https://github.com/RetroCirce/HTS-Audio-Transformer).

## Ambiente

Kernel registrado: **TCC Transfer Learning (PyTorch CUDA)** (`tcc-transfer`), executável `.venv-transfer/Scripts/python.exe`. O ambiente é uma camada local com `--system-site-packages` sobre o Python 3.11 do FxNet, reutilizando PyTorch 2.10.0+cu128, Torchaudio 2.10.0+cu128 e Torchvision 0.25.0+cu128. As dependências adicionais ficam em `.venv-transfer`; o ambiente base não foi modificado. A junção antiga de compatibilidade e os kernels anteriores foram preservados.

`requirements-transfer.txt` fixa as dependências principais. Para reconstruir nesta máquina:

```powershell
.\scripts\setup_transfer_environment.ps1
```

Em um checkout novo, com Python 3.11 e sem esse ambiente base, use `-BasePython CAMINHO_DO_PYTHON -Standalone`: cria um venv e instala a distribuição CUDA 12.8 antes das demais dependências. Para outra GPU/plataforma, ajustar a distribuição PyTorch; a configuração atual e a estimativa são específicas desta RTX 5080. Este projeto não depende de instalar TorchCodec para WAV: a leitura usa SoundFile.

## Preparação reproduzível, sem treinamento

Da raiz do projeto:

```powershell
$pyTransfer = ".\.venv-transfer\Scripts\python.exe"
& $pyTransfer scripts/prepare_transfer_models.py
& $pyTransfer scripts/prepare_transfer_data.py --scenario mono_disc --extract --workers 3
& $pyTransfer scripts/prepare_transfer_experiment.py index --scenario mono_disc
& $pyTransfer scripts/prepare_transfer_experiment.py preflight --model passt
& $pyTransfer scripts/prepare_transfer_experiment.py preflight --model htsat
```

O downloader retoma `.partial` por HTTP Range, verifica os volumes e usa 7-Zip para o ZIP multipartes. Não sobrescreve arquivos existentes diferentes ou ignora um erro de checksum. A extração não publica dados e não materializa as features antigas do baseline, que continuam nos volumes oficiais. O teste prévio recusa ausência de índice/weights, alteração de WAVs, mapeamento de classes inconsistente ou fonte atravessando partições.

## Quando decidir iniciar

Nos dois notebooks, **`RUN_EXTRACTION=False` e `RUN_TRAINING=False`**. Abrir ou executar todas as células nessa configuração não inicia nenhuma das duas etapas longas. O primeiro controle autoriza uma passagem de inferência por todo o cenário para gerar embeddings; o segundo autoriza treinar a cabeça. Esta entrega não gera esses embeddings e não executa épocas/optimizer steps do experimento.

A extração salva `(N, 768)` float32 em arquivo memmap local: cerca de **0,35 GiB** para 123.552 amostras, evitando cache de espectrogramas ampliados em RAM. Um cache só é reutilizado quando sua conclusão, checksum, índice, checkpoint, implementação, precisão e versões coincidem. Um arquivo parcial interrompido é recusado; preservá-lo/renomeá-lo antes de reiniciar a extração, em vez de reutilizá-lo como completo.

O treinamento usa somente `Linear(768, 13)`: 9.997 parâmetros, CrossEntropyLoss, AdamW, LR 0,001, weight decay 0,01, 10 épocas e batches de embeddings de 1024. A extração usa batches de áudio de 16, quatro workers e GPU CUDA. O tamanho maior da cabeça é possível porque ela trabalha sobre vetores pequenos; não representa batch 1024 de Transformers. O checkpoint é escolhido na validação e avaliado depois no teste. Rótulos de teste não participam da extração do encoder congelado nem de seu ajuste.

Cada execução futura cria `results/transfer_learning/<cenario>/<modelo>/probe_<datahora>/`, com `best_head.pt`, histórico, configuração/cache, relatório por classe, métricas globais e matriz de confusão. Dados, caches, pesos, ambiente, manifests e resultados de execução estão ignorados pelo Git. Código, notebooks e configurações podem ser versionados.

## Estimativa e verificações

Hardware verificado: **RTX 5080, 16.303 MiB de VRAM**, Ryzen 7 9800X3D e aproximadamente 30,88 GiB de RAM. Após a restauração, pequenos benchmarks de inferência usaram os checkpoints reais e uma amostra aleatória de 128 WAVs do manifest auditado, em oito batches de 16. A leitura/reamostragem foi medida separadamente e nenhum embedding completo foi persistido. Um subconjunto anterior, obtido durante o download, permanece explicitamente identificado em `data/raw/guitar_fx_dist/subsets/benchmark/`; não define partições do experimento.

Na medição curta em batch 16, PaSST processou aproximadamente 1.115 exemplos/s, HTS-AT aproximadamente 1.426 exemplos/s; leitura, hash e reamostragem serial custaram 1,5–3,3 ms por WAV. A projeção para **123.552 exemplos + 10 épocas da cabeça** deu aproximadamente 3–11 minutos para PaSST e 2–17 minutos para HTS-AT. Para planejamento da primeira execução, reservar **5–20 minutos por modelo**, considerando inicialização de workers, leitura de muitos arquivos, escrita de cache e variação de throughput. O treinamento da cabeça com cache pronto é estimado em dezenas de segundos a cerca de dois minutos.

São projeções, não durações de treinamento observadas: o benchmark não mediu backward/AdamW, e o tempo de download/preparação dos dados não está incluído. A execução real futura permitirá atualizar os valores. As medições detalhadas ficam em `data/audits/transfer_learning/*_benchmark.json` e podem ser repetidas com `scripts/prepare_transfer_experiment.py benchmark --model passt|htsat`, que não possui comando de treino.

Verificações realizadas: importações e dependências (`pip check`), carregamento estrito dos dois checkpoints, inferência finita de 768 dimensões na GPU, equivalência exata entre o adaptador HTS-AT e seu vetor latente oficial em float32, auditoria completa dos WAVs selecionados e 14 testes de integridade/partições/isolamento do encoder/controles de execução. Um teste do DataLoader no Windows leu 64 WAVs com quatro workers, nos batches/tamanhos configurados, sem carregar encoder ou treinar. Os dois notebooks foram conferidos com extração e treinamento desativados, sem salvar outputs de execução nem gerar caches/resultados do experimento.
