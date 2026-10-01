# Consolidação do transfer learning: PaSST e HTS-AT

Protocolo definido em 30/09/2026 antes desta seleção. Escopo: **linear probe**, encoder pré-treinado congelado e cabeça linear de 768 entradas/13 saídas (9.997 parâmetros). Fine-tuning permanece para outra etapa.

## Dados e comparação com o AST

Usamos os mesmos 123.552 WAVs Mono Discrete auditados: 13 efeitos, sem MT2/NoFX, 624 gravações de origem. A divisão por origem continua com seed 42: treino 88.902 amostras/449 fontes; validação 9.702/49; teste 24.948/126. Manifest SHA-256: `ddeb86c177deab02e364585be60680866b8cbc94926840dc669d3c17e02c816b`.

Os WAVs são reamostrados para 32 kHz e passam pelo frontend nativo de cada checkpoint: 128 bandas no PaSST, 64 no HTS-AT. Os mel32 Kaldi recebidos da dupla não são a entrada destes modelos. Reutilizamos os embeddings completos da primeira execução, após verificar hashes, versões, implementação, dimensões e valores finitos. Nenhum encoder é instanciado no treinamento desta consolidação.

O notebook `train_ast` inspirou a estratégia de congelar o encoder e treinar uma camada linear, mas não é uma reprodução idêntica: ele usa outro frontend e divisão aleatória por arquivo. O batch 4 do AST produz cerca de 22.239 atualizações por época; aqui batch 128 produz 695 e batch 1.024 produz 87. Assim, época e tempo não representam o mesmo esforço de otimização entre esses protocolos. O cache elimina a repetição do encoder a cada época.

## Seleção previamente delimitada

Configuração executável: `configs/linear_probe/consolidation_mono_disc.json`.

1. Oito configurações por modelo: batch 128/1.024 × normalização ausente/padronização por dimensão × pesos de classe ausentes/inversamente proporcionais às frequências de treino.
2. Busca inicial com seed de treinamento 42; promoção das duas melhores configurações por F1 macro de validação.
3. Finalistas também treinadas com seeds 7 e 21. Escolha pela média do melhor F1 macro de validação nas três seeds; desempate pelo identificador da configuração. Essas seeds não alteram a divisão dos dados.
4. Gravação de `selection_locked.json`, com hash, antes das execuções de confirmação.
5. Configuração escolhida treinada do início com cinco seeds novas: 101, 202, 303, 404 e 505. Todas as dez cabeças são finalizadas e registradas antes de qualquer avaliação de teste nesta consolidação.
6. Avaliação de cada melhor checkpoint no teste; apresentação de média e desvio padrão amostral entre as cinco seeds.

Total previsto: 34 cabeças (17 por modelo), com orçamento máximo de 200 épocas cada. AdamW, learning rate inicial 0,001 e weight decay 0,01. ReduceLROnPlateau usa F1 macro de validação, fator 0,5, patience 5, melhoria absoluta 0,001, LR mínimo 0,00001. Parada antecipada após 15 épocas sem melhoria acumulada superior a 0,001. O checkpoint salvo é o maior F1 macro observado, mesmo que uma melhoria pequena não reinicie a paciência. A validação usa cross-entropy sem pesos para tornar sua loss comparável entre candidatos; no treino, a loss ponderada é agregada pelo denominador dos pesos.

Média/desvio dos embeddings e pesos das classes são ajustados **somente no treino** e salvos junto a cada cabeça. Colunas de desvio menor que 1e-6 usam esse piso. Não há augmentation, nova extração, alterações nos pesos do encoder nem busca adicional guiada pelo teste.

O executor fixa sementes, limita threads CPU a quatro, desabilita TF32 e solicita algoritmos determinísticos do PyTorch. Isso melhora repetibilidade no mesmo ambiente; não garante identidade numérica entre versões e dispositivos.

## Limites da conclusão

O teste já foi consultado na rodada inicial de dez épocas. Portanto, esta avaliação **não é um teste totalmente cego**. Os resultados antigos motivaram investigar classes desbalanceadas; a escolha atual usa apenas validação. Antes de uma conclusão final do TCC, convém usar outro conjunto de fontes ainda não consultado ou um protocolo de validação externa.

O desvio entre seeds mede variação de inicialização/ordem do treinamento com a **mesma divisão**, não a incerteza entre novas gravações. As amostras compartilham fontes e presets; não são todas observações independentes. Não interpretar diferenças pequenas entre modelos como significância estatística. A comparação com a primeira execução (uma seed) é descritiva.

O limite de 200 épocas pode encerrar uma execução ainda melhorando; `stop_reason`, curvas, melhor época e número de atualizações ficam registrados. Este trabalho consolida a etapa congelada no cenário Mono Discrete; outros cenários e comparação controlada com AST continuam pendentes.

## Execução e artefatos

Ambiente usado: Windows, Python 3.11, PyTorch/Torchaudio 2.10.0 + CUDA 12.8, RTX 5080 de 16 GB. Dependências fixadas em `requirements-transfer.txt`; versões da extração e SHA-256 dos pesos estão em `cache_identity` no resumo portátil. Kernel `tcc-transfer`; ambiente local `.venv-transfer` sobre o ambiente existente da reprodução FxNet. O script `scripts/setup_transfer_environment.ps1` prepara a camada local; para uma máquina nova, usar `-Standalone -BasePython <python311>` em checkout sem ambiente prévio.

Pré-requisitos para **repetir esta consolidação**: manifesto auditado `data/manifests/transfer_learning/mono_disc/samples.csv` e `samples.json`, pesos oficiais em `checkpoints/transfer_learning/pretrained/` e caches completos `results/transfer_learning/mono_disc/{passt,htsat}/embeddings.npy`/`embedding_cache.json`. Todos são locais; o Git não transporta esses arquivos. A extração é uma etapa anterior, habilitada explicitamente nos notebooks `train_passt.ipynb`/`train_htsat.ipynb`, usando os WAVs e metadados validados já preparados. Em instalação sem esses dados, restaurar essa etapa antes de executar a consolidação; o executor recusa caches ausentes ou incompatíveis e não os substitui silenciosamente.

Preparação anterior: `scripts/prepare_transfer_models.py` baixa/verifica pesos; `scripts/prepare_transfer_data.py --scenario mono_disc --extract` restaura WAVs e CSVs oficiais; `scripts/prepare_transfer_experiment.py index --scenario mono_disc` cria o manifesto a partir dos WAVs e dos CSVs alinhados em `data/metadata/guitar_fx_dist/validated/Mono_Discrete/<efeito>/proc_settings.csv`. Esses CSVs alinhados são pré-requisito do indexador, não são produzidos pelo download. Na máquina deste experimento já estão preservados com auditorias locais. O catálogo e os documentos antigos de auditoria foram mantidos localmente após a simplificação remota em `5a45230`; a consolidação tem este protocolo e seu próprio resumo versionado.

```powershell
& .\.venv-transfer\Scripts\python.exe scripts/consolidate_transfer_learning.py
# Mostra o plano, sem treinar.
& .\.venv-transfer\Scripts\python.exe scripts/consolidate_transfer_learning.py --run
```

O executor cria uma sessão nova em `results/transfer_learning/mono_disc/consolidation_*`, preservando execuções anteriores. Registra plano/hash da implementação, caches, horários, histórico por época, melhor checkpoint, estatísticas de pré-processamento, seleção, lock de todas as cabeças, métricas e predições. Os arquivos de dados, embeddings e pesos permanecem ignorados pelo Git. Não executar novamente só para consultar resultados: use o relatório e o notebook de leitura.

O relatório consolidado fica em [results/transfer_learning_consolidated_report.md](../results/transfer_learning_consolidated_report.md); o resumo portátil em JSON está ao lado dele. O [notebook de consulta](../notebooks/consolidate_transfer_learning.ipynb) lê resultados e gráficos sem iniciar treinamento (`RUN_CONSOLIDATION=False`). A execução original é pelo CLI; a cópia executada do notebook registra a leitura desses resultados, não uma segunda execução de treinamento.

Para verificar as dez cabeças e regenerar o relatório após uma sessão completa:

```powershell
& .\.venv-transfer\Scripts\python.exe scripts/report_transfer_consolidation.py results/transfer_learning/mono_disc/consolidation_<sessao>
```

Para inferência futura, `src.models.consolidated_probe.load_consolidated_probe` recarrega o encoder oficial, a cabeça e o scaler ajustado no treino. Preserva encoder congelado/eval e precisão CUDA BF16 da extração; cabeça e scaler usam float32. O tensor de entrada deve conter waveform mono a 32 kHz com 64.000 amostras. O relatório identifica uma cabeça representativa pela melhor validação entre as cinco seeds, sem seleção por teste.
