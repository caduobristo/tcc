# PASST Embeddings Test

## Rodar

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python extract_passt_embeddings.py
```

## Dataset

- Use o GEC-GIM: [link](https://seafile.cloud.uni-hannover.de/d/5398a844db214b5fb31b/)
- O caminho padrão é `data/raw/gec_gim/` na raiz do TCC. Coloque os áudios diretamente nessa pasta, sem subpastas, ou passe `--dataset CAMINHO`. Também é possível configurar `TCC_DATA_ROOT`.
- Nome esperado: `ag_G_Classe40_0_.wav`
- O script lê a classe a partir do nome do arquivo.

## Resultados

- `results_10samples/` contém os resultados do experimento com 10 amostras por classe.
- A auditoria de 30/09/2026 confirmou 109 embeddings de 768 dimensões: 9 de Chorus e 10 de cada uma das outras 10 classes. Os WAVs de origem não foram encontrados localmente. A issue histórica chama o dataset de GEC-PIM, mas o datasheet vinculado identifica GEC-GIM; a origem exata deve ser confirmada quando os áudios forem restaurados. Consulte [o inventário](../../docs/data_inventory.md).
