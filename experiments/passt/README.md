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
- Coloque os áudios diretamente em `dataset/`, sem subpastas.
- Nome esperado: `ag_G_Classe40_0_.wav`
- O script lê a classe a partir do nome do arquivo.

## Resultados

- `results_10samples/` contém os resultados do experimento com 10 amostras por classe.
