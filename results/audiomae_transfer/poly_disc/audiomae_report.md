# Transfer Learning AudioMAE: Poly Discrete

> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_audiomae_results.py`.

Resultados baseados no experimento executado em: `probe_20261003_194550`

## 1. Configurações de Treinamento

- **Learning Rate:** 0.001
- **Máximo de Épocas:** 200
- **Paciência (Early Stopping):** 15
- **Melhor Época Alcançada:** 113

## 2. Desempenho no Teste

| Métrica | Resultado |
| :--- | ---: |
| **Acurácia (Média Multi-seed)** | 74.84% ± 0.31% |
| **Acurácia (Melhor Época)** | 74.74% |
| **F1 Macro** | 70.70% |
| **Precisão Macro** | 72.37% |
| **Recall Macro** | 70.40% |

## 3. Desempenho Detalhado por Classe

| Efeito | F1-Score (%) | Recall (%) | Precisão (%) | Suporte |
| :--- | ---: | ---: | ---: | ---: |
| 808 | 50.60 | 49.47 | 51.79 | 1700 |
| BD2 | 82.65 | 85.18 | 80.27 | 1700 |
| BMF | 87.69 | 89.71 | 85.77 | 1700 |
| DPL | 47.53 | 38.24 | 62.80 | 340 |
| DS1 | 78.70 | 79.24 | 78.18 | 1700 |
| FFC | 82.20 | 82.59 | 81.82 | 425 |
| MGS | 78.83 | 84.88 | 73.58 | 1700 |
| OD1 | 22.74 | 15.88 | 40.00 | 340 |
| RAT | 86.50 | 86.12 | 86.88 | 1700 |
| RBM | 98.63 | 99.59 | 97.69 | 1700 |
| SD1 | 60.92 | 60.53 | 61.32 | 1700 |
| TS9 | 51.61 | 50.41 | 52.87 | 1700 |
| VTB | 90.54 | 93.41 | 87.83 | 425 |

*(Consulte a matriz de confusão `confusion_matrix.png` gerada na pasta da run para a análise matricial e de falsos positivos completos que constam no JSON.)*
