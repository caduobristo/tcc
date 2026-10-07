# Transfer Learning AudioMAE: Poly Continuous

> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_audiomae_results.py`.

Resultados baseados no experimento executado em: `probe_20261003_170521`

## 1. Configurações de Treinamento

- **Learning Rate:** 0.001
- **Máximo de Épocas:** 200
- **Paciência (Early Stopping):** 15
- **Melhor Época Alcançada:** 116

## 2. Desempenho no Teste

| Métrica | Resultado |
| :--- | ---: |
| **Acurácia (Média Multi-seed)** | 77.68% ± 0.50% |
| **Acurácia (Melhor Época)** | 77.40% |
| **F1 Macro** | 77.10% |
| **Precisão Macro** | 77.11% |
| **Recall Macro** | 77.41% |

## 3. Desempenho Detalhado por Classe

| Efeito | F1-Score (%) | Recall (%) | Precisão (%) | Suporte |
| :--- | ---: | ---: | ---: | ---: |
| 808 | 48.79 | 43.11 | 56.19 | 2074 |
| BD2 | 79.52 | 81.65 | 77.50 | 1995 |
| BMF | 88.95 | 89.77 | 88.16 | 2023 |
| DPL | 79.33 | 78.86 | 79.81 | 2025 |
| DS1 | 84.35 | 84.35 | 84.35 | 2090 |
| FFC | 88.66 | 89.76 | 87.58 | 2042 |
| MGS | 78.85 | 85.98 | 72.82 | 2075 |
| OD1 | 64.57 | 65.66 | 63.52 | 2021 |
| RAT | 84.26 | 80.73 | 88.12 | 2039 |
| RBM | 99.16 | 99.36 | 98.97 | 2023 |
| SD1 | 55.86 | 52.57 | 59.59 | 2039 |
| TS9 | 53.72 | 56.23 | 51.42 | 2024 |
| VTB | 96.32 | 98.26 | 94.46 | 2064 |

*(Consulte a matriz de confusão `confusion_matrix.png` gerada na pasta da run para a análise matricial e de falsos positivos completos que constam no JSON.)*
