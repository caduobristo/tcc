# Transfer Learning AudioMAE: Mono Continuous

> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_audiomae_results.py`.

Resultados baseados no experimento executado em: `probe_20261003_185245`

## 1. Configurações de Treinamento

- **Learning Rate:** 0.001
- **Máximo de Épocas:** 200
- **Paciência (Early Stopping):** 15
- **Melhor Época Alcançada:** 101

## 2. Desempenho no Teste

| Métrica | Resultado |
| :--- | ---: |
| **Acurácia (Média Multi-seed)** | 77.64% ± 0.32% |
| **Acurácia (Melhor Época)** | 77.58% |
| **F1 Macro** | 77.13% |
| **Precisão Macro** | 77.04% |
| **Recall Macro** | 77.49% |

## 3. Desempenho Detalhado por Classe

| Efeito | F1-Score (%) | Recall (%) | Precisão (%) | Suporte |
| :--- | ---: | ---: | ---: | ---: |
| 808 | 60.54 | 60.18 | 60.91 | 2092 |
| BD2 | 74.65 | 76.04 | 73.31 | 2066 |
| BMF | 95.26 | 94.77 | 95.76 | 2025 |
| DPL | 80.02 | 79.78 | 80.26 | 2008 |
| DS1 | 92.52 | 92.54 | 92.50 | 2145 |
| FFC | 85.36 | 86.47 | 84.28 | 2003 |
| MGS | 80.26 | 88.19 | 73.64 | 1999 |
| OD1 | 55.90 | 56.26 | 55.55 | 2037 |
| RAT | 83.12 | 83.26 | 82.98 | 2061 |
| RBM | 94.45 | 95.13 | 93.79 | 2031 |
| SD1 | 54.01 | 49.10 | 60.01 | 2002 |
| TS9 | 53.71 | 49.61 | 58.55 | 1947 |
| VTB | 92.92 | 95.99 | 90.04 | 1997 |

*(Consulte a matriz de confusão `confusion_matrix.png` gerada na pasta da run para a análise matricial e de falsos positivos completos que constam no JSON.)*
