# Transfer Learning AST: Mono Continuous

> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_ast_results.py`.

Resultados baseados no experimento executado em: `probe_20261002_011419`

## 1. Configurações de Treinamento

- **Learning Rate:** 0.001
- **Máximo de Épocas:** 200
- **Paciência (Early Stopping):** 15
- **Melhor Época Alcançada:** 109

## 2. Desempenho no Teste

| Métrica | Resultado |
| :--- | ---: |
| **Acurácia (Média Multi-seed)** | 87.22% ± 0.27% |
| **Acurácia (Melhor Época)** | 87.24% |
| **F1 Macro** | 87.03% |
| **Precisão Macro** | 87.06% |
| **Recall Macro** | 87.16% |

## 3. Desempenho Detalhado por Classe

| Efeito | F1-Score (%) | Recall (%) | Precisão (%) | Suporte |
| :--- | ---: | ---: | ---: | ---: |
| 808 | 68.06 | 70.75 | 65.57 | 2092 |
| BD2 | 90.95 | 90.03 | 91.90 | 2066 |
| BMF | 98.84 | 98.62 | 99.06 | 2025 |
| DPL | 95.50 | 96.17 | 94.84 | 2008 |
| DS1 | 96.06 | 95.43 | 96.69 | 2145 |
| FFC | 94.38 | 95.66 | 93.15 | 2003 |
| MGS | 88.78 | 94.40 | 83.79 | 1999 |
| OD1 | 74.91 | 74.32 | 75.51 | 2037 |
| RAT | 95.04 | 93.93 | 96.17 | 2061 |
| RBM | 98.99 | 98.92 | 99.06 | 2031 |
| SD1 | 68.59 | 66.98 | 70.28 | 2002 |
| TS9 | 63.11 | 57.99 | 69.22 | 1947 |
| VTB | 98.18 | 99.95 | 96.47 | 1997 |

*(Consulte a matriz de confusão `confusion_matrix.png` gerada na pasta da run para a análise matricial e de falsos positivos completos que constam no JSON.)*
