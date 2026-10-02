# Transfer Learning AST: Poly Continuous

> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_ast_results.py`.

Resultados baseados no experimento executado em: `probe_20261002_165042`

## 1. Configurações de Treinamento

- **Learning Rate:** 0.001
- **Máximo de Épocas:** 200
- **Paciência (Early Stopping):** 15
- **Melhor Época Alcançada:** 123

## 2. Desempenho no Teste

| Métrica | Resultado |
| :--- | ---: |
| **Acurácia (Média Multi-seed)** | 90.26% ± 0.26% |
| **Acurácia (Melhor Época)** | 90.33% |
| **F1 Macro** | 90.31% |
| **Precisão Macro** | 90.31% |
| **Recall Macro** | 90.34% |

## 3. Desempenho Detalhado por Classe

| Efeito | F1-Score (%) | Recall (%) | Precisão (%) | Suporte |
| :--- | ---: | ---: | ---: | ---: |
| 808 | 62.85 | 64.51 | 61.26 | 2074 |
| BD2 | 98.20 | 97.29 | 99.13 | 1995 |
| BMF | 99.73 | 99.70 | 99.75 | 2023 |
| DPL | 98.01 | 98.27 | 97.74 | 2025 |
| DS1 | 98.37 | 98.37 | 98.37 | 2090 |
| FFC | 98.86 | 99.36 | 98.35 | 2042 |
| MGS | 93.21 | 93.25 | 93.16 | 2075 |
| OD1 | 86.86 | 88.77 | 85.02 | 2021 |
| RAT | 97.54 | 97.20 | 97.88 | 2039 |
| RBM | 99.93 | 99.90 | 99.95 | 2023 |
| SD1 | 81.45 | 81.07 | 81.83 | 2039 |
| TS9 | 59.21 | 56.77 | 61.87 | 2024 |
| VTB | 99.81 | 99.95 | 99.66 | 2064 |

*(Consulte a matriz de confusão `confusion_matrix.png` gerada na pasta da run para a análise matricial e de falsos positivos completos que constam no JSON.)*
