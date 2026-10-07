# Transfer Learning AST: Mono Discrete

> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_ast_results.py`.

Resultados baseados no experimento executado em: `probe_20261001_225323`

## 1. Configurações de Treinamento

- **Learning Rate:** 0.001
- **Máximo de Épocas:** 200
- **Paciência (Early Stopping):** 15
- **Melhor Época Alcançada:** 94

## 2. Desempenho no Teste

| Métrica | Resultado |
| :--- | ---: |
| **Acurácia (Média Multi-seed)** | 86.22% ± 0.18% |
| **Acurácia (Melhor Época)** | 86.07% |
| **F1 Macro** | 83.81% |
| **Precisão Macro** | 85.41% |
| **Recall Macro** | 83.49% |

## 3. Desempenho Detalhado por Classe

| Efeito | F1-Score (%) | Recall (%) | Precisão (%) | Suporte |
| :--- | ---: | ---: | ---: | ---: |
| 808 | 67.46 | 70.44 | 64.73 | 2520 |
| BD2 | 94.13 | 91.90 | 96.46 | 2520 |
| BMF | 98.39 | 99.21 | 97.58 | 2520 |
| DPL | 84.84 | 82.14 | 87.71 | 504 |
| DS1 | 92.89 | 92.34 | 93.45 | 2520 |
| FFC | 92.12 | 95.56 | 88.92 | 630 |
| MGS | 88.15 | 92.70 | 84.03 | 2520 |
| OD1 | 38.89 | 27.78 | 64.81 | 504 |
| RAT | 95.29 | 95.52 | 95.06 | 2520 |
| RBM | 99.35 | 99.48 | 99.21 | 2520 |
| SD1 | 76.97 | 78.93 | 75.11 | 2520 |
| TS9 | 64.61 | 61.11 | 68.54 | 2520 |
| VTB | 96.42 | 98.25 | 94.65 | 630 |

*(Consulte a matriz de confusão `confusion_matrix.png` gerada na pasta da run para a análise matricial e de falsos positivos completos que constam no JSON.)*
