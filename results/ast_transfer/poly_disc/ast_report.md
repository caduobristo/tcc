# Transfer Learning AST: Poly Discrete

> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_ast_results.py`.

Resultados baseados no experimento executado em: `probe_20261002_135006`

## 1. Configurações de Treinamento

- **Learning Rate:** 0.001
- **Máximo de Épocas:** 200
- **Paciência (Early Stopping):** 15
- **Melhor Época Alcançada:** 68

## 2. Desempenho no Teste

| Métrica | Resultado |
| :--- | ---: |
| **Acurácia (Média Multi-seed)** | 89.03% ± 0.37% |
| **Acurácia (Melhor Época)** | 88.37% |
| **F1 Macro** | 87.94% |
| **Precisão Macro** | 88.55% |
| **Recall Macro** | 87.54% |

## 3. Desempenho Detalhado por Classe

| Efeito | F1-Score (%) | Recall (%) | Precisão (%) | Suporte |
| :--- | ---: | ---: | ---: | ---: |
| 808 | 64.22 | 65.47 | 63.02 | 1700 |
| BD2 | 97.71 | 97.82 | 97.59 | 1700 |
| BMF | 99.56 | 99.59 | 99.53 | 1700 |
| DPL | 90.12 | 88.53 | 91.77 | 340 |
| DS1 | 96.81 | 97.18 | 96.44 | 1700 |
| FFC | 96.85 | 97.65 | 96.06 | 425 |
| MGS | 92.15 | 94.88 | 89.56 | 1700 |
| OD1 | 62.90 | 55.59 | 72.41 | 340 |
| RAT | 97.45 | 96.65 | 98.27 | 1700 |
| RBM | 99.65 | 99.59 | 99.71 | 1700 |
| SD1 | 84.53 | 85.47 | 83.60 | 1700 |
| TS9 | 62.17 | 60.12 | 64.36 | 1700 |
| VTB | 99.18 | 99.53 | 98.83 | 425 |

*(Consulte a matriz de confusão `confusion_matrix.png` gerada na pasta da run para a análise matricial e de falsos positivos completos que constam no JSON.)*
