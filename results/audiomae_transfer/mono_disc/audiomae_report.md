# Transfer Learning AudioMAE: Mono Discrete

> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_audiomae_results.py`.

Resultados baseados no experimento executado em: `probe_20261003_174831`

## 1. Configurações de Treinamento

- **Learning Rate:** 0.001
- **Máximo de Épocas:** 200
- **Paciência (Early Stopping):** 15
- **Melhor Época Alcançada:** 137

## 2. Desempenho no Teste

| Métrica | Resultado |
| :--- | ---: |
| **Acurácia (Média Multi-seed)** | 78.35% ± 0.10% |
| **Acurácia (Melhor Época)** | 78.35% |
| **F1 Macro** | 73.37% |
| **Precisão Macro** | 75.77% |
| **Recall Macro** | 73.22% |

## 3. Desempenho Detalhado por Classe

| Efeito | F1-Score (%) | Recall (%) | Precisão (%) | Suporte |
| :--- | ---: | ---: | ---: | ---: |
| 808 | 59.17 | 61.15 | 57.31 | 2520 |
| BD2 | 86.55 | 85.91 | 87.19 | 2520 |
| BMF | 94.97 | 95.08 | 94.85 | 2520 |
| DPL | 50.42 | 41.87 | 63.36 | 504 |
| DS1 | 88.97 | 89.48 | 88.47 | 2520 |
| FFC | 77.70 | 83.49 | 72.65 | 630 |
| MGS | 78.65 | 85.40 | 72.90 | 2520 |
| OD1 | 25.00 | 16.07 | 56.25 | 504 |
| RAT | 89.40 | 92.34 | 86.63 | 2520 |
| RBM | 95.14 | 95.95 | 94.34 | 2520 |
| SD1 | 65.12 | 63.29 | 67.04 | 2520 |
| TS9 | 55.07 | 52.26 | 58.20 | 2520 |
| VTB | 87.65 | 89.52 | 85.84 | 630 |

*(Consulte a matriz de confusão `confusion_matrix.png` gerada na pasta da run para a análise matricial e de falsos positivos completos que constam no JSON.)*
