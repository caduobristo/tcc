# Resumo Consolidado de Transfer Learning: AST

> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_ast_results.py`.

O modelo AST (Audio Spectrogram Transformer) foi utilizado como extrator de características congelado (*frozen backbone*), com o treinamento apenas de uma nova camada linear.

## Desempenho por Cenário

| Cenário | Acurácia Média ± DP (%) | F1 Macro (%) | Detalhes |
| :--- | ---: | ---: | :--- |
| Mono Discrete | 86.22 ± 0.18 | 83.81 | [Relatório Completo](mono_disc/ast_report.md) |
| Mono Continuous | 87.22 ± 0.27 | 87.03 | [Relatório Completo](mono_cont/ast_report.md) |
| Poly Discrete | 89.03 ± 0.37 | 87.94 | [Relatório Completo](poly_disc/ast_report.md) |
| Poly Continuous | 90.26 ± 0.26 | 90.31 | [Relatório Completo](poly_cont/ast_report.md) |

## Observações Principais do AST
- **Riqueza Acústica vs. Desempenho:** Os cenários contínuos e polifônicos geraram os melhores desempenhos absolutos (Poly Continuous bateu >90% F1), indicando que acordes densos e mudanças contínuas de sinal alimentam melhor o Transformer.
- **Gargalo de Overdrives:** Ao analisar as tabelas detalhadas de cada cenário, nota-se que OD1 e TS9 continuam puxando as métricas macro para baixo, refletindo a altíssima similaridade desses circuitos de clipagem.
