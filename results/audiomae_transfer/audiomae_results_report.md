# Resumo Consolidado de Transfer Learning: AudioMAE

> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_audiomae_results.py`.

O modelo AudioMAE (Audio Masked Autoencoder) foi utilizado como extrator de características congelado (*frozen backbone*). Foi testado o modelo com Fine-Tuning supervisionado no AudioSet (Link 2).

## Desempenho por Cenário

| Cenário | Acurácia Média ± DP (%) | F1 Macro (%) | Detalhes |
| :--- | ---: | ---: | :--- |
| Mono Discrete | 78.35 ± 0.10 | 73.37 | [Relatório Completo](mono_disc/audiomae_report.md) |
| Mono Continuous | 77.64 ± 0.32 | 77.13 | [Relatório Completo](mono_cont/audiomae_report.md) |
| Poly Discrete | 74.84 ± 0.31 | 70.70 | [Relatório Completo](poly_disc/audiomae_report.md) |
| Poly Continuous | 77.68 ± 0.50 | 77.10 | [Relatório Completo](poly_cont/audiomae_report.md) |

## Observações Principais do AudioMAE
- **A Maldição do Linear Probing em Modelos MAE:** Mesmo com Fine-Tuning no AudioSet, o AudioMAE apresentou desempenho (~77%) consideravelmente inferior ao AST (~90%). O espaço latente de modelos Masked Autoencoder é sabidamente não-linear e tem dificuldade com *Linear Probing* estrito. A literatura aponta que a verdadeira força do MAE se mostra no *Full Fine-Tuning*.
- **O Gargalo da Resolução e Stride:** O AudioMAE extrai *patches* sem sobreposição (*stride* 16), gerando 512 tokens, diferentemente do AST (*stride* 10), que gera 1212 tokens com alta sobreposição. Isso compromete seriamente a resolução espectral e temporal microscópica da rede.
- **Extremo vs. Sutil:** As matrizes de confusão mostram que o modelo continua performando excelentemente em distorções extremas e óbvias (BMF 95%, RBM 94%, DS1 92%), mas sofre colapso completo para untar overdrives sutis (TS9 49%, SD1 49%), provando sua incapacidade de resolver timbres microscopicamente similares sem sobreposição espacial de *patches* e sem interpolação temporal.
