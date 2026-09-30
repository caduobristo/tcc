# Evaluation Report: 20201025_fxnet_poly_cont_best

- Split: `full`
- Source dir: `C:\TCC\Reproduzir_artigo\models_and_results\results\_model_eval`

## Summary

| Dataset | Samples | Accuracy | Avg Loss | Macro P | Macro R | Macro F1 | Weighted F1 | Balanced Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mono_Continuous | 130000 | 9.22% | 35.951940 | 11.71% | 9.22% | 4.31% | 4.31% | 9.22% |
| Mono_Discrete | 123552 | 11.31% | 27.950966 | 11.30% | 8.75% | 4.76% | 6.16% | 8.75% |
| Poly_Continuous | 130000 | 92.16% | 0.121675 | 92.41% | 92.16% | 91.49% | 91.49% | 92.16% |
| Poly_Discrete | 83160 | 83.78% | 0.761938 | 86.64% | 87.48% | 85.81% | 82.72% | 87.48% |

## Mono_Continuous

- Samples: `130000`
- Accuracy: `9.22%`
- Avg loss: `35.951940`
- Macro precision: `11.71%`
- Macro recall: `9.22%`
- Macro F1: `4.31%`
- Weighted precision: `11.71%`
- Weighted recall: `9.22%`
- Weighted F1: `4.31%`
- Balanced accuracy: `9.22%`
- Predictions CSV: `20201025_fxnet_poly_cont_best_full_20260407_193857_predictions.csv`
- Per-class CSV: `20201025_fxnet_poly_cont_best_full_20260407_193857_per_class.csv`
- Confusion matrix CSV: `20201025_fxnet_poly_cont_best_full_20260407_193857_confusion_matrix.csv`

| Main confusion | Count |
|---|---:|
| DPL -> DS1 | 8503 |
| RBM -> DS1 | 8225 |
| MGS -> DS1 | 8189 |
| BD2 -> DS1 | 7632 |
| RAT -> DS1 | 7580 |
| FFC -> DS1 | 7341 |
| OD1 -> DS1 | 7185 |
| SD1 -> DS1 | 6678 |

## Mono_Discrete

- Samples: `123552`
- Accuracy: `11.31%`
- Avg loss: `27.950966`
- Macro precision: `11.30%`
- Macro recall: `8.75%`
- Macro F1: `4.76%`
- Weighted precision: `13.52%`
- Weighted recall: `11.31%`
- Weighted F1: `6.16%`
- Balanced accuracy: `8.75%`
- Predictions CSV: `20201025_fxnet_poly_cont_best_full_20260407_194754_predictions.csv`
- Per-class CSV: `20201025_fxnet_poly_cont_best_full_20260407_194754_per_class.csv`
- Confusion matrix CSV: `20201025_fxnet_poly_cont_best_full_20260407_194754_confusion_matrix.csv`

| Main confusion | Count |
|---|---:|
| MGS -> DS1 | 9951 |
| RBM -> DS1 | 9579 |
| RAT -> DS1 | 9511 |
| SD1 -> DS1 | 8011 |
| BMF -> DS1 | 6466 |
| TS9 -> DS1 | 5300 |
| 808 -> DS1 | 5292 |
| 808 -> SD1 | 5130 |

## Poly_Continuous

- Samples: `130000`
- Accuracy: `92.16%`
- Avg loss: `0.121675`
- Macro precision: `92.41%`
- Macro recall: `92.16%`
- Macro F1: `91.49%`
- Weighted precision: `92.41%`
- Weighted recall: `92.16%`
- Weighted F1: `91.49%`
- Balanced accuracy: `92.16%`
- Predictions CSV: `20201025_fxnet_poly_cont_best_full_20260407_195712_predictions.csv`
- Per-class CSV: `20201025_fxnet_poly_cont_best_full_20260407_195712_per_class.csv`
- Confusion matrix CSV: `20201025_fxnet_poly_cont_best_full_20260407_195712_confusion_matrix.csv`

| Main confusion | Count |
|---|---:|
| TS9 -> 808 | 7612 |
| 808 -> TS9 | 1806 |
| SD1 -> OD1 | 485 |
| BMF -> DS1 | 101 |
| OD1 -> SD1 | 50 |
| OD1 -> MGS | 29 |
| SD1 -> 808 | 23 |
| SD1 -> MGS | 21 |

## Poly_Discrete

- Samples: `83160`
- Accuracy: `83.78%`
- Avg loss: `0.761938`
- Macro precision: `86.64%`
- Macro recall: `87.48%`
- Macro F1: `85.81%`
- Weighted precision: `84.73%`
- Weighted recall: `83.78%`
- Weighted F1: `82.72%`
- Balanced accuracy: `87.48%`
- Predictions CSV: `20201025_fxnet_poly_cont_best_full_20260407_200307_predictions.csv`
- Per-class CSV: `20201025_fxnet_poly_cont_best_full_20260407_200307_per_class.csv`
- Confusion matrix CSV: `20201025_fxnet_poly_cont_best_full_20260407_200307_confusion_matrix.csv`

| Main confusion | Count |
|---|---:|
| TS9 -> 808 | 6258 |
| BD2 -> BMF | 2164 |
| 808 -> TS9 | 2138 |
| BD2 -> DS1 | 789 |
| BD2 -> SD1 | 617 |
| SD1 -> OD1 | 446 |
| BD2 -> 808 | 195 |
| BMF -> DS1 | 126 |
