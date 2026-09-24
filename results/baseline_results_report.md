# FxNet Baseline Training - Benchmark Results Report

This report is automatically fed and updated at the end of each training scenario.

## Overall Performance Comparison

| Dataset Scenario | Test Acc (%) | Paper Baseline (%) | Diff (%) | Best Val Acc (%) | Macro F1 (%) | Final Train Loss | Final Val Loss | Epochs | Time (min) | Run Date |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mono Discrete | 86.61% | 86.30% | +0.31% | 87.22% | 83.31% | 0.174 | 0.2148 | 30 | 9.45 | 2026-09-24 20:02 |
| Mono Continuous | 90.02% | 90.90% | -0.88% | 90.64% | 90.03% | 0.1429 | 0.2768 | 30 | 10.9 | 2026-09-24 20:23 |
| Poly Discrete | 87.73% | 88.40% | -0.67% | 88.35% | 88.90% | 0.1581 | 2.0247 | 30 | 6.49 | 2026-09-24 20:30 |
| Poly Continuous | 89.83% | 91.40% | -1.57% | 90.66% | 88.19% | 0.1171 | 1.096 | 30 | 10.16 | 2026-09-24 20:42 |

## Detailed Scenario Artifacts

### Mono Discrete
- **Test Accuracy**: 86.61%
- **Paper Benchmark (Comunità et al.)**: 86.3%
- **Macro F1-Score**: 83.31%
- **Macro Precision / Recall**: 82.54% / 85.56%
- **Best Validation Accuracy**: 87.22%
- **Final Losses (Train / Val)**: 0.174 / 0.2148
- **Total Training Time**: 9.45 minutes (30 epochs)
- **Checkpoint**: `checkpoints/best_model_mono_disc.pt`
- **Confusion Matrix**: `results/mono_disc/confusion_matrix_mono_disc.png`
- **Training Curves (Loss & Acc)**: `results/mono_disc/training_curves_mono_disc.png`
- **Classification Report**: `results/mono_disc/classification_report_mono_disc.txt`

### Mono Continuous
- **Test Accuracy**: 90.02%
- **Paper Benchmark (Comunità et al.)**: 90.9%
- **Macro F1-Score**: 90.03%
- **Macro Precision / Recall**: 90.25% / 90.24%
- **Best Validation Accuracy**: 90.64%
- **Final Losses (Train / Val)**: 0.1429 / 0.2768
- **Total Training Time**: 10.9 minutes (30 epochs)
- **Checkpoint**: `checkpoints/best_model_mono_cont.pt`
- **Confusion Matrix**: `results/mono_cont/confusion_matrix_mono_cont.png`
- **Training Curves (Loss & Acc)**: `results/mono_cont/training_curves_mono_cont.png`
- **Classification Report**: `results/mono_cont/classification_report_mono_cont.txt`

### Poly Discrete
- **Test Accuracy**: 87.73%
- **Paper Benchmark (Comunità et al.)**: 88.4%
- **Macro F1-Score**: 88.9%
- **Macro Precision / Recall**: 89.72% / 88.27%
- **Best Validation Accuracy**: 88.35%
- **Final Losses (Train / Val)**: 0.1581 / 2.0247
- **Total Training Time**: 6.49 minutes (30 epochs)
- **Checkpoint**: `checkpoints/best_model_poly_disc.pt`
- **Confusion Matrix**: `results/poly_disc/confusion_matrix_poly_disc.png`
- **Training Curves (Loss & Acc)**: `results/poly_disc/training_curves_poly_disc.png`
- **Classification Report**: `results/poly_disc/classification_report_poly_disc.txt`

### Poly Continuous
- **Test Accuracy**: 89.83%
- **Paper Benchmark (Comunità et al.)**: 91.4%
- **Macro F1-Score**: 88.19%
- **Macro Precision / Recall**: 90.05% / 90.11%
- **Best Validation Accuracy**: 90.66%
- **Final Losses (Train / Val)**: 0.1171 / 1.096
- **Total Training Time**: 10.16 minutes (30 epochs)
- **Checkpoint**: `checkpoints/best_model_poly_cont.pt`
- **Confusion Matrix**: `results/poly_cont/confusion_matrix_poly_cont.png`
- **Training Curves (Loss & Acc)**: `results/poly_cont/training_curves_poly_cont.png`
- **Classification Report**: `results/poly_cont/classification_report_poly_cont.txt`

