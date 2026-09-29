# AudioMAE on GUITAR-FX vs FXNet Reports

## Run

AudioMAE checkpoint:

```text
C:\TCC\AudioMAE\ckpt\finetuned.pth
```

Input data:

```text
C:\TCC\Reproduzir_artigo\data\GUITAR-FX
```

The run used the same broad evaluation subset as the FXNet reports: `MT2` was excluded and `NoFX` was not included. This gives 13 pedal-effect classes.

| Dataset | WAVs | Runtime |
|---|---:|---:|
| Mono_Continuous | 130000 | 354.04 s |
| Mono_Discrete | 123552 | 347.93 s |
| Poly_Continuous | 130000 | 358.31 s |
| Poly_Discrete | 83160 | 233.63 s |

## Why the comparison is not direct

The FXNet reports evaluate a supervised GUITAR-FX pedal classifier. Its labels are pedal IDs such as `808`, `BD2`, `DS1`, `RAT`, `VTB`, etc.

The AudioMAE checkpoint evaluated here is the paper's AudioSet fine-tuned model. It predicts 527 AudioSet classes such as `Music`, `Guitar`, `Distortion`, `Effects unit`, `Synthesizer`, and `Air horn, truck horn`. It does not have output classes for individual GUITAR-FX pedals.

Therefore FXNet accuracy/loss cannot be compared numerically against AudioMAE top-1 accuracy on GUITAR-FX, because AudioMAE has no way to output the correct GUITAR-FX label. The useful comparison is qualitative: what broad AudioSet labels does the paper model assign to GUITAR-FX audio?

## AudioMAE results

| Dataset | WAVs | Top-1 AudioSet class | Top-1 fraction | Music top-1 | Distortion top-1 | Distortion top-5 | Guitar top-5 | Effects unit top-5 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Mono_Continuous | 130000 | Music | 66.09% | 66.09% | 0.00% | 11.19% | 11.85% | 15.80% |
| Mono_Discrete | 123552 | Music | 68.49% | 68.49% | 0.00% | 11.83% | 13.48% | 16.87% |
| Poly_Continuous | 130000 | Music | 93.76% | 93.76% | 0.12% | 47.73% | 52.99% | 42.38% |
| Poly_Discrete | 83160 | Music | 91.11% | 91.11% | 0.17% | 44.59% | 48.04% | 41.56% |

Top-1 AudioSet predictions were dominated by broad classes:

| Dataset | Main top-1 predictions |
|---|---|
| Mono_Continuous | Music 66.09%, Air horn, truck horn 24.20%, Synthesizer 6.71% |
| Mono_Discrete | Music 68.49%, Air horn, truck horn 19.16%, Synthesizer 7.59% |
| Poly_Continuous | Music 93.76%, Vehicle 2.91%, Cacophony 1.16% |
| Poly_Discrete | Music 91.11%, Vehicle 4.56%, Cacophony 1.27% |

## FXNet report values

From:

```text
C:\TCC\Reproduzir_artigo\models_and_results\results\_model_eval\20201024_fxnet_mono_cont_best_all_datasets_report.md
C:\TCC\Reproduzir_artigo\models_and_results\results\_model_eval\20201025_fxnet_poly_cont_best_all_datasets_report.md
```

| Dataset | FXNet mono-cont best accuracy | FXNet poly-cont best accuracy |
|---|---:|---:|
| Mono_Continuous | 91.73% | 9.22% |
| Mono_Discrete | 81.31% | 11.31% |
| Poly_Continuous | 11.65% | 92.16% |
| Poly_Discrete | 14.15% | 83.78% |

## Interpretation

FXNet is task-specific. The mono model works well on mono guitar effects and fails on polyphonic data. The poly model shows the opposite pattern. That is expected because the reports evaluate a supervised model trained for those GUITAR-FX labels and data regimes.

AudioMAE is not task-specific here. It was fine-tuned on AudioSet, not on GUITAR-FX. It identifies broad audio concepts, not pedal identities. On polyphonic samples it more often places relevant AudioSet classes such as `Guitar`, `Distortion`, and `Effects unit` in the top-5. On monophonic isolated notes it more often confuses the signal with AudioSet classes such as `Air horn, truck horn` or `Synthesizer`.

The biggest causes of the difference are:

- Label space mismatch: FXNet predicts pedal IDs; AudioMAE predicts AudioSet classes.
- Training data mismatch: FXNet was trained/evaluated on GUITAR-FX; this AudioMAE checkpoint was fine-tuned on YouTube AudioSet clips.
- Input distribution mismatch: GUITAR-FX clips are 2-second controlled guitar samples; AudioMAE's AudioSet setup expects 10-second AudioSet-style clips.
- Taxonomy mismatch: AudioSet has broad labels like `Distortion` and `Effects unit`, but no classes for `808`, `TS9`, `BD2`, `RAT`, etc.

## Output files

Per-dataset outputs are under:

```text
C:\TCC\AudioMAE\outputs\guitarfx_audiomae_all_datasets_fxnet_comparison
```

Each dataset folder contains:

- `predictions_top5.csv`: per-file AudioSet top-5 predictions.
- `overall_top1_counts.csv`: top-1 AudioSet class distribution.
- `overall_top5_counts.csv`: top-5 AudioSet class distribution.
- `top1_by_guitarfx_label.csv`: top-1 AudioSet distribution grouped by GUITAR-FX pedal label.
- `interesting_audioset_classes_by_guitarfx_label.csv`: probabilities and top-5 rates for `Music`, `Guitar`, `Distortion`, `Effects unit`, etc.
