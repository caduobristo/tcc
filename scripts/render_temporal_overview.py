"""Render aggregate scientific figures from saved metrics; never trains models."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/temporal_interpolation/summary.json'
OUTPUT = ROOT / 'results/temporal_interpolation/overview'
SCENARIOS = {'mono_disc': 'Mono Discrete', 'mono_cont': 'Mono Continuous',
             'poly_disc': 'Poly Discrete', 'poly_cont': 'Poly Continuous'}
MODELS = {'passt': 'PaSST', 'ast': 'AST', 'audiomae': 'AudioMAE'}
COLORS = {'passt': '#2563eb', 'ast': '#059669', 'audiomae': '#d97706'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_metrics(summary):
    if summary['classes'] != summary['protocol']['classes'] or len(summary['classes']) != 13:
        raise ValueError('Expected the verified thirteen-class experiment')
    if summary['fine_tuning'] or not summary['encoder_frozen']:
        raise ValueError('Only frozen transfer learning belongs in these figures')
    for scenario in SCENARIOS:
        for model in MODELS:
            pair = summary['scenarios'][scenario][model]
            for condition in ('control', 'bicubic'):
                runs = pair['conditions'][condition]['final_runs']
                if [r['training_seed'] for r in runs] != summary['final_seeds']:
                    raise ValueError('Final seeds changed')
                for metric, aggregate in pair['conditions'][condition]['aggregate'].items():
                    values = np.array([r['test_metrics'][metric] for r in runs])
                    np.testing.assert_allclose(values, aggregate['values'], rtol=0, atol=1e-12)
                    np.testing.assert_allclose([values.mean(), values.std(ddof=1)],
                        [aggregate['mean'], aggregate['std']], rtol=0, atol=1e-12)
            for metric, recorded in pair['paired_difference'].items():
                values = 100*(np.asarray(pair['conditions']['bicubic']['aggregate'][metric]['values']) -
                              np.asarray(pair['conditions']['control']['aggregate'][metric]['values']))
                np.testing.assert_allclose(values, recorded['values_pp'], rtol=0, atol=1e-12)
                np.testing.assert_allclose([values.mean(), values.std(ddof=1)],
                    [recorded['mean_pp'], recorded['std_pp']], rtol=0, atol=1e-12)


def comparison(summary):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True, layout='constrained')
    x, width = np.arange(4), .36
    for ax, model in zip(axes, MODELS):
        for condition, offset, color, label in (
                ('control', -width/2, '#94a3b8', 'Controle novo'),
                ('bicubic', width/2, COLORS[model], 'Alongamento bicúbico')):
            records = [summary['scenarios'][s][model]['conditions'][condition]['aggregate']['macro_f1'] for s in SCENARIOS]
            means = np.array([r['mean'] for r in records])*100
            errors = np.array([r['std'] for r in records])*100
            bars = ax.bar(x+offset, means, width, color=color, yerr=errors,
                          capsize=3, error_kw={'elinewidth': 1}, label=label)
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x()+bar.get_width()/2, mean+1.2,
                        f'{mean:.2f}'.replace('.', ','), ha='center', va='bottom', fontsize=9)
        ax.set_title(MODELS[model], fontsize=15, weight='bold')
        ax.set_xticks(x, ['Mono\nDiscrete', 'Mono\nContinuous', 'Poly\nDiscrete', 'Poly\nContinuous'])
        ax.set_ylim(0, 105)
        ax.grid(axis='y', color='#e2e8f0', linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='lower left', fontsize=9)
    axes[0].set_ylabel('F1 macro no teste (%)')
    fig.suptitle('Alongamento temporal: controles novos versus variante\n13 classes incluindo TS9 | encoders congelados', fontsize=17)
    fig.supxlabel('Média ± DP amostral de 5 seeds no mesmo split por fonte; barras de erro não representam variação entre partições.', fontsize=10)
    fig.savefig(OUTPUT/'macro_f1_comparison.png', dpi=180)
    plt.close(fig)


def gains(summary):
    fig, ax = plt.subplots(figsize=(12.5, 6.5), layout='constrained')
    x, width = np.arange(4), .24
    for offset, model in enumerate(MODELS):
        records = [summary['scenarios'][s][model]['paired_difference']['macro_f1'] for s in SCENARIOS]
        means, errors = [r['mean_pp'] for r in records], [r['std_pp'] for r in records]
        bars = ax.bar(x+(offset-1)*width, means, width, color=COLORS[model],
                      yerr=errors, capsize=4, label=MODELS[model])
        for bar, mean, error in zip(bars, means, errors):
            ax.text(bar.get_x()+bar.get_width()/2, mean+error+.25,
                    f'+{mean:.2f}'.replace('.', ','), ha='center', fontsize=10)
    ax.set_xticks(x, SCENARIOS.values())
    ax.set_ylim(0, 16)
    ax.set_ylabel('Δ F1 macro: alongamento − controle (pontos percentuais)')
    ax.set_title('Ganho de F1 macro nos quatro cenários\nMesmos dados, checkpoint e orçamento de seleção dentro de cada modelo', fontsize=15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e2e8f0', linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left')
    fig.supxlabel('Média ± DP das 5 diferenças pareadas por seed; comparação exploratória, sem teste de significância.', fontsize=10)
    fig.savefig(OUTPUT/'macro_f1_gain.png', dpi=180)
    plt.close(fig)


def adaptation_diagram():
    """Schematic dimensions only: publishes no dataset example or spectrogram."""
    fig, ax = plt.subplots(figsize=(15.5, 9), layout='constrained')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.text(8, 8.65, 'Onde ocorre a mudança?', ha='center', fontsize=21, weight='bold')
    ax.text(8, 8.16, 'Frontend/normalização → adaptação temporal → encoder congelado → embedding 768 → cabeça de 13 classes',
            ha='center', fontsize=11)
    def box(x, y, width, text, color):
        ax.add_patch(Rectangle((x, y), width, .55, facecolor=color, edgecolor='#334155', linewidth=1))
        ax.text(x+width/2, y+.275, text, ha='center', va='center', fontsize=10, color='white' if color=='#2563eb' else '#172033')
    ax.text(.2, 7.30, 'AST / AudioMAE', fontsize=16, weight='bold')
    ax.text(.2, 6.94, 'mel16 normalizado: 198 quadros × 128 bandas', fontsize=11)
    ax.text(.2, 6.33, 'Controle', fontsize=11, va='center')
    box(2.1, 6.05, 1.89, '198', '#2563eb')
    box(3.99, 6.05, 7.89, '826 quadros de padding (valor 0 após normalização)', '#e2e8f0')
    ax.text(12.2, 6.33, '1024 × 128', fontsize=11, va='center')
    ax.text(.2, 5.27, 'Variante', fontsize=11, va='center')
    box(2.1, 4.99, 9.78, '198 → 1024: valores interpolados ao longo de todo o eixo temporal', '#2563eb')
    ax.text(12.2, 5.27, '1024 × 128', fontsize=11, va='center')
    ax.text(.2, 3.93, 'PaSST', fontsize=16, weight='bold')
    ax.text(.2, 3.56, 'WAV 32 kHz → frontend nativo: aproximadamente 200 quadros × 128 bandas', fontsize=11)
    ax.text(.2, 2.95, 'Controle', fontsize=11, va='center')
    box(2.1, 2.67, 1.96, '~200', '#2563eb')
    ax.text(4.3, 2.95, 'Entrada curta nativa, sem preencher até 998', fontsize=11, va='center')
    ax.text(12.2, 2.95, '~200 × 128', fontsize=11, va='center')
    ax.text(.2, 1.89, 'Variante', fontsize=11, va='center')
    box(2.1, 1.61, 9.78, '~200 → 998: interpolação temporal antes dos patches', '#2563eb')
    ax.text(12.2, 1.89, '998 × 128', fontsize=11, va='center')
    ax.text(8, .95, 'Bicúbica • align_corners=True • somente o tempo • após normalização • 128 bandas preservadas', ha='center', fontsize=11)
    ax.text(8, .49, 'Esquema de dimensões: as faixas coloridas são ilustrativas, não dados do dataset.', ha='center', fontsize=10, color='#475569')
    ax.text(8, .10, 'Não alonga o WAV nem cria informação acústica; não aplica o dobramento de imagem específico do Swin/HTS-AT.', ha='center', fontsize=10, color='#475569')
    fig.savefig(OUTPUT/'temporal_adaptation_diagram.png', dpi=180)
    plt.close(fig)


def main():
    summary = json.loads(SOURCE.read_text(encoding='utf-8'))
    check_metrics(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison(summary)
    gains(summary)
    adaptation_diagram()
    metadata = {'schema': 1, 'experiment': summary['experiment'],
                'source': SOURCE.relative_to(ROOT).as_posix(), 'source_sha256': sha256(SOURCE),
                'final_seeds': summary['final_seeds'], 'classes': summary['classes'],
                'error_bars': 'sample SD over five seeds, or five paired differences; same source split',
                'adaptation_diagram': 'schematic dimensions only, no dataset image or raw data',
                'no_new_extraction_training_or_fine_tuning': True,
                'figures': {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size}
                            for p in sorted(OUTPUT.glob('*.png'))}}
    (OUTPUT/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('TEMPORAL_OVERVIEW_COMPLETE', OUTPUT)


if __name__ == '__main__':
    main()
