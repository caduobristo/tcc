"""Publish confusion figures from completed predictions, without training models."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'results/passt_htsat_transfer'
SCENARIOS = {'mono_disc': 'Mono Discrete', 'mono_cont': 'Mono Continuous',
             'poly_disc': 'Poly Discrete', 'poly_cont': 'Poly Continuous'}
CLASSES = ('808', 'BD2', 'BMF', 'DPL', 'DS1', 'FFC', 'MGS', 'OD1', 'RAT', 'RBM', 'SD1', 'TS9', 'VTB')
SEEDS = (101, 202, 303, 404, 505)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(cm):
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) != 0)
    recall = tp / cm.sum(1)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=precision+recall != 0)
    return {'accuracy': float(tp.sum()/cm.sum()), 'macro_f1': float(f1.mean()),
            'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean())}


def load_completed(scenario, variant):
    stem = ('transfer_learning_consolidated' if scenario == 'mono_disc' else f'transfer_learning_{scenario}_consolidated')
    if variant == 'no_ts9':
        stem = f'transfer_learning_no_ts9_{scenario}_consolidated'
    source = PUBLIC / f'{stem}_summary.json'
    public = read(source)
    local = read(ROOT / public['session'] / 'summary.json')
    assert public['all_predictions_reproduced'] and public['encoder_frozen'] and not public['fine_tuning']
    classes = CLASSES if variant == '13_classes' else tuple(c for c in CLASSES if c != 'TS9')
    assert tuple(public.get('classes', CLASSES)) == classes
    results = {}
    for model in ('passt', 'htsat'):
        records = local['final_runs'][model]
        assert tuple(r['training_seed'] for r in records) == SEEDS
        matrices, sources = [], []
        expected_truth, expected_indices = None, None
        for record in records:
            folder = ROOT / record['folder']
            run = read(folder / 'run.json')
            assert run['status'] == 'complete' and run['encoder_frozen'] and run['test_evaluated']
            assert tuple(run.get('classes', CLASSES)) == classes
            assert digest(folder / 'best_head.pt') == record['head_sha256']
            assert digest(folder / 'preprocessing.pt') == record['preprocessing_sha256']
            file = folder / 'test_predictions.npz'
            with np.load(file, allow_pickle=False) as saved:
                truth, prediction, indices = saved['true'], saved['predicted'], saved['indices']
                n = len(classes)
                assert truth.shape == prediction.shape == indices.shape and len(truth) == public['split_counts']['test']
                assert np.all((truth >= 0) & (truth < n)) and np.all((prediction >= 0) & (prediction < n))
                if expected_truth is None:
                    expected_truth, expected_indices = truth.copy(), indices.copy()
                assert np.array_equal(expected_truth, truth) and np.array_equal(expected_indices, indices)
                cm = np.bincount(truth*n+prediction, minlength=n*n).reshape(n, n)
            assert cm.tolist() == record['test_metrics']['confusion_matrix']
            computed = metrics(cm)
            for key, value in computed.items():
                assert abs(value-record['test_metrics'][key]) < 1e-12
            matrices.append(cm)
            sources.append({'seed': record['training_seed'], 'predictions_sha256': digest(file)})
        matrices = np.asarray(matrices)
        supports = matrices.sum(2)
        assert np.all(supports > 0) and np.all(supports == supports[0])
        rates = (matrices / supports[:, :, None]).mean(0) * 100
        assert np.allclose(rates.sum(1), 100)
        for key, reported in public['models'][model]['aggregate'].items():
            values = [metrics(cm)[key] for cm in matrices]
            assert np.allclose(values, reported['values'], atol=1e-12, rtol=0)
            assert abs(np.mean(values)-reported['mean']) < 1e-12
            assert abs(np.std(values, ddof=1)-reported['std']) < 1e-12
        results[model] = {'classes': list(classes), 'mean_row_percent': rates.tolist(),
                          'support_per_seed': supports[0].tolist(), 'unique_test_samples': int(matrices[0].sum()),
                          'aggregate_metrics': public['models'][model]['aggregate'],
                          'source_summary': source.relative_to(ROOT).as_posix(), 'source_summary_sha256': digest(source),
                          'prediction_sources': sources}
    return results


def figure(info, model, scenario, variant, destination):
    values = np.asarray(info['mean_row_percent'])
    classes = info['classes']
    label = 'PaSST' if model == 'passt' else 'HTS-AT'
    count = '13 classes' if variant == '13_classes' else '12 classes, sem TS9'
    agg = info['aggregate_metrics']
    fig, ax = plt.subplots(figsize=(12.5, 10.3), layout='constrained')
    im = ax.imshow(values, cmap='Blues', vmin=0, vmax=100, interpolation='nearest')
    positions = np.arange(len(classes))
    ax.set(xticks=positions, yticks=positions, xticklabels=classes, yticklabels=classes,
           xlabel='Classe prevista', ylabel='Classe verdadeira')
    ax.tick_params(axis='both', labelsize=11)
    ax.set_title(f'{label} — {SCENARIOS[scenario]} — {count}\n'
                 f'Média de 5 seeds | Acurácia: {100*agg["accuracy"]["mean"]:.2f}% | '
                 f'F1 macro: {100*agg["macro_f1"]["mean"]:.2f}%', fontsize=15, pad=17)
    ax.set_xticks(np.arange(-.5, len(classes), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(classes), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.4)
    ax.tick_params(which='minor', bottom=False, left=False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{values[i,j]:.1f}'.replace('.', ','), ha='center', va='center', fontsize=9,
                    color='white' if values[i,j] >= 55 else '#17202a')
    colorbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.025)
    colorbar.set_label('Percentual médio dentro da classe verdadeira (%)', fontsize=11)
    fig.supxlabel(f'{info["unique_test_samples"]:,} WAVs de teste por seed; linhas normalizadas em 100%. '
                  'Diagonal = recall. Valores arredondados a 0,1%.', fontsize=10)
    fig.savefig(destination, dpi=180, metadata={'Software': 'Matplotlib; saved predictions only'})
    plt.close(fig)


def render(output):
    output.mkdir(parents=True, exist_ok=True)
    summary = {'schema': 1, 'seeds': list(SEEDS),
               'aggregation': 'mean of per-seed row-normalized confusion matrices, in percent',
               'std': 'sample std of per-seed metrics, ddof=1; fixed source split',
               'training_performed': False, 'raw_predictions_published': False, 'variants': {}}
    for variant in ('13_classes', 'no_ts9'):
        summary['variants'][variant] = {}
        for scenario in SCENARIOS:
            items = load_completed(scenario, variant)
            for model, info in items.items():
                name = f'{model}_{scenario}_{variant}.png'
                figure(info, model, scenario, variant, output/name)
                info.update(image=name, image_sha256=digest(output/name))
                print('CONFUSION_VERIFIED', model, scenario, variant, flush=True)
            summary['variants'][variant][scenario] = items
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# Matrizes de confusão: PaSST e HTS-AT', '',
             'Figuras científicas para a [issue 5](https://github.com/caduobristo/tcc/issues/5), geradas a partir das predições de teste já concluídas. Não houve extração, treinamento ou fine-tuning.', '',
             'Cada célula contém a média dos percentuais nas seeds 101, 202, 303, 404 e 505. As linhas representam as classes verdadeiras; as colunas, as classes previstas. Cada linha soma 100% antes do arredondamento; a diagonal corresponde ao recall médio da classe.', '',
             'As cinco seeds usam os mesmos WAVs de teste. Os suportes são registrados por seed: não representam cinco vezes mais exemplos independentes. Acurácia, F1, precisão e recall macro são calculados separadamente por seed e depois agregados; o F1 médio não é calculado a partir da matriz média.', '',
             'As matrizes inteiras foram reconstruídas com os arquivos de predições, comparadas com os registros originais e usadas para conferir todas as métricas agregadas. Hashes de heads, scalers, predições e resumos foram conferidos. `summary.json` registra percentuais, suportes por seed e fontes, sem publicar predições por WAV.', '',
             '## Figuras', '', '| Cenário | PaSST, 13 classes | HTS-AT, 13 classes | PaSST, sem TS9 | HTS-AT, sem TS9 |',
             '| --- | --- | --- | --- | --- |']
    for scenario, name in SCENARIOS.items():
        links = [f'[{model.upper()}]({model}_{scenario}_{variant}.png)' for variant in ('13_classes','no_ts9') for model in ('passt','htsat')]
        lines.append('| '+name+' | '+' | '.join(links)+' |')
    lines += ['', '## Limites e reprodução', '',
              'A ablação sem TS9 remove essa classe de treino, validação e teste, reutiliza os encoders/embeddings e refaz a seleção e o treinamento das cabeças. As partições dos demais WAVs são mantidas. Os resultados de 12 classes não substituem os de 13 classes; a análise é exploratória, com teste previamente consultado.', '',
              'Para regenerar as figuras na raiz, com os resultados locais completos: `python scripts/render_transfer_confusion_matrices.py`. Predições, caches, pesos, logs e documentos de contexto continuam locais. Apenas estas figuras selecionadas e seus metadados agregados são publicados.', '',
              'Relatórios: [13 classes](../transfer_learning_all_scenarios_report.md), [sem TS9](../transfer_learning_no_ts9_report.md).']
    (output/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PUBLIC/'confusion_matrices')
    args = parser.parse_args()
    render(args.output)
