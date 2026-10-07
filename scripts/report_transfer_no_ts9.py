"""Compare verified 12-class ablations with the preserved 13-class experiments."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RESULTS = ROOT / 'results/passt_htsat_transfer'
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import torch

from src.data.transfer import EFFECTS, read_manifest, sha256
from src.training.consolidation import dataset_view, classes_for_plan, evaluate, write_json
from src.training.linear_probe import load_config, validate_cache, paths_for

SCENARIOS = ('mono_disc', 'mono_cont', 'poly_disc', 'poly_cont')
METRICS = ('accuracy', 'macro_f1', 'macro_precision', 'macro_recall')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def restricted_original_metrics(truth, prediction, classes):
    """TS9 predictions remain errors when scoring a 13-output head on 12 true classes."""
    labels = [EFFECTS.index(effect) for effect in classes]
    if not set(np.unique(truth)).issubset(labels):
        raise ValueError('The restricted test still contains an excluded true class')
    return {'accuracy': float(accuracy_score(truth, prediction)),
            'macro_f1': float(f1_score(truth, prediction, labels=labels, average='macro', zero_division=0)),
            'macro_precision': float(precision_score(truth, prediction, labels=labels, average='macro', zero_division=0)),
            'macro_recall': float(recall_score(truth, prediction, labels=labels, average='macro', zero_division=0)),
            'per_class': classification_report(truth, prediction, labels=labels, target_names=classes,
                                               output_dict=True, zero_division=0)}


def aggregate(records):
    return {key: {'mean': float(np.mean([r[key] for r in records])),
                  'std': float(np.std([r[key] for r in records], ddof=1)),
                  'values': [r[key] for r in records]} for key in METRICS}


def percent(value):
    return f'{100*value:.2f}'.replace('.', ',')


def mean_std(value):
    return f'{percent(value["mean"])} ± {percent(value["std"])}'


def report():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    public = {'schema': 1, 'excluded_effects': ['TS9'], 'encoder_frozen': True, 'fine_tuning': False,
              'comparison_scope': 'same source partitions and original encoder caches; freshly selected and trained 12-class heads',
              'test_previously_inspected': True, 'scenarios': {}}
    lines = ['# Transfer learning sem TS9: PaSST e HTS-AT', '',
             'Ablação exploratória em 02/10/2026. Retiramos somente TS9, mantendo os outros 12 efeitos, as partições por fonte e os encoders congelados. Cada cenário recebeu 34 treinamentos novos, com cinco seeds finais por modelo. Fine-tuning não foi realizado.', '',
             '## Resultados finais', '',
             'Média ± desvio padrão amostral entre as seeds 101, 202, 303, 404 e 505. A seleção de configuração/checkpoint usa somente F1 macro de validação.', '',
             '| Cenário | Modelo | Acurácia sem TS9 (%) | F1 macro sem TS9 (%) | Precisão macro (%) | Recall macro (%) |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    comparisons, retained_comparisons, details = [], [], []
    total_seconds, total_training = 0., 0.
    for scenario in SCENARIOS:
        old_stem = 'transfer_learning_consolidated' if scenario == 'mono_disc' else f'transfer_learning_{scenario}_consolidated'
        old = read_json(PUBLIC_RESULTS / f'{old_stem}_summary.json')
        new_stem = f'transfer_learning_no_ts9_{scenario}_consolidated'
        new = read_json(PUBLIC_RESULTS / f'{new_stem}_summary.json')
        if not new['all_predictions_reproduced'] or new['excluded_effects'] != ['TS9']:
            raise ValueError('Unverified ablation or unexpected excluded class')
        if old['manifest_sha256'] != new['manifest_sha256']:
            raise ValueError('Original source partitions changed')
        old_plan, new_plan = dict(old['protocol']), dict(new['protocol'])
        new_plan.pop('excluded_effects')
        new_plan['historical_test_already_inspected'] = old_plan['historical_test_already_inspected']
        if old_plan != new_plan:
            raise ValueError('The ablation changed the training/search budget')
        rows = read_manifest(scenario)
        classes = classes_for_plan(new['protocol'])
        retained, view = dataset_view(rows, classes)
        if view != new['dataset_view'] or list(classes) != new['classes']:
            raise ValueError('The reported 12-class dataset view changed')
        if len(new['verification']) != 10 or not all(check['exact_match'] for check in new['waveform_inference_check'].values()):
            raise ValueError('Incomplete final artifact or waveform verification')
        indices = np.asarray([i for i in retained if rows[i]['split'] == 'test'])
        original_indices = np.asarray([i for i, row in enumerate(rows) if row['split'] == 'test'])
        original_truth = np.asarray([int(rows[i]['label']) for i in original_indices])
        keep = np.isin(original_indices, indices)
        if not np.array_equal(original_indices[keep], indices):
            raise ValueError('Retained test WAV order changed')
        session_old = read_json(ROOT / old['session'] / 'summary.json')
        total_seconds += new['elapsed_seconds']
        item = {'without_ts9': new, 'original_13_class_report': f'results/passt_htsat_transfer/{old_stem}_report.md',
                'dataset_view': view, 'comparison': {}}
        label = scenario.replace('_', ' ')
        for model in ('passt', 'htsat'):
            model_name = 'PaSST' if model == 'passt' else 'HTS-AT'
            old_model, new_model = old['models'][model], new['models'][model]
            expected_seeds = set(new['protocol']['final_seeds'])
            if (len(new_model['final_runs']) != 5 or new_model['training_runs'] != 17
                    or {r['training_seed'] for r in new_model['final_runs']} != expected_seeds
                    or len(session_old['final_runs'][model]) != 5
                    or {r['training_seed'] for r in session_old['final_runs'][model]} != expected_seeds):
                raise ValueError('The original and ablation confirmation seed sets differ')
            if old_model['embedding_sha256'] != new_model['embedding_sha256']:
                raise ValueError('The frozen representation changed')
            cache = validate_cache(load_config(ROOT / f'configs/linear_probe/{model}_{scenario}.json'))
            _, vectors, _ = paths_for(load_config(ROOT / f'configs/linear_probe/{model}_{scenario}.json'))
            array = np.load(vectors, mmap_mode='r', allow_pickle=False)
            x = torch.from_numpy(array[original_indices].copy()).cuda()
            y = torch.tensor(original_truth, device='cuda')
            restricted, verified = [], []
            for record in session_old['final_runs'][model]:
                folder = ROOT / record['folder']
                if sha256(folder/'best_head.pt') != record['head_sha256'] or sha256(folder/'preprocessing.pt') != record['preprocessing_sha256']:
                    raise ValueError('A preserved 13-class artifact changed')
                if record['embedding_sha256'] != cache['sha256']:
                    raise ValueError('Original head and frozen cache no longer match')
                prep = torch.load(folder/'preprocessing.pt', map_location='cuda', weights_only=True)
                head = torch.nn.Linear(768, 13).cuda()
                head.load_state_dict(torch.load(folder/'best_head.pt', map_location='cuda', weights_only=True), strict=True)
                metrics, prediction = evaluate(head, (x-prep['mean'])/prep['scale'], y)
                with np.load(folder/'test_predictions.npz', allow_pickle=False) as saved:
                    if not (np.array_equal(saved['indices'], original_indices) and np.array_equal(saved['true'], original_truth)
                            and np.array_equal(saved['predicted'], prediction)):
                        raise ValueError('Preserved 13-class predictions were not reproduced')
                if metrics['confusion_matrix'] != record['test_metrics']['confusion_matrix']:
                    raise ValueError('Preserved original confusion matrix changed')
                restricted.append(restricted_original_metrics(original_truth[keep], prediction[keep], classes))
                verified.append({'seed': record['training_seed'], 'exact_predictions': True,
                                 'original_test_samples': len(original_indices), 'retained_test_samples': len(indices)})
            same_wavs = aggregate(restricted)
            a, b = old_model['aggregate'], new_model['aggregate']
            item['comparison'][model] = {'original_13_classes_full_test': a,
                'original_13_output_head_on_same_12_class_wavs': same_wavs,
                'new_12_class_head': b, 'original_heads_verified': verified,
                'delta_percentage_points_vs_original_full_test': {key: 100*(b[key]['mean']-a[key]['mean']) for key in METRICS},
                'delta_percentage_points_on_same_retained_wavs': {key: 100*(b[key]['mean']-same_wavs[key]['mean']) for key in METRICS}}
            lines.append('| '+label+' | '+model_name+' | '+' | '.join(mean_std(b[key]) for key in METRICS)+' |')
            comparisons.append('| '+label+' | '+model_name+' | '+mean_std(a['accuracy'])+' | '+mean_std(b['accuracy'])+' | '+
                f'{100*(b["accuracy"]["mean"]-a["accuracy"]["mean"]):+.2f}'+' | '+mean_std(a['macro_f1'])+' | '+mean_std(b['macro_f1'])+' |')
            retained_comparisons.append('| '+label+' | '+model_name+' | '+mean_std(same_wavs['accuracy'])+' | '+mean_std(b['accuracy'])+' | '+
                mean_std(same_wavs['macro_f1'])+' | '+mean_std(b['macro_f1'])+' |')
            old_808 = np.asarray([r['per_class']['808']['recall'] for r in restricted])
            new_808 = new_model['per_class']['808']['recall']
            details.append(f'- {label}, {model_name}: recall de 808 nos mesmos WAVs, cabeça antiga **{percent(old_808.mean())}%**; cabeça sem TS9 **{mean_std(new_808)}%**. Configuração nova `{new_model["configuration"]}`. Treinos finais: '+
                ', '.join(f'seed {r["training_seed"]}: {r["epochs_run"]} épocas/{r["elapsed_seconds"]:.2f} s' for r in new_model['final_runs'])+'.')
            total_training += new_model['total_training_seconds']
            del x, y, array, head, prep
            torch.cuda.empty_cache()
        public['scenarios'][scenario] = item
    public.update(total_consolidation_seconds=total_seconds, total_training_seconds=total_training,
                  training_runs=136, final_heads=40, original_heads_reverified=40)
    lines += ['', '## Comparação com os resultados anteriores', '',
              'Esta tabela compara problemas com 13 e 12 classes e conjuntos de teste de tamanhos diferentes. A diferença não significa melhoria no problema original.', '',
              '| Cenário | Modelo | Acurácia 13 classes (%) | Acurácia sem TS9 (%) | Diferença (p.p.) | F1 macro 13 classes (%) | F1 macro sem TS9 (%) |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |', *comparisons, '',
              '## Comparação nos mesmos WAVs restantes', '',
              'Recarregamos também as 40 cabeças antigas e reproduzimos todas as predições originais. A tabela abaixo avalia essas cabeças de 13 saídas somente nos WAVs das 12 classes restantes. Uma predição TS9 continua contando como erro; não removemos logits nem corrigimos previsões antigas. F1 macro é calculado sobre as 12 classes verdadeiras. As cabeças novas usam exatamente esses mesmos WAVs de teste.', '',
              '| Cenário | Modelo | Acurácia cabeça antiga nos mesmos WAVs (%) | Acurácia cabeça nova (%) | F1 macro antiga nas 12 classes (%) | F1 macro nova (%) |',
              '| --- | --- | ---: | ---: | ---: | ---: |', *retained_comparisons, '',
              '## Classe 808, configurações e tempos', '', *details, '',
              f'136 treinamentos: 8 candidatos + 4 confirmações de finalistas + 5 seeds finais por modelo/cenário. Consolidação total: **{total_seconds/60:.2f} minutos**; soma dos tempos registrados dos treinos: **{total_training/60:.2f} minutos**. Os tempos excluem extração anterior, edição do código e verificação/reportagem posterior. Não houve nova rodada inicial de dez épocas.', '',
              '## Integridade e limites', '',
              'Retiramos TS9 do treino, validação e teste, remapeando VTB do índice 12 para 11. Encoders, WAVs, caches, partições por fonte e arquivos originais foram preservados. Scalers e pesos de classes foram reajustados somente nos dados de treino restantes. A busca foi repetida com o mesmo orçamento e seleção por validação; suas configurações podem diferir das escolhidas para 13 classes.', '',
              'As 40 cabeças novas reproduziram todas as predições de teste. O caminho WAV → encoder → scaler → cabeça foi conferido em 16 WAVs por modelo/cenário, com logits exatos. Poly Continuous mantém os pares duplicados MGS na mesma partição do experimento original.', '',
              'O teste já foi consultado e motivou esta ablação. Os resultados são exploratórios, com DP entre treinamentos na mesma partição; não são uma avaliação externa cega nem medida de incerteza entre novas fontes. O experimento com 13 classes continua sendo a referência principal do TCC.', '',
              '## Consulta e reprodução', '',
              '- Notebook: `notebooks/transfer_learning_no_ts9.ipynb`, execução desabilitada por padrão.',
              '- Executar: `python scripts/run_transfer_no_ts9.py --run` (requer manifests e os oito caches originais validados).',
              '- Gerar comparação sem treinar: `python scripts/report_transfer_no_ts9.py`.',
              '- Para inferência, carregar a cabeça com `load_consolidated_probe` e interpretar as saídas em `probe.classes`.', '',
              'Relatórios individuais:', '']
    for scenario in SCENARIOS:
        lines.append(f'- [{scenario}](transfer_learning_no_ts9_{scenario}_consolidated_report.md).')
    lines += ['', 'Resumo estruturado: [transfer_learning_no_ts9_summary.json](transfer_learning_no_ts9_summary.json). Dados, pesos, caches, gráficos, logs e documentos de contexto permanecem locais.']
    write_json(PUBLIC_RESULTS/'transfer_learning_no_ts9_summary.json', public)
    (PUBLIC_RESULTS/'transfer_learning_no_ts9_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('NO_TS9_REPORT_COMPLETE', 'new heads: 40; preserved original heads reverified: 40')
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report()
