"""Scientific paired comparison and confusion figures; never trains models."""
import json
from datetime import datetime
import math
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCENARIOS = {'mono_disc': 'Mono Discrete', 'mono_cont': 'Mono Continuous', 'poly_disc': 'Poly Discrete', 'poly_cont': 'Poly Continuous'}
LABELS = {'passt': 'PaSST', 'ast': 'AST', 'audiomae': 'AudioMAE'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stats(item):
    return f'{item["mean"]*100:.2f} ± {item["std"]*100:.2f}'.replace('.', ',')


def class_statistics(item):
    """Keep metrics per seed: nonlinear F1 cannot be computed from a mean matrix."""
    import numpy as np
    result = {}
    for index, label in enumerate(item['classes']):
        records = [run['test_metrics']['per_class'][label] for run in item['final_runs']]
        for run, record in zip(item['final_runs'], records):
            matrix = np.asarray(run['test_metrics']['confusion_matrix'])
            support, predicted = int(matrix[index].sum()), int(matrix[:, index].sum())
            precision = float(matrix[index, index]/predicted) if predicted else 0.0
            recall = float(matrix[index, index]/support) if support else 0.0
            f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
            if record['support'] != support or any(abs(record[key]-value)>1e-12 for key, value in
                    (('precision', precision), ('recall', recall), ('f1', f1))):
                raise ValueError('Per-class metrics differ from the individual confusion matrix')
        if len({record['support'] for record in records}) != 1:
            raise ValueError('Per-class support changed between seeds')
        result[label] = {'support_per_seed': records[0]['support']}
        for metric in ('precision', 'recall', 'f1'):
            values = np.asarray([record[metric] for record in records], dtype=float)
            result[label][metric] = {'mean': float(values.mean()), 'std': float(values.std(ddof=1)), 'values': values.tolist()}
    return result


def strongest_confusions(matrix, limit=5):
    import numpy as np
    values = np.asarray(matrix['mean_row_percent'])
    classes = matrix['classes']
    return [{'true': classes[i], 'predicted': classes[j], 'mean_percent': float(values[i, j])}
            for i, j in sorted(((i, j) for i in range(len(classes)) for j in range(len(classes)) if i != j),
                               key=lambda pair: (-values[pair], pair))[:limit]]


def validate_training_history(folder, record, protocol, sha256):
    """Audit every fitted head, including candidates never evaluated on test."""
    history = read(folder/'history.json')
    if record['status'] != 'complete' or not record['encoder_frozen'] or record['trainable_parameters'] != 9997:
        raise ValueError(f'Invalid completed frozen probe: {folder}')
    if len(history) != record['epochs_run'] or [row['epoch'] for row in history] != list(range(1, len(history)+1)):
        raise ValueError(f'Truncated training history: {folder}')
    per_epoch = math.ceil(record['dataset_view']['split_counts']['train']/record['candidate']['batch_size'])
    reference, waiting = -1.0, 0
    for row in history:
        if row['updates'] != per_epoch*row['epoch'] or not all(math.isfinite(v) for v in row.values()):
            raise ValueError(f'Invalid optimization history: {folder}')
        if row['validation_macro_f1'] > reference+protocol['min_delta']:
            reference, waiting = row['validation_macro_f1'], 0
        else:
            waiting += 1
        if waiting >= protocol['early_stopping_patience'] and row['epoch'] != len(history):
            raise ValueError(f'Training continued after the stopping rule: {folder}')
    expected_reason = 'early_stopping' if waiting >= protocol['early_stopping_patience'] else 'max_epochs'
    if record['stop_reason'] != expected_reason or (expected_reason == 'max_epochs' and len(history) != protocol['max_epochs']):
        raise ValueError(f'Incorrect stopping reason: {folder}')
    best = max(history, key=lambda row: row['validation_macro_f1'])
    if best['epoch'] != record['best_epoch'] or best['validation_macro_f1'] != record['best_validation_macro_f1']:
        raise ValueError(f'Selection differs from training history: {folder}')
    if read(folder/'best_validation_metrics.json')['macro_f1'] != record['best_validation_macro_f1']:
        raise ValueError(f'Saved validation score changed: {folder}')
    if record['optimizer_updates'] != history[-1]['updates']:
        raise ValueError(f'Optimization update count differs: {folder}')
    for name, key in (('best_head.pt', 'head_sha256'), ('preprocessing.pt', 'preprocessing_sha256')):
        if sha256(folder/name) != record[key]:
            raise ValueError(f'Fitted artifact checksum changed: {folder/name}')
    if record['training_seed'] in (7, 21, 42) and record.get('test_evaluated'):
        raise ValueError('Selection candidate was evaluated on test')
    return {'history_sha256': sha256(folder/'history.json'), 'stop_reason': expected_reason}


def confusion_figure(item, model, scenario, condition, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    matrices = np.asarray([run['test_metrics']['confusion_matrix'] for run in item['final_runs']], dtype=float)
    support = matrices.sum(2)
    if not np.all(support > 0) or not np.all(support == support[0]):
        raise ValueError('Five seeds must evaluate the same per-class supports')
    rates = (matrices/support[:, :, None]).mean(0)*100
    assert np.allclose(rates.sum(1), 100)
    classes = item['classes']
    fig, ax = plt.subplots(figsize=(12.5, 10.3), layout='constrained')
    image = ax.imshow(rates, cmap='Blues', vmin=0, vmax=100, interpolation='nearest')
    ticks = np.arange(len(classes))
    ax.set(xticks=ticks, yticks=ticks, xticklabels=classes, yticklabels=classes,
           xlabel='Classe prevista', ylabel='Classe verdadeira')
    title = 'Controle: entrada curta nativa' if model == 'passt' and condition == 'control' else ('Controle: padding' if condition == 'control' else 'Variante: interpolação temporal bicúbica')
    ax.set_title(f'{LABELS[model]} — {SCENARIOS[scenario]} — 13 classes\n{title}\n'
                 f'5 seeds | Acurácia: {stats(item["aggregate"]["accuracy"])}% | F1 macro: {stats(item["aggregate"]["macro_f1"])}%', fontsize=14, pad=15)
    ax.tick_params(labelsize=11)
    for i in ticks:
        for j in ticks:
            ax.text(j, i, f'{rates[i,j]:.1f}'.replace('.', ','), ha='center', va='center', fontsize=9,
                    color='white' if rates[i,j] >= 55 else '#17202a')
    bar = fig.colorbar(image, ax=ax, fraction=.045, pad=.025)
    bar.set_label('Média do percentual por classe verdadeira (%)')
    fig.supxlabel(f'{int(matrices[0].sum()):,} amostras de teste por seed; linhas normalizadas em 100%. Diagonal = recall médio.', fontsize=10)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return {'classes': classes, 'mean_row_percent': rates.tolist(), 'support_per_seed': support[0].astype(int).tolist()}


def report():
    import numpy as np
    from src.data.transfer import sha256
    from src.training.temporal_experiment import OUTPUT
    from scripts.run_temporal_interpolation import verify_protected
    state = read(ROOT/'data/audits/transfer_learning/temporal_interpolation_execution.json')
    if state['status'] != 'complete' or len(state['jobs']) != 12 or any(x['status'] != 'complete' for x in state['jobs'].values()):
        raise ValueError('All twelve model/scenario jobs must finish before a final report')
    verify_protected(state['protected_artifacts'])
    public = ROOT/'results/temporal_interpolation'
    public.mkdir(parents=True, exist_ok=True)
    images = public/'confusion_matrices'
    images.mkdir(exist_ok=True)
    document = read(ROOT/'configs/linear_probe/temporal_interpolation.json')
    summary = {'schema': 1, 'experiment': 'temporal_interpolation_v1', 'protocol': document,
        'encoder_frozen': True, 'fine_tuning': False, 'htsat_retrained': False,
        'classes': document['classes'], 'final_seeds': document['final_seeds'],
        'std_interpretation': 'sample standard deviation over training seeds on one fixed source split',
        'test_previously_inspected': True, 'comparison_table': [], 'scenarios': {},
        'protected_historical_artifacts': len(state['protected_artifacts']), 'protected_hashes_unchanged': True,
        'started_at_utc': state['started_at_utc'], 'completed_at_utc': state['completed_at_utc'],
        'checkpoint_sources': read(ROOT/'checkpoints/pretrained_models/temporal_sources.json')}
    lines = ['# Interpolação temporal: comparação pareada com encoders congelados', '',
        'Experimento exploratório nos quatro cenários GUITAR-FX-DIST, mantendo 13 classes (incluindo TS9) e partições por gravação-fonte com seed 42. Novos controles e variantes usam o mesmo checkpoint, frontend, precisão BF16, implementação de atenção SDPA e orçamento de cabeça dentro de cada modelo. HTS-AT não foi reexecutado.', '',
        'AST/AudioMAE: mel16 log-Mel de 198×128, normalização histórica e padding até 1024 no controle; interpolação 198→1024 na variante. PaSST: WAVs oficiais a 32 kHz, frontend nativo 128 bandas e entrada curta no controle; interpolação ~200→998 na variante. Somente o tempo é alongado, com bicúbica e align_corners=True; não se copia o dobramento da imagem Swin.', '',
        'A interpolação cria valores intermediários no espectrograma, sem adicionar conteúdo acústico ou alongar o WAV. A hipótese de mudar a escala/normalização dos mel16 é separada deste experimento.', '',
        '## Resultados no teste', '',
        'Média ± DP amostral de cinco seeds finais (101, 202, 303, 404 e 505). Δ é variante − controle em pontos percentuais. Não escolher a melhor seed pelo teste. O DP não representa variação entre partições.', '',
        '| Cenário | Modelo | Acurácia controle (%) | Acurácia alongada (%) | Δ acurácia (pp) | F1 macro controle (%) | F1 macro alongada (%) | Δ F1 (pp) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    class_lines = ['# Resultados por classe: interpolação temporal', '',
        'Média ± DP amostral nas mesmas cinco seeds. Suporte = exemplos de teste por seed; não são cinco conjuntos independentes. Δ recall e Δ F1 = interpolação − controle em pontos percentuais. F1 é calculado por seed antes da média.', '']
    total_training = 0.0
    extraction_total = 0.0
    new_heads = 0
    for scenario in SCENARIOS:
        summary['scenarios'][scenario] = {}
        for model in LABELS:
            pair = {condition: read(OUTPUT/scenario/model/condition/'consolidation/summary.json') for condition in ('control', 'bicubic')}
            if any([run['training_seed'] for run in item['final_runs']] != document['final_seeds'] for item in pair.values()):
                raise ValueError('Expected the same ordered final seeds in both conditions')
            if pair['control']['dataset_view'] != pair['bicubic']['dataset_view']:
                raise ValueError('Control and variant must use the same dataset view')
            for key in ('manifest_sha256', 'checkpoint_sha256', 'encoder_precision', 'attention', 'packages', 'implementation'):
                if pair['control']['cache']['identity'][key] != pair['bicubic']['cache']['identity'][key]:
                    raise ValueError(f'Control and variant differ in {key}')
            verification_path = ROOT/'data/audits/transfer_learning'/f'temporal_verified_{scenario}_{model}.json'
            verified = read(verification_path)
            if set(verified['conditions']) != {'control', 'bicubic'} or len(verified['test_classes_covered']) != 13:
                raise ValueError('Input/cache/head verification incomplete')
            portable = {'conditions': {}, 'verification_sha256': sha256(verification_path),
                        'verification': verified, 'paired_difference': {}}
            for condition, item in pair.items():
                source = OUTPUT/scenario/model/condition/'consolidation'
                for interruption in state.get('interruptions', []):
                    interrupted_cache = interruption.get('partial_cache', {})
                    if (interrupted_cache.get('scenario'), interrupted_cache.get('model')) == (scenario, model) or (not interrupted_cache and (scenario, model) == ('mono_cont', 'ast')):
                        partial = interrupted_cache.get('conditions', {}).get(condition) if interrupted_cache else interruption.get('partial_ast_mono_cont', {}).get(condition)
                        if partial:
                            array = np.load(source.parent/'embeddings.npy', mmap_mode='r', allow_pickle=False)
                            digest = hashlib.sha256()
                            for start in range(0, partial['written'], 4096):
                                digest.update(array[start:min(start+4096, partial['written'])].tobytes())
                            if digest.hexdigest() != partial['prefix_sha256']:
                                raise ValueError('Resumed extraction changed the previously written prefix')
                runs = [read(path) for path in source.rglob('run.json') if '_interrupted_' not in str(path)]
                if len(runs) != 17 or any(run['status'] != 'complete' for run in runs):
                    raise ValueError('Expected 17 complete classifier trainings per condition')
                new_heads += len(runs)
                histories = []
                for path in source.rglob('run.json'):
                    if '_interrupted_' in str(path):
                        continue
                    record = read(path)
                    checked = validate_training_history(path.parent, record, item['protocol'], sha256)
                    histories.append({'folder': path.parent.relative_to(ROOT).as_posix(),
                                      'seed': record['training_seed'], 'epochs': record['epochs_run'], **checked})
                total_training += sum(run['elapsed_seconds'] for run in runs)
                image_name = f'{model}_{scenario}_{condition}.png'
                matrix = confusion_figure(item, model, scenario, condition, images/image_name)
                final = []
                for run in item['final_runs']:
                    final.append({k: run[k] for k in ('training_seed', 'epochs_run', 'best_epoch', 'elapsed_seconds', 'head_sha256', 'preprocessing_sha256', 'test_metrics')})
                portable['conditions'][condition] = {'session': item['session'], 'aggregate': item['aggregate'],
                    'selected_configuration': item['selection']['candidate'], 'dataset_view': item['dataset_view'],
                    'final_runs': final, 'training_seconds_all_17_heads': sum(run['elapsed_seconds'] for run in runs),
                    'embedding_sha256': item['cache']['sha256'], 'cache_identity': item['cache']['identity'],
                    'paired_extraction_seconds': item['cache']['elapsed_seconds'],
                    'epochs_final': {'values': [run['epochs_run'] for run in item['final_runs']],
                                     'best_epochs': [run['best_epoch'] for run in item['final_runs']]},
                    'all_17_training_histories_verified': histories,
                    'per_class': class_statistics(item),
                    'source_summary_sha256': sha256(source/'summary.json'),
                    'matrix': {**matrix, 'image': 'confusion_matrices/'+image_name, 'image_sha256': sha256(images/image_name)}}
                portable['conditions'][condition]['strongest_confusions'] = strongest_confusions(matrix)
            paired_seconds = max(item['cache']['elapsed_seconds'] for item in pair.values())
            extraction_total += paired_seconds  # one pass generates both caches; never count twice
            portable['paired_extraction_seconds'] = paired_seconds
            for key in ('accuracy', 'macro_f1', 'macro_precision', 'macro_recall'):
                control, resized = pair['control']['aggregate'][key], pair['bicubic']['aggregate'][key]
                values = np.asarray(resized['values'])-np.asarray(control['values'])
                portable['paired_difference'][key] = {'mean_pp': float(values.mean()*100), 'std_pp': float(values.std(ddof=1)*100), 'values_pp': (values*100).tolist()}
            summary['scenarios'][scenario][model] = portable
            control, resized = pair['control']['aggregate'], pair['bicubic']['aggregate']
            row = {'scenario': SCENARIOS[scenario], 'model': LABELS[model],
                'accuracy_control_percent': control['accuracy']['mean']*100,
                'accuracy_bicubic_percent': resized['accuracy']['mean']*100,
                'delta_accuracy_pp': portable['paired_difference']['accuracy']['mean_pp'],
                'macro_f1_control_percent': control['macro_f1']['mean']*100,
                'macro_f1_bicubic_percent': resized['macro_f1']['mean']*100,
                'delta_macro_f1_pp': portable['paired_difference']['macro_f1']['mean_pp']}
            summary['comparison_table'].append(row)
            lines.append(f'| {row["scenario"]} | {row["model"]} | {stats(control["accuracy"])} | {stats(resized["accuracy"])} | {row["delta_accuracy_pp"]:+.2f} | {stats(control["macro_f1"])} | {stats(resized["macro_f1"])} | {row["delta_macro_f1_pp"]:+.2f} |'.replace('.', ','))
            class_lines += [f'## {SCENARIOS[scenario]} — {LABELS[model]}', '',
                '| Classe | Suporte | Recall controle (%) | Recall alongado (%) | Δ recall (pp) | F1 controle (%) | F1 alongado (%) | Δ F1 (pp) |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            portable['per_class_difference'] = {}
            for label in document['classes']:
                before = portable['conditions']['control']['per_class'][label]
                after = portable['conditions']['bicubic']['per_class'][label]
                if before['support_per_seed'] != after['support_per_seed']:
                    raise ValueError('Control and variant evaluate different per-class supports')
                differences = {}
                for key in ('precision', 'recall', 'f1'):
                    values = (np.asarray(after[key]['values']) - np.asarray(before[key]['values']))*100
                    differences[key] = {'mean_pp': float(values.mean()), 'std_pp': float(values.std(ddof=1)), 'values_pp': values.tolist()}
                portable['per_class_difference'][label] = differences
                class_lines.append(f'| {label} | {before["support_per_seed"]} | {stats(before["recall"])} | {stats(after["recall"])} | {differences["recall"]["mean_pp"]:+.2f} | {stats(before["f1"])} | {stats(after["f1"])} | {differences["f1"]["mean_pp"]:+.2f} |'.replace('.', ','))
            class_lines += ['', '| Condição | Cinco maiores confusões (verdadeira → prevista; % da classe verdadeira) |', '| --- | --- |']
            for condition, item in portable['conditions'].items():
                errors = '; '.join(f'{entry["true"]} → {entry["predicted"]}: {entry["mean_percent"]:.2f}%'.replace('.', ',') for entry in item['strongest_confusions'])
                class_lines.append(f'| {condition} | {errors} |')
            class_lines += ['', f'[Matriz controle](confusion_matrices/{model}_{scenario}_control.png) · [Matriz interpolação](confusion_matrices/{model}_{scenario}_bicubic.png)', '']
    if new_heads != 408:
        raise ValueError('The complete planned budget was not run')
    wall_seconds = (datetime.fromisoformat(state['completed_at_utc'])-datetime.fromisoformat(state['started_at_utc'])).total_seconds()
    measured_invocation_seconds = sum(item.get('elapsed_wall_seconds', 0) for item in state.get('invocations', []))
    summary['timing'] = {'sum_training_seconds_408_heads': total_training, 'paired_extraction_seconds_12_jobs': extraction_total,
                         'first_start_to_final_completion_seconds_including_interruptions': wall_seconds,
                         'measured_invocation_seconds_including_checks': measured_invocation_seconds,
                         'interrupted_invocation_seconds_estimate': sum(item.get('elapsed_wall_seconds_estimate_until_last_progress',0) for item in state.get('invocations', [])),
                         'previous_active_seconds_until_last_progress_estimate': state.get('previous_active_seconds_until_last_progress_estimate'),
                         'interruptions': state.get('interruptions', []), 'invocations': state.get('invocations', [])}
    summary['new_heads'] = new_heads
    summary['final_heads'] = 120
    summary['all_408_training_histories_and_fitted_hashes_verified'] = True
    lines += ['', '## O que mudou com o alongamento', '']
    summary['observations'] = {}
    for model in LABELS:
        rows = [row for row in summary['comparison_table'] if row['model'] == LABELS[model]]
        deltas = [row['delta_macro_f1_pp'] for row in rows]
        positive = sum(value > 0 for value in deltas)
        summary['observations'][model] = {'positive_f1_scenarios': positive, 'f1_delta_min_pp': min(deltas), 'f1_delta_max_pp': max(deltas)}
        lines.append(f'- **{LABELS[model]}**: F1 macro aumentou em {positive}/4 cenários; variação de {min(deltas):+.2f} a {max(deltas):+.2f} pp.'.replace('.', ','))
    largest_gain = max(summary['comparison_table'], key=lambda row: row['delta_macro_f1_pp'])
    lines += ['', f'O maior ganho de F1 macro foi de **{largest_gain["delta_macro_f1_pp"]:+.2f} pp**, em {largest_gain["model"]}/{largest_gain["scenario"]}.'.replace('.', ',')]
    ts9_losses = []
    for scenario, models in summary['scenarios'].items():
        for model, info in models.items():
            difference = info['per_class_difference']['TS9']['f1']['mean_pp']
            if difference < 0:
                ts9_losses.append(f'{LABELS[model]}/{SCENARIOS[scenario]}: {difference:+.2f} pp'.replace('.', ','))
    lines += ['', 'A melhora do F1 macro não implica melhora em todos os pedais. ' +
              ('TS9 perdeu F1 em ' + '; '.join(ts9_losses) + '. ' if ts9_losses else '') +
              'As mudanças de precisão, recall e F1 de cada classe estão registradas separadamente.']
    lines += ['', 'A leitura é descritiva neste split: não declaramos significância estatística nem generalização para novas fontes. A comparação por classe, inclusive TS9 e OD1, está em [resultados por classe](per_class.md).']
    lines += ['', '## Amostras e fontes', '', '| Cenário | Total | Treino | Validação | Teste | Fontes treino / validação / teste |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for scenario, models in summary['scenarios'].items():
        view = models['passt']['conditions']['control']['dataset_view']
        if any(item['conditions']['control']['dataset_view'] != view for item in models.values()):
            raise ValueError('Different models must use identical source partitions and labels')
        counts, groups = view['split_counts'], view['group_counts']
        lines.append(f'| {SCENARIOS[scenario]} | {view["samples"]} | {counts["train"]} | {counts["validation"]} | {counts["test"]} | {groups["train"]} / {groups["validation"]} / {groups["test"]} |')
    lines += ['', '## Precisão e recall macro', '',
        '| Cenário | Modelo | Condição | Precisão macro (%) | Recall macro (%) |', '| --- | --- | --- | ---: | ---: |']
    for scenario, models in summary['scenarios'].items():
        for model, info in models.items():
            for condition, item in info['conditions'].items():
                aggregate = item['aggregate']
                lines.append(f'| {SCENARIOS[scenario]} | {LABELS[model]} | {condition} | {stats(aggregate["macro_precision"])} | {stats(aggregate["macro_recall"])} |')
    lines += ['', '## Matrizes de confusão', '',
        'Cada célula é a média das cinco matrizes normalizadas por classe verdadeira. Linhas verdadeiras, colunas previstas; diagonal = recall. Os suportes são por seed, sem multiplicar o número de exemplos independentes. A média do F1 é calculada por seed, não a partir da matriz média.', '',
        '| Cenário | Modelo | Controle | Interpolação |', '| --- | --- | --- | --- |']
    for scenario in SCENARIOS:
        for model in LABELS:
            lines.append(f'| {SCENARIOS[scenario]} | {LABELS[model]} | [Matriz](confusion_matrices/{model}_{scenario}_control.png) | [Matriz](confusion_matrices/{model}_{scenario}_bicubic.png) |')
    lines += ['', '## Configurações escolhidas e épocas', '',
        'Seleção somente por validação. As cinco seeds finais têm a mesma configuração por condição. Épocas: mínimo–máximo executado; melhor época: mínimo–máximo entre as cinco heads. Cada condição contém 17 treinamentos ao todo.', '',
        '| Cenário | Modelo | Condição | Padronização | Pesos da loss | Batch | Épocas | Melhor época | Soma dos 17 treinos (min) |',
        '| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for scenario, models in summary['scenarios'].items():
        for model, info in models.items():
            for condition, item in info['conditions'].items():
                config, epochs = item['selected_configuration'], item['epochs_final']
                lines.append(f'| {SCENARIOS[scenario]} | {LABELS[model]} | {condition} | {config["normalization"]} | {config["class_weighting"]} | {config["batch_size"]} | {min(epochs["values"])}–{max(epochs["values"])} | {min(epochs["best_epochs"])}–{max(epochs["best_epochs"])} | {item["training_seconds_all_17_heads"]/60:.2f} |')
    lines += ['', '## Protocolo, custo e integridade', '',
        '408 cabeças novas: oito candidatos (padronização none/train_standardize × pesos none/train_balanced × batch 128/1024) na seed 42; dois finalistas recebem seeds 7 e 21; seleção pela maior média de F1 macro de validação; cinco seeds finais novas. Todas as cinco cabeças são congeladas antes da avaliação de teste.', '',
        'Cabeça 768→13 (9.997 parâmetros), AdamW LR 0,001 e weight decay 0,01, limite de 200 épocas, ReduceLROnPlateau (fator 0,5, patience 5, min_lr 1e-5), early stopping (patience 15, min_delta 0,001). Scaler e pesos de loss ajustados somente no treino. A seleção pode resultar em configurações distintas por condição; a comparação mede a receita de entrada sob o mesmo orçamento de seleção.', '',
        f'Soma dos tempos de extração pareada: **{extraction_total/60:.2f} min**. Soma dos 408 treinamentos de cabeça: **{total_training/60:.2f} min**. Extração pareada conta leitura e as duas inferências uma vez, não duas vezes; os tempos de cabeça excluem extração e teste. Hardware: RTX 5080, CUDA e BF16 no encoder; heads em float32.', '',
        f'Intervalo entre a primeira inicialização e a conclusão: **{wall_seconds/3600:.2f} h**, incluindo pausas por interrupção. Início UTC: {state["started_at_utc"]}; fim UTC: {state["completed_at_utc"]}. Esse intervalo não é o tempo efetivo de computação. As invocações instrumentadas registraram **{measured_invocation_seconds/3600:.2f} h**, incluindo revalidação e preparação; o período ativo anterior à instrumentação é apenas uma estimativa no JSON. Os tempos de extração e treinamento acima excluem a pausa de retomada.', '',
        f'As 120 heads/scalers foram recarregadas; predições de teste reproduzidas exatamente e métricas/médias/DP recalculadas por sklearn. Scalers e pesos da loss foram reproduzidos usando somente treino. Inferência dos mesmos lotes de extração foi comparada byte a byte com os caches; 16 exemplos de teste por modelo/condição/cenário, cobrindo as 13 classes, tiveram logits exatos. Os **{len(state["protected_artifacts"])} artefatos históricos** mantiveram seus hashes.', '',
        'Os 408 históricos foram conferidos quanto a épocas consecutivas, valores finitos, atualizações do otimizador, melhor época/score de validação, parada antecipada e hashes das heads/scalers. Os candidatos de seleção não foram avaliados no teste.', '',
        'Todos os NPYs utilizados passam por SHA-256 por membro, CRC, shape/dtype e valores finitos. A identidade por efeito/nome, os índices, rótulos e partições dos manifests originais são preservados; nenhuma amostra faltante é descartada silenciosamente.', '',
        '## Limites da conclusão', '',
        'A comparação controle/variante desta rodada é pareada em dados, checkpoint, precisão e protocolo. Os testes já foram consultados e motivaram a hipótese; não se trata de teste externo cego. DP entre seeds mede apenas a variação da cabeça no mesmo split. A arquitetura, as bandas/normalização e os checkpoints diferem entre modelos, portanto diferenças entre modelos não isolam o frontend.', '',
        'A versão AST é a conversão publicada do checkpoint AudioSet 0.4593: nomes e Q/K/V são revertidos sem modificar valores e todos os tensores do backbone são contabilizados. A identidade byte a byte com o checkpoint privado utilizado pelo colega não é comprovada. Resultados históricos AST/AudioMAE também usaram outro protocolo de seleção; a inferência sobre o efeito do alongamento deve usar os novos controles.', '',
        'O kernel SDPA foi comparado com a atenção original em float32; o caminho numérico BF16 pode variar frente às execuções históricas. Por isso também refizemos o controle PaSST. O frontend e os encoders originais permanecem intactos; resultados antigos não são sobrescritos.', '',
        '## Reprodução e fontes', '',
        '- [Notebook](../../notebooks/temporal_interpolation.ipynb): consulta padrão com flags False.',
        '- [Protocolo](../../configs/linear_probe/temporal_interpolation.json).',
        '- [Coordenador](../../scripts/run_temporal_interpolation.py), [verificação](../../scripts/verify_temporal_interpolation.py) e [relatório](../../scripts/report_temporal_interpolation.py).',
        '- [Preparação do ambiente legado](../../scripts/setup_temporal_runtime.py), [dependências](../../requirements-temporal-legacy.txt) e [checkpoints](../../scripts/prepare_temporal_checkpoints.py).',
        '- [Método de adaptação HTS-AT, implementação oficial](https://github.com/RetroCirce/HTS-Audio-Transformer/blob/main/model/htsat.py).',
        '- [Checkpoint AST MIT](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) e [AudioMAE oficial](https://github.com/facebookresearch/AudioMAE).', '',
        'Usar o Python de `.venv-transfer`. Preparar o runtime separado com `python scripts/setup_temporal_runtime.py --prepare` e os pesos com `python scripts/prepare_temporal_checkpoints.py`; esse último comando verifica SHA e exige o checkpoint AudioMAE oficial local. Nunca alterar runtime, código ou protocolo durante a execução. Com entradas e checkpoints locais verificados, executar `python scripts/run_temporal_interpolation.py --run`; o runner reutiliza sessões/caches completos e não sobrescreve artefatos antigos. Depois: `python scripts/report_temporal_interpolation.py`. Dados, pesos, predições individuais, logs e contexto IA continuam locais.']
    (public/'per_class.md').write_text('\n'.join(class_lines).rstrip()+'\n', encoding='utf-8')
    (public/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (public/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('TEMPORAL_REPORT_COMPLETE', public, flush=True)
    return public


if __name__ == '__main__':
    report()
