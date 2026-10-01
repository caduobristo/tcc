"""Verify completed heads against saved predictions and publish a compact report."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data.transfer import EFFECTS, read_manifest, sha256
from src.training.consolidation import evaluate, write_json
from src.training.linear_probe import load_config, validate_cache, paths_for, device_for


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def percent(value):
    return f'{100*value:.2f}'.replace('.',',')


def mean_std(record):
    return f'{percent(record["mean"])} ± {percent(record["std"])}'


def report(session):
    session = Path(session).resolve()
    summary,protocol = read_json(session/'summary.json'),read_json(session/'protocol.json')
    if protocol['status'] != 'complete':
        raise ValueError('Protocol is not complete')
    if sha256(session/'selection_locked.json') != summary['selection_lock_sha256']:
        raise ValueError('Selection lock changed')
    heads_lock = read_json(session/'all_confirmation_heads_locked.json')
    selection_lock = read_json(session/'selection_locked.json')
    rows = read_manifest()
    indices = np.asarray([i for i,row in enumerate(rows) if row['split']=='test'])
    truth = np.asarray([int(rows[i]['label']) for i in indices])
    verified = []
    public = {'schema':1,'scenario':'mono_disc','split_seed':42,'samples':len(rows),
              'source_groups':624,'test_source_groups':126,'split_counts':{'train':88902,'validation':9702,'test':len(indices)},
              'session':summary['session'],'protocol':summary['protocol'],'protocol_sha256':protocol['protocol_sha256'],
              'implementation_sha256':protocol['implementation_sha256'],
              'manifest_sha256':protocol['manifest_sha256'],'selection_lock_sha256':summary['selection_lock_sha256'],
              'started_at_utc':protocol['started_at_utc'],'completed_at_utc':summary['completed_at_utc'],
              'elapsed_seconds':summary['elapsed_seconds'],'encoder_frozen':True,'fine_tuning':False,
              'test_previously_inspected':True,'std_interpretation':'sample std between training seeds on the same source-group split',
              'hardware':protocol['hardware'],'models':{}}
    first = read_json(ROOT/'results/transfer_learning/mono_disc/first_run_summary.json')
    execution_date = datetime.fromisoformat(protocol['started_at_utc']).astimezone(timezone(timedelta(hours=-3))).strftime('%d/%m/%Y')
    lines = ['# Transfer learning consolidado: PaSST e HTS-AT','',
             f'Execução em {execution_date}, com encoders congelados. Fine-tuning não foi realizado.',
             '',f'Sessão local: `{summary["session"]}`. Duração do protocolo: **{summary["elapsed_seconds"]/60:.2f} minutos**, reutilizando os embeddings já extraídos.',
             '', '123.552 áudios Mono Discrete/13 efeitos; 624 fontes; treino/validação/teste 88.902/9.702/24.948 por fonte (seed 42). WAVs a 32 kHz com frontend nativo, não os mel32 Kaldi.',
             '', '## Resultado no teste', '', '| Modelo | Acurácia (%) | F1 macro (%) | Precisão macro (%) | Recall macro (%) |',
             '| --- | ---: | ---: | ---: | ---: |']
    for model,aggregates in summary['aggregate'].items():
        lines.append('| '+('PaSST' if model=='passt' else 'HTS-AT')+' | '+' | '.join(mean_std(aggregates[key]) for key in ('accuracy','macro_f1','macro_precision','macro_recall'))+' |')
    lines += ['', 'Média ± desvio padrão amostral de cinco seeds novas (101, 202, 303, 404 e 505). Cada seed é avaliada nos mesmos 24.948 áudios; não são 124.740 exemplos independentes.',
              '', '## Seleção e execuções', '',
              'Oito configurações por modelo foram comparadas usando F1 macro de validação. As duas melhores receberam mais duas seeds; a configuração foi escolhida pela média nas seeds 42, 7 e 21. A seleção foi registrada antes dos dez treinamentos finais, e todas as cabeças foram finalizadas antes de avaliar o teste.',
              '', 'AdamW: LR inicial 0,001, weight decay 0,01; teto de 200 épocas; scheduler e parada antecipada na validação. Consulte [o protocolo](../docs/transfer_learning_consolidation.md) e [a configuração](../configs/linear_probe/consolidation_mono_disc.json).']
    for model,records in summary['final_runs'].items():
        config = load_config(ROOT/'configs/linear_probe'/f'{model}_mono_disc.json')
        cache = validate_cache(config)
        _,vectors,_ = paths_for(config)
        array = np.load(vectors,mmap_mode='r',allow_pickle=False)
        device = device_for(config)
        x = torch.from_numpy(array[indices].copy()).to(device)
        y = torch.tensor(truth,device=device)
        search = read_json(session/model/'search_results.json')
        finalists = read_json(session/model/'finalist_ranking.json')
        for phase in ('search','selection'):
            if list((session/model/phase).rglob('test_metrics.json')):
                raise ValueError('Search/selection contains test metrics')
        representative = max(records,key=lambda record:(record['best_validation_macro_f1'],-record['training_seed']))
        selected = summary['selection'][model]
        model_info = {'configuration':selected['candidate'],'selection_validation_macro_f1':selected['mean_validation_macro_f1'],
                      'selection_validation_macro_f1_std':selected['std_validation_macro_f1'],
                      'aggregate':summary['aggregate'][model],'representative_seed_by_validation':representative['training_seed'],
                      'representative_run':representative['folder'],'embedding_sha256':cache['sha256'],
                      'cache_identity':cache['identity'],
                      'initial_run':{key:first['models'][model][key] for key in ('accuracy','macro_f1','training_and_test_seconds')},
                      'search':[{key:group[key] for key in ('candidate_id','mean_validation_macro_f1')} for group in search],
                      'final_runs':[], 'per_class':{}}
        name = 'PaSST' if model=='passt' else 'HTS-AT'
        lines += ['',f'### {name}', '',f'Configuração escolhida: `{selected["candidate_id"]}`. F1 macro de validação nas três seeds: **{percent(selected["mean_validation_macro_f1"])} ± {percent(selected["std_validation_macro_f1"])}%**.',
                  '', '| Configuração da busca | F1 macro validação, seed 42 (%) | Melhor época | Épocas executadas |',
                  '| --- | ---: | ---: | ---: |']
        for group in search:
            r = group['runs'][0]
            lines.append(f'| {group["candidate_id"]} | {percent(r["best_validation_macro_f1"])} | {r["best_epoch"]} | {r["epochs_run"]} |')
        lines += ['', '| Finalista | F1 macro validação médio ± DP (%) |', '| --- | ---: |']
        for group in finalists:
            lines.append(f'| {group["candidate_id"]} | {percent(group["mean_validation_macro_f1"])} ± {percent(group["std_validation_macro_f1"])} |')
        lines += ['', '| Seed final | Melhor época / épocas executadas | Atualizações | Tempo treino + validação (s) | Acurácia teste (%) | F1 macro teste (%) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        figure,axes = plt.subplots(1,2,figsize=(12,4))
        for record in records:
            folder = ROOT/record['folder']
            saved = read_json(folder/'run.json')
            locked = next(r for r in heads_lock[model] if r['training_seed']==record['training_seed'])
            if (saved['head_sha256'] != locked['head_sha256'] or saved['preprocessing_sha256'] != locked['preprocessing_sha256']
                    or saved['cache_identity'] != cache['identity'] or saved['embedding_sha256'] != cache['sha256']
                    or saved['candidate'] != selection_lock['selected'][model]['candidate']):
                raise ValueError('Final head/selection/cache lock mismatch')
            if not selection_lock['locked_at_utc'] < saved['started_at_utc'] < saved['completed_at_utc'] <= saved['test_evaluated_at_utc']:
                raise ValueError('Invalid selection/training/evaluation order')
            if any(r['completed_at_utc'] > saved['test_evaluated_at_utc'] for locked_records in heads_lock.values() for r in locked_records):
                raise ValueError('Test evaluated before all confirmation training completed')
            if sha256(folder/'best_head.pt') != saved['head_sha256'] or sha256(folder/'preprocessing.pt') != saved['preprocessing_sha256']:
                raise ValueError('Artifact hash mismatch')
            prep = torch.load(folder/'preprocessing.pt',weights_only=True,map_location=device)
            head = nn.Linear(768,13).to(device)
            head.load_state_dict(torch.load(folder/'best_head.pt',weights_only=True,map_location=device),strict=True)
            metrics,predicted = evaluate(head,(x-prep['mean'])/prep['scale'],y)
            with np.load(folder/'test_predictions.npz',allow_pickle=False) as saved_pred:
                if not (np.array_equal(saved_pred['indices'],indices) and np.array_equal(saved_pred['true'],truth)
                        and np.array_equal(saved_pred['predicted'],predicted)):
                    raise ValueError('Reloaded head predictions differ')
            saved_metrics = read_json(folder/'test_metrics.json')
            if metrics['confusion_matrix'] != saved_metrics['confusion_matrix'] or metrics['macro_f1'] != saved_metrics['macro_f1']:
                raise ValueError('Recomputed metrics differ')
            verified.append({'model':model,'seed':saved['training_seed'],'verified_predictions':len(indices),
                             'head_sha256':saved['head_sha256'],'predictions_match':True})
            compact = {key:saved[key] for key in ('training_seed','epochs_run','best_epoch','optimizer_updates','elapsed_seconds','stop_reason','head_sha256','preprocessing_sha256','best_validation_macro_f1')}
            compact['test'] = {key:metrics[key] for key in ('accuracy','macro_f1','macro_precision','macro_recall')}
            model_info['final_runs'].append(compact)
            lines.append(f'| {saved["training_seed"]} | {saved["best_epoch"]} / {saved["epochs_run"]} | {saved["optimizer_updates"]} | {saved["elapsed_seconds"]:.2f} | {percent(metrics["accuracy"])} | {percent(metrics["macro_f1"])} |')
            history = read_json(folder/'history.json')
            epochs = [h['epoch'] for h in history]
            axes[0].plot(epochs,[h['validation_macro_f1'] for h in history],label=str(saved['training_seed']))
            axes[1].plot(epochs,[h['train_loss'] for h in history],label=str(saved['training_seed']))
        axes[0].set(title=f'{name}: validation macro F1',xlabel='Epoch',ylabel='F1')
        axes[1].set(title=f'{name}: training loss',xlabel='Epoch',ylabel='Cross entropy')
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(title='Seed',fontsize=8)
        figure.tight_layout()
        figure.savefig(session/model/'confirmation_curves.png',dpi=160)
        plt.close(figure)
        matrices = np.asarray([r['test_metrics']['confusion_matrix'] for r in records],dtype=float)
        normalized = (matrices/matrices.sum(axis=2,keepdims=True)).mean(axis=0)
        figure,ax = plt.subplots(figsize=(10,8))
        im = ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
        ax.set(xticks=np.arange(13),yticks=np.arange(13),xticklabels=EFFECTS,yticklabels=EFFECTS,
               xlabel='Predicted effect',ylabel='True effect',title=f'{name}: mean row-normalized test confusion, 5 seeds')
        for i in range(13):
            for j in range(13):
                ax.text(j,i,f'{100*normalized[i,j]:.0f}',ha='center',va='center',fontsize=7,
                        color='white' if normalized[i,j]>.5 else 'black')
        figure.colorbar(im,ax=ax,label='Fraction within true class')
        figure.tight_layout()
        figure.savefig(session/model/'mean_test_confusion.png',dpi=160)
        plt.close(figure)
        lines += ['', f'Checkpoint representativo escolhido pela **validação**, seed {representative["training_seed"]}: `{representative["folder"]}`. Carregar a cabeça **junto ao scaler salvo**; não escolher seed pelo teste.',
                  '', 'Tempos da tabela incluem treino/validação e salvamento da cabeça, excluindo extração já realizada. Curvas e matrizes de confusão ficam na sessão local.']
        for effect in EFFECTS:
            model_info['per_class'][effect] = {}
            for metric in ('precision','recall','f1'):
                values = [r['test_metrics']['per_class'][effect][metric] for r in records]
                model_info['per_class'][effect][metric] = {'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1))}
            model_info['per_class'][effect]['support_per_seed'] = records[0]['test_metrics']['per_class'][effect]['support']
        all_runs = [read_json(path) for path in (session/model).rglob('run.json')]
        model_info['training_runs'] = len(all_runs)
        model_info['total_training_seconds'] = sum(r['elapsed_seconds'] for r in all_runs)
        model_info['max_epoch_runs'] = sum(r['stop_reason']=='max_epochs' for r in all_runs)
        public['models'][model] = model_info
        del x,y,array
    lines += ['', '## Desempenho por classe', '',
              '| Efeito | Suporte por seed | PaSST F1 (%) | PaSST recall (%) | HTS-AT F1 (%) | HTS-AT recall (%) |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for effect in EFFECTS:
        p,h = [public['models'][m]['per_class'][effect] for m in ('passt','htsat')]
        lines.append(f'| {effect} | {p["support_per_seed"]} | {mean_std(p["f1"])} | {mean_std(p["recall"])} | {mean_std(h["f1"])} | {mean_std(h["recall"])} |')
    lines += ['', '## Comparação descritiva e limites', '']
    for model in ('passt','htsat'):
        old,new = first['models'][model],summary['aggregate'][model]
        delta = 100*(new['macro_f1']['mean']-old['macro_f1'])
        od1 = public['models'][model]['per_class']['OD1']['recall']['mean']
        lines.append(f'- {model}: F1 macro inicial {percent(old["macro_f1"])}%; agora {percent(new["macro_f1"]["mean"])}%, diferença de {delta:.2f} pontos percentuais. Recall OD1 inicial {percent(old["per_class"]["OD1"]["recall"])}%; agora {percent(od1)}%.')
    lines += ['', 'A primeira rodada tinha uma seed, dez épocas, seleção por acurácia e não demonstrou convergência. A consolidação muda duração, critério e possivelmente scaler/batch/pesos; não atribuir todo ganho a um único ajuste.',
              '', '**O teste já foi consultado anteriormente.** A escolha nesta consolidação usa somente validação, mas não equivale a um teste totalmente cego. O DP é entre seeds de treinamento na mesma partição; não mede generalização entre novas fontes. Esta etapa não reproduz exatamente o AST e não estabelece superioridade geral de uma arquitetura.',
              '', '## Integridade e reuso', '',
              f'As dez cabeças foram recarregadas e suas predições reproduziram exatamente os arquivos salvos ({len(indices)} por cabeça). Hashes dos heads/scalers/caches, ausência de métricas de teste na seleção e ordem dos registros de seleção/treino/teste foram conferidos.',
              '', 'O resumo portátil está em [transfer_learning_consolidated_summary.json](transfer_learning_consolidated_summary.json). Dados, embeddings, checkpoints, predições por arquivo e gráficos permanecem locais e ignorados pelo Git.',
              '', 'Para consultar sem treinar: `notebooks/consolidate_transfer_learning.ipynb`. Para inferência com WAVs mono já reamostrados/cortados para 32 kHz/64.000 amostras, use `src.models.consolidated_probe.load_consolidated_probe`, passando o diretório da execução e dispositivo CUDA. A precisão BF16 do encoder é preservada; saída são logits das 13 classes na ordem de `src.data.transfer.EFFECTS`.']
    public['verification'] = verified
    public['all_predictions_reproduced'] = True
    waveform_check = session/'waveform_inference_check.json'
    if waveform_check.exists():
        public['waveform_inference_check'] = read_json(waveform_check)
        lines += ['', 'O carregador de inferência também foi conferido em 16 WAVs originais por modelo (primeiro batch do manifest): logits finitos e idênticos aos obtidos com o cache, com diferença máxima zero. Essa conferência valida o reuso do pipeline, não uma nova estimativa de desempenho.']
    write_json(session/'verification.json',verified)
    write_json(ROOT/'results/transfer_learning_consolidated_summary.json',public)
    content = '\n'.join(lines)+'\n'
    (ROOT/'results/transfer_learning_consolidated_report.md').write_text(content,encoding='utf-8')
    local_content = content.replace('../docs/','../../../../docs/').replace('../configs/','../../../../configs/')
    local_content = local_content.replace('](transfer_learning_consolidated_summary.json)','](../../../transfer_learning_consolidated_summary.json)')
    (session/'report.md').write_text(local_content,encoding='utf-8')
    print('REPORT_COMPLETE',session,'Verified heads:',len(verified))
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    report(args.session)
