"""Validation-only selection and independent-seed confirmation of frozen linear probes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix

from src.data.transfer import EFFECTS, read_manifest, manifest_path, sha256
from src.data.paths import SCENARIOS
from src.models.transfer import PROJECT_ROOT
from src.training.linear_probe import load_config, validate_cache, paths_for, device_for, seed_all


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_plan(plan):
    if (plan['schema'] != 1 or plan['scenario'] not in SCENARIOS or plan['split_seed'] != 42
            or plan['train_encoder'] or plan['test_used_for_selection']
            or plan['primary_metric'] != 'validation_macro_f1'):
        raise ValueError('Unsupported consolidation protocol')
    if not plan['models'] or len(set(plan['models'])) != len(plan['models']) or set(plan['models']) - {'passt','htsat'}:
        raise ValueError('Invalid model list')
    if set(plan['normalization']) - {'none','train_standardize'} or set(plan['class_weighting']) - {'none','train_balanced'}:
        raise ValueError('Invalid preprocessing')
    for key in ('selection_seeds','final_seeds','batch_sizes','normalization','class_weighting'):
        if not plan[key] or len(set(plan[key])) != len(plan[key]):
            raise ValueError(f'Empty or duplicate {key}')
    if plan['search_seed'] not in plan['selection_seeds'] or len(plan['final_seeds']) < 2:
        raise ValueError('Invalid selection/confirmation seeds')
    if set(plan['final_seeds']) & set(plan['selection_seeds']):
        raise ValueError('Confirmation seeds must be separate from selection seeds')
    if any(b <= 0 for b in plan['batch_sizes']) or not 1 <= plan['finalists_per_model'] <= len(candidate_grid(plan)):
        raise ValueError('Invalid search budget')
    if min(plan['max_epochs'],plan['early_stopping_patience']) <= 0 or plan['min_delta'] < 0:
        raise ValueError('Invalid stopping rules')
    if not 0 < plan['min_learning_rate'] <= plan['learning_rate'] or not 0 < plan['scheduler_factor'] < 1:
        raise ValueError('Invalid learning-rate schedule')


def metrics_from_cm(cm):
    cm = np.asarray(cm, dtype=np.int64)
    diagonal = np.diag(cm).astype(float)
    precision = np.divide(diagonal, cm.sum(0), out=np.zeros_like(diagonal), where=cm.sum(0)!=0)
    recall = np.divide(diagonal, cm.sum(1), out=np.zeros_like(diagonal), where=cm.sum(1)!=0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(diagonal), where=precision+recall!=0)
    return {'accuracy':float(diagonal.sum()/cm.sum()), 'macro_f1':float(f1.mean()),
            'macro_precision':float(precision.mean()), 'macro_recall':float(recall.mean()),
            'confusion_matrix':cm.tolist(),
            'per_class':{name:{'precision':float(precision[i]),'recall':float(recall[i]),
                               'f1':float(f1[i]),'support':int(cm[i].sum())} for i,name in enumerate(EFFECTS)}}


def fit_preprocessing(train_x, normalization):
    if normalization == 'none':
        return torch.zeros(train_x.shape[1], device=train_x.device), torch.ones(train_x.shape[1],device=train_x.device)
    if normalization != 'train_standardize':
        raise ValueError('Unknown embedding normalization')
    return train_x.mean(0), train_x.std(0,unbiased=False).clamp_min(1e-6)


def class_weights(train_y, weighting, classes=13):
    if weighting == 'none':
        return torch.ones(classes,device=train_y.device)
    if weighting != 'train_balanced':
        raise ValueError('Unknown training class weighting')
    counts = torch.bincount(train_y, minlength=classes)
    if bool((counts==0).any()):
        raise ValueError('Training partition lacks a class')
    return len(train_y)/(classes*counts.float())


@dataclass
class SelectionData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    cache: dict
    model: str


def load_selection_data(model, plan):
    """Only train/validation rows are made available to candidate training."""
    scenario = plan['scenario']
    config = load_config(PROJECT_ROOT/'configs/linear_probe'/f'{model}_{scenario}.json')
    report = json.loads(manifest_path(scenario).with_suffix('.json').read_text())
    if (report['seed'] != plan['split_seed'] or report['scenario'] != scenario
            or report['manifest_sha256'] != sha256(manifest_path(scenario)) or report['classes'] != list(EFFECTS)):
        raise ValueError('Consolidation must preserve the original source-group partitions')
    cache = validate_cache(config)
    _, vectors, _ = paths_for(config)
    array = np.load(vectors, mmap_mode='r', allow_pickle=False)
    rows = read_manifest(scenario)
    group_splits = {}
    for row in rows:
        if int(row['label']) != EFFECTS.index(row['effect']) or row['split'] not in ('train','validation','test'):
            raise ValueError('Invalid manifest class mapping or partition')
        prior = group_splits.setdefault(row['source_group'],row['split'])
        if prior != row['split']:
            raise ValueError('Source recording leaks across partitions')
    device = device_for(config)
    data = {}
    for split in ('train','validation'):
        indices = [i for i,row in enumerate(rows) if row['split']==split]
        data[split+'_x'] = torch.from_numpy(array[indices].copy()).to(device)
        data[split+'_y'] = torch.tensor([int(rows[i]['label']) for i in indices], device=device)
    return SelectionData(**data,cache=cache,model=model)


def candidate_grid(plan):
    return [{'normalization':normalization,'class_weighting':weighting,'batch_size':batch}
            for normalization,weighting,batch in itertools.product(plan['normalization'],plan['class_weighting'],plan['batch_sizes'])]


def candidate_id(candidate):
    return f'{candidate["normalization"]}_{candidate["class_weighting"]}_b{candidate["batch_size"]}'


def evaluate(head, x, y, batch_size=2048):
    head.eval()
    predicted = []
    loss_sum = torch.zeros((),device=x.device)
    with torch.inference_mode():
        for start in range(0,len(y),batch_size):
            logits = head(x[start:start+batch_size])
            predicted.append(logits.argmax(1).cpu().numpy())
            loss_sum += nn.functional.cross_entropy(logits,y[start:start+batch_size],reduction='sum')
    prediction = np.concatenate(predicted)
    labels = y.cpu().numpy()
    metrics = metrics_from_cm(confusion_matrix(labels,prediction,labels=list(range(13))))
    metrics['loss'] = float(loss_sum/len(y))
    return metrics, prediction


def fit_candidate(data, candidate, seed, plan, folder, *, allow_training=False):
    if not allow_training:
        raise RuntimeError('Consolidation training disabled')
    if plan.get('train_encoder') or plan.get('test_used_for_selection'):
        raise ValueError('Only frozen encoders and validation selection are allowed')
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    seed_all(seed)
    mean,scale = fit_preprocessing(data.train_x,candidate['normalization'])
    x_train = (data.train_x-mean)/scale
    x_val = (data.validation_x-mean)/scale
    weights = class_weights(data.train_y,candidate['class_weighting'])
    device = x_train.device
    head = nn.Linear(x_train.shape[1],13).to(device)
    optimizer = torch.optim.AdamW(head.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'],fused=device.type=='cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=plan['scheduler_factor'],
                  patience=plan['scheduler_patience'],threshold=plan['min_delta'],threshold_mode='abs',min_lr=plan['min_learning_rate'])
    generator = torch.Generator(device=device).manual_seed(seed)
    history = []
    best, reference, waiting, best_epoch, updates = -1.,-1.,0,0,0
    started = time.perf_counter()
    record = {'model':data.model,'candidate':candidate,'training_seed':seed,'split_seed':plan['split_seed'],
              'status':'running','started_at_utc':utc_now(),'encoder_frozen':True,'test_evaluated':False,
              'trainable_parameters':sum(p.numel() for p in head.parameters()),'cache_identity':data.cache['identity'],
              'embedding_sha256':data.cache['sha256']}
    write_json(folder/'run.json',record)
    torch.save({'mean':mean.cpu(),'scale':scale.cpu(),'class_weights':weights.cpu()},folder/'preprocessing.pt')
    for epoch in range(1,plan['max_epochs']+1):
        epoch_started = time.perf_counter()
        head.train()
        permutation = torch.randperm(len(data.train_y),generator=generator,device=device)
        loss_sum, denominator, correct = [torch.zeros((),device=device) for _ in range(3)]
        for start in range(0,len(permutation),candidate['batch_size']):
            indices = permutation[start:start+candidate['batch_size']]
            labels = data.train_y[indices]
            logits = head(x_train[indices])
            loss = nn.functional.cross_entropy(logits,labels,weight=weights)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite classifier loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            weight_sum = weights[labels].sum()
            loss_sum += loss.detach()*weight_sum
            denominator += weight_sum
            correct += (logits.detach().argmax(1)==labels).sum()
            updates += 1
        validation,_ = evaluate(head,x_val,data.validation_y)
        score = validation['macro_f1']
        lr = optimizer.param_groups[0]['lr']
        scheduler.step(score)
        stats = {'epoch':epoch,'train_loss':float(loss_sum/denominator),'train_accuracy':float(correct/len(data.train_y)),
                 'validation_loss':validation['loss'],'validation_accuracy':validation['accuracy'],
                 'validation_macro_f1':score,'learning_rate':lr,'updates':updates,
                 'elapsed_seconds':time.perf_counter()-epoch_started}
        history.append(stats)
        if score > best:
            best,best_epoch = score,epoch
            torch.save(head.state_dict(),folder/'best_head.pt')
            write_json(folder/'best_validation_metrics.json',validation)
        if score > reference + plan['min_delta']:
            reference,waiting = score,0
        else:
            waiting += 1
        write_json(folder/'history.json',history)
        if epoch==1 or epoch%5==0 or waiting>=plan['early_stopping_patience']:
            print(f'{data.model} {candidate_id(candidate)} seed={seed} epoch={epoch} val_F1={score:.4f} best={best:.4f} lr={lr:.6f}',flush=True)
        if waiting>=plan['early_stopping_patience']:
            break
    record.update(status='complete',completed_at_utc=utc_now(),elapsed_seconds=time.perf_counter()-started,
                  epochs_run=len(history),best_epoch=best_epoch,best_validation_macro_f1=best,
                  optimizer_updates=updates,stop_reason='early_stopping' if waiting>=plan['early_stopping_patience'] else 'max_epochs',
                  head_sha256=sha256(folder/'best_head.pt'),preprocessing_sha256=sha256(folder/'preprocessing.pt'))
    write_json(folder/'run.json',record)
    return record


def rank_finalists(records):
    """Ranking accepts validation records, never test metrics."""
    for group in records:
        if any(r.get('test_evaluated') for r in group['runs']):
            raise ValueError('Test-exposed records cannot select a configuration')
        scores = [r['best_validation_macro_f1'] for r in group['runs']]
        group['mean_validation_macro_f1'] = float(np.mean(scores))
        group['std_validation_macro_f1'] = float(np.std(scores,ddof=1)) if len(scores)>1 else 0.
    return sorted(records,key=lambda r:(-r['mean_validation_macro_f1'],r['candidate_id']))


def evaluate_final_run(folder, model, lock, cache):
    """This is the only phase loading test vectors/labels."""
    folder = Path(folder)
    record = json.loads((folder/'run.json').read_text())
    selected = lock['selected'][model]
    if record['candidate'] != selected['candidate'] or record['training_seed'] not in lock['final_seeds']:
        raise ValueError('Final evaluation does not match the locked selection')
    if sha256(folder/'best_head.pt')!=record['head_sha256'] or sha256(folder/'preprocessing.pt')!=record['preprocessing_sha256']:
        raise ValueError('Final checkpoint or training-fitted scaler changed')
    if cache['identity'] != record['cache_identity'] or cache['sha256'] != record['embedding_sha256']:
        raise ValueError('Final evaluation embeddings differ from training')
    scenario = cache['identity']['scenario']
    config = load_config(PROJECT_ROOT/'configs/linear_probe'/f'{model}_{scenario}.json')
    device = device_for(config)
    _,path,_ = paths_for(config)
    array = np.load(path,mmap_mode='r',allow_pickle=False)
    rows = read_manifest(scenario)
    indices = [i for i,row in enumerate(rows) if row['split']=='test']
    x = torch.from_numpy(array[indices].copy()).to(device)
    y = torch.tensor([int(rows[i]['label']) for i in indices],device=device)
    prep = torch.load(folder/'preprocessing.pt',map_location=device,weights_only=True)
    head = nn.Linear(768,13).to(device).eval()
    head.load_state_dict(torch.load(folder/'best_head.pt',map_location=device,weights_only=True),strict=True)
    if (not all(bool(torch.isfinite(p).all()) for p in head.parameters())
            or not bool(torch.isfinite(prep['mean']).all()) or not bool(torch.isfinite(prep['scale']).all())
            or not bool((prep['scale']>0).all())):
        raise ValueError('Final head contains nonfinite parameters')
    metrics,predicted = evaluate(head,(x-prep['mean'])/prep['scale'],y)
    true = y.cpu().numpy()
    metrics['classification_report'] = classification_report(true,predicted,labels=list(range(13)),target_names=EFFECTS,output_dict=True,zero_division=0)
    if int(np.asarray(metrics['confusion_matrix']).sum()) != len(indices):
        raise ValueError('Test coverage mismatch')
    write_json(folder/'test_metrics.json',metrics)
    np.savez_compressed(folder/'test_predictions.npz',indices=np.asarray(indices),true=true,predicted=predicted)
    record.update(test_evaluated=True,test_evaluated_at_utc=utc_now(),selection_lock_sha256=lock['sha256'])
    write_json(folder/'run.json',record)
    return metrics


def run_consolidation(plan_path, *, allow_training=False, on_session_started=None):
    if not allow_training:
        raise RuntimeError('Consolidation disabled; explicit authorization is required')
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    started = time.perf_counter()
    scenario = plan['scenario']
    session = PROJECT_ROOT/'results/transfer_learning'/scenario/('consolidation_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    session.mkdir(parents=True,exist_ok=False)
    snapshot = {'protocol':plan,'protocol_sha256':sha256(plan_path),'implementation_sha256':sha256(Path(__file__)),
                'manifest_sha256':sha256(manifest_path(scenario)),'started_at_utc':utc_now(),
                'hardware':{'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                            'torch':torch.__version__,'device':'cuda' if torch.cuda.is_available() else 'cpu'},
                'status':'selection','test_used_for_selection':False}
    write_json(session/'protocol.json',snapshot)
    if on_session_started is not None:
        on_session_started(session)
    print('CONSOLIDATION_SESSION',session,flush=True)
    selection = {}
    for model in plan['models']:
        data = load_selection_data(model,plan)
        search = []
        for candidate in candidate_grid(plan):
            ident = candidate_id(candidate)
            folder = session/model/'search'/ident/f'seed_{plan["search_seed"]}'
            record = fit_candidate(data,candidate,plan['search_seed'],plan,folder,allow_training=True)
            record['folder'] = folder.relative_to(PROJECT_ROOT).as_posix()
            search.append({'candidate_id':ident,'candidate':candidate,'runs':[record]})
            write_json(session/model/'search_results.json',rank_finalists(search))
        finalists = rank_finalists(search)[:plan['finalists_per_model']]
        for finalist in finalists:
            for seed in plan['selection_seeds']:
                if seed==plan['search_seed']:
                    continue
                folder = session/model/'selection'/finalist['candidate_id']/f'seed_{seed}'
                record = fit_candidate(data,finalist['candidate'],seed,plan,folder,allow_training=True)
                record['folder'] = folder.relative_to(PROJECT_ROOT).as_posix()
                finalist['runs'].append(record)
        ranked = rank_finalists(finalists)
        write_json(session/model/'finalist_ranking.json',ranked)
        selection[model] = ranked[0]
        del data
    lock = {'locked_at_utc':utc_now(),'selected':selection,'final_seeds':plan['final_seeds'],
            'criterion':'highest mean validation macro F1 across selection seeds; ties by candidate ID',
            'test_used_for_selection':False,'protocol_sha256':snapshot['protocol_sha256']}
    write_json(session/'selection_locked.json',lock)
    lock['sha256'] = sha256(session/'selection_locked.json')
    snapshot.update(status='confirmation_training',selection_lock_sha256=lock['sha256'])
    write_json(session/'protocol.json',snapshot)
    final_records = {}
    for model in plan['models']:
        data = load_selection_data(model,plan)
        chosen = selection[model]['candidate']
        final_records[model] = []
        for seed in plan['final_seeds']:
            folder = session/model/'final'/f'seed_{seed}'
            record = fit_candidate(data,chosen,seed,plan,folder,allow_training=True)
            record['folder'] = folder.relative_to(PROJECT_ROOT).as_posix()
            final_records[model].append(record)
        del data
    write_json(session/'all_confirmation_heads_locked.json',final_records)
    snapshot.update(status='final_evaluation')
    write_json(session/'protocol.json',snapshot)
    for model,records in final_records.items():
        cache = validate_cache(load_config(PROJECT_ROOT/'configs/linear_probe'/f'{model}_{scenario}.json'))
        for record in records:
            folder = PROJECT_ROOT/record['folder']
            metrics = evaluate_final_run(folder,model,lock,cache)
            record['test_metrics'] = metrics
            record['test_evaluated'] = True
            print('FINAL_TEST',model,record['training_seed'],metrics['accuracy'],metrics['macro_f1'],flush=True)
    summary = {'session':session.relative_to(PROJECT_ROOT).as_posix(),'protocol':plan,'selection':selection,
               'final_runs':final_records,'selection_lock_sha256':lock['sha256'],'completed_at_utc':utc_now(),
               'elapsed_seconds':time.perf_counter()-started}
    for model,records in final_records.items():
        summary.setdefault('aggregate',{})[model] = {}
        for key in ('accuracy','macro_f1','macro_precision','macro_recall'):
            values = [r['test_metrics'][key] for r in records]
            summary['aggregate'][model][key] = {'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)),
                                               'values':values}
    write_json(session/'summary.json',summary)
    snapshot.update(status='complete',completed_at_utc=utc_now())
    write_json(session/'protocol.json',snapshot)
    print('CONSOLIDATION_COMPLETE',session,flush=True)
    return session,summary
