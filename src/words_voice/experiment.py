"""Train on train, select on validation, evaluate only through an explicit command."""
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .data import read_json, write_json
from .features import FeatureStore
from .metrics import classification_metrics
from .splits import load_partition

KINDS = ['majority', 'tfidf', 'mfcc', 'neural_text', 'neural_audio', 'fusion']


def hardware():
    import torch
    return {'platform': platform.platform(), 'processor': platform.machine(),
            'torch': torch.__version__, 'cuda': torch.cuda.is_available(),
            'mps': torch.backends.mps.is_available(),
            'packages': {p: importlib.metadata.version(p) for p in ('numpy','scikit-learn','librosa','transformers')}}


def examples(records, task):
    if task == 'review':
        return records
    return [{**r, 'example_id': f"{r['id']}:{s['idx']}", 'emotion': s['emotion'],
             'start': s['start'], 'end': s['end']} for r in records for s in r['segments']
            if s['emotion'] is not None and s['start'] is not None]


def feature_matrix(rows, kind, store):
    if kind == 'majority':
        return np.ones((len(rows), 1), dtype='float32')
    if kind == 'tfidf':
        return [r['text'] for r in rows]
    modalities = {'mfcc': ['mfcc'], 'neural_text': ['text'], 'neural_audio': ['audio'], 'fusion': ['text','audio']}[kind]
    return np.stack([np.concatenate([store.get(r, m, r.get('start'), r.get('end')) for m in modalities]) for r in rows])


def eligible(rows, tasks):
    # Complete-case cohort is shared across every compared model, including majority.
    return [r for r in rows if all(r.get(t) is not None for t in tasks)]


def check_classes(rows, tasks, classes, split):
    if not rows:
        raise ValueError(f'No eligible examples in {split}')
    for t in tasks:
        found = {r[t] for r in rows if r.get(t) is not None}
        if found != set(classes[t]):
            raise ValueError(f'{split} task {t} needs {classes[t]}, found {sorted(found)}')


def train(manifest, splits, kind, task, out, cfg, allow_download=False, smoke=False):
    if kind not in KINDS or (task == 'emotion' and kind in ('tfidf','neural_text','fusion')):
        raise ValueError('Emotion requires an audio model (majority, mfcc or neural_audio)')
    records, m, split_meta = load_partition(manifest, splits)
    rows = examples(records, task)
    tasks = ['emotion'] if task == 'emotion' else ['sentiment','sarcasm']
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = eligible([r for r in rows if r['split'] == 'validation'], tasks)
    classes = {t: sorted({r[t] for r in train_rows if r.get(t) is not None}) for t in tasks}
    # Missing entire heads remain absent; known binary tasks must have both classes.
    classes = {t: labels for t,labels in classes.items() if labels}
    tasks = list(classes)
    if not tasks:
        raise ValueError('No verified labels for requested task; run emotion training independently')
    for t in tasks:
        if len(classes[t]) < 2:
            raise ValueError(f'Train task {t} has fewer than two classes')
    train_rows = [r for r in train_rows if any(r.get(t) is not None for t in tasks)]
    val_rows = eligible([r for r in rows if r['split'] == 'validation'], tasks)
    check_classes(train_rows, tasks, classes, 'train'); check_classes(val_rows, tasks, classes, 'validation')
    if smoke:
        # Only shrink training; preserve every class. Mark resulting checkpoints explicitly.
        rng = np.random.default_rng(cfg['seed'])
        ordered = list(rng.permutation(len(train_rows)))
        chosen = set(ordered[:min(24,len(ordered))])
        for t in tasks:
            for c in classes[t]:
                chosen.add(next(i for i,r in enumerate(train_rows) if r.get(t) == c))
        train_rows = [train_rows[i] for i in sorted(chosen)]
    store = FeatureStore(cfg, allow_download=allow_download)
    started = time.perf_counter()
    x_train, x_val = feature_matrix(train_rows,kind,store), feature_matrix(val_rows,kind,store)
    bundle = {'format_version': 1, 'kind': kind, 'task': task, 'classes': classes, 'config': cfg,
              'label_mapping': m['label_mapping'], 'manifest_fingerprint': split_meta['manifest_fingerprint'],
              'split_manifest': split_meta, 'smoke_only': smoke, 'hardware': hardware(),
              'trained_origins': sorted({r['origin'] for r in train_rows}),
              'exposure_records': [{k:r[k] for k in ('id','audio_sha256','pcm_sha256','text','segments','source_id','script_id','template_id')} for r in records if r['split'] in ('train','validation')],
              'score_note': 'Uncalibrated model scores, not reliable confidence estimates'}
    if kind == 'tfidf':
        transform = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True, max_features=30000)
        x_train = transform.fit_transform(x_train); x_val = transform.transform(x_val)
        bundle['transform'] = transform
    elif kind != 'majority':
        transform = StandardScaler().fit(x_train)
        x_train = transform.transform(x_train); x_val = transform.transform(x_val)
        bundle['transform'] = transform
    if not kind.startswith('neural') and kind != 'fusion':
        best, best_score, history = None, -1., []
        for c in ([1.] if kind == 'majority' else [.1,1.,10.]):
            models = {}
            for t in tasks:
                mask = np.array([r.get(t) is not None for r in train_rows])
                y = [r[t] for r in train_rows if r.get(t) is not None]
                model = DummyClassifier(strategy='most_frequent') if kind == 'majority' else LogisticRegression(C=c, max_iter=1000, class_weight='balanced', random_state=cfg['seed'])
                models[t] = model.fit(x_train[mask], y)
            score = np.mean([classification_metrics([r[t] for r in val_rows], models[t].predict(x_val), classes[t], bootstraps=0)['macro_f1'] for t in tasks])
            history.append({'C': c, 'validation_macro_f1': float(score)})
            if score > best_score:
                best, best_score = models, score
        bundle['models'] = best
    else:
        import torch
        from .models import MultiTaskMLP, masked_task_loss, seed_everything
        seed_everything(cfg['seed'])
        model = MultiTaskMLP(x_train.shape[1], classes, cfg['hidden_size']).to(cfg['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
        x = torch.tensor(x_train, dtype=torch.float32, device=cfg['device'])
        vx = torch.tensor(x_val, dtype=torch.float32, device=cfg['device'])
        ys = {t: torch.tensor([classes[t].index(r[t]) if r.get(t) is not None else -1 for r in train_rows], device=cfg['device']) for t in tasks}
        best_score, best, history = -1., None, []
        for epoch in range(2 if smoke else cfg['epochs']):
            model.train()
            for ix in torch.randperm(len(x), device=cfg['device']).split(cfg['batch_size']):
                optimizer.zero_grad(); loss = masked_task_loss(model(x[ix]), {t:y[ix] for t,y in ys.items()})
                loss.backward(); optimizer.step()
            model.eval()
            with torch.inference_mode():
                logits = model(vx)
            score = np.mean([classification_metrics([r[t] for r in val_rows], [classes[t][i] for i in logits[t].argmax(1).cpu().tolist()], classes[t], bootstraps=0)['macro_f1'] for t in tasks])
            history.append({'epoch': epoch+1, 'validation_macro_f1': float(score)})
            if score > best_score:
                best_score = score
                best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        bundle['state_dict'] = best; bundle['input_size'] = x_train.shape[1]
    bundle['history'] = history
    bundle['training_seconds'] = time.perf_counter()-started
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # Local trusted checkpoints only: sklearn estimators require pickle serialization.
    joblib.dump(bundle, out)
    write_json(str(out)+'.json', {k:v for k,v in bundle.items() if k not in ('models','transform','state_dict','split_manifest','exposure_records')})
    return bundle


def load_checkpoint(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Checkpoint missing: {path}. Train this model first.')
    bundle = joblib.load(path)
    if bundle.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    return bundle


def predict_features(bundle, features):
    if 'transform' in bundle:
        features = bundle['transform'].transform(features)
    if 'models' in bundle:
        return {t: model.predict_proba(features)[:, [list(model.classes_).index(c) for c in bundle['classes'][t]]] for t,model in bundle['models'].items()}
    import torch
    from .models import MultiTaskMLP
    model = MultiTaskMLP(bundle['input_size'], bundle['classes'], bundle['config']['hidden_size'])
    model.load_state_dict(bundle['state_dict']); model.eval()
    with torch.inference_mode():
        return {t: v.softmax(1).numpy() for t,v in model(torch.tensor(features,dtype=torch.float32)).items()}


def evaluate(manifest, splits, checkpoint, out, partition='test', noise_snr=None, transcripts=None, challenge=False):
    records, m, s = load_partition(manifest, splits)
    b = load_checkpoint(checkpoint)
    if not challenge and (b['manifest_fingerprint'] != s['manifest_fingerprint'] or b['split_manifest']['assignments'] != s['assignments']):
        raise ValueError('Checkpoint and evaluation splits differ')
    tasks = list(b['classes'])
    rows = eligible([r for r in examples(records,b['task']) if r['split']==partition], tasks)
    check_classes(rows, tasks, b['classes'], partition)
    if transcripts:
        automatic = read_json(transcripts)
        if any(r['id'] not in automatic for r in rows):
            raise ValueError('Automatic transcripts missing for shared evaluation cohort')
        rows = [{**r, 'text': automatic[r['id']]['text']} for r in rows]
    cfg = {**b['config'], 'noise_snr_db': noise_snr}
    started = time.perf_counter()
    scores = predict_features(b, feature_matrix(rows,b['kind'],FeatureStore(cfg)))
    elapsed = time.perf_counter()-started
    table = []
    for i,r in enumerate(rows):
        row = {k:r.get(k) for k in ('id','example_id','group','origin','voice_id','template_id','start','end')}
        for t,values in scores.items():
            row['true_'+t] = r[t]
            row['pred_'+t] = b['classes'][t][int(values[i].argmax())]
            for c,v in zip(b['classes'][t], values[i]):
                row[f'score_{t}_{c}'] = float(v)
        if 'sentiment' in scores and 'sarcasm' in scores:
            for prefix in ('true','pred'):
                row[prefix+'_combined'] = row[prefix+'_sarcasm']+' '+row[prefix+'_sentiment']
        table.append(row)
    metric_tasks = {**b['classes']}
    if table and 'true_combined' in table[0]:
        metric_tasks['combined'] = [a+' '+c for a in b['classes']['sarcasm'] for c in b['classes']['sentiment']]
    metrics = {t: classification_metrics([r['true_'+t] for r in table], [r['pred_'+t] for r in table], labels,
                [r['group'] for r in table], cfg['bootstrap_samples'],cfg['seed']) for t,labels in metric_tasks.items()}
    subgroup = {}
    for field in ('origin','voice_id','template_id'):
        for value in sorted({r[field] for r in table if r[field] is not None}):
            subset = [r for r in table if r[field] == value]
            subgroup[f'{field}={value}'] = {t:classification_metrics([r['true_'+t] for r in subset],[r['pred_'+t] for r in subset],labels,bootstraps=0) for t,labels in metric_tasks.items()}
    ids = [r.get('example_id') or r['id'] for r in table]
    summary = {'metrics': metrics,'subgroups':subgroup, 'partition':partition,'model':b['kind'],'task':b['task'],
               'cohort_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(), 'cohort_ids':ids,
               'elapsed_seconds_including_feature_loading':elapsed,'seconds_per_example':elapsed/len(rows),
               'cache_note':'Feature cache may be warm; this is not an uncached real-time latency claim.',
               'hardware':hardware(), 'smoke_only':b['smoke_only'], 'noise_snr_db':noise_snr,
               'transcript_condition':'automatic' if transcripts else 'supplied', 'score_note':b['score_note']}
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    write_json(out/'metrics.json',summary); write_json(out/'predictions.json',table)
    with (out/'errors.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    return summary
