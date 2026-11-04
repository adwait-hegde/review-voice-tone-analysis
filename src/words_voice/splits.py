"""Group connected components BEFORE slicing or augmentation."""
import hashlib
import json
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GroupShuffleSplit
from .data import normalize, read_json, write_json


def fingerprint(records):
    portable = [{k:v for k,v in r.items() if k not in ('audio','metadata')} for r in records]
    return hashlib.sha256(json.dumps(portable, sort_keys=True).encode()).hexdigest()


def make_groups(records, threshold=.85, holdout='source'):
    n = len(records)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(j)] = find(i)
    seen = {}
    fields = ['source_id','script_id','template_id','pcm_sha256','audio_sha256']
    if holdout in ('voice', 'speaker'):
        field = holdout + '_id'
        if any(not r.get(field) for r in records):
            raise ValueError(f'{field} missing; cannot claim disjoint {holdout} evaluation')
        fields.append(field)
    texts, owners = [], []
    for i, r in enumerate(records):
        keys = [(k, r[k]) for k in fields if r.get(k)]
        keys += [('text', normalize(s['text'])) for s in r['segments']]
        keys += [('text', normalize(r['text']))]
        for key in keys:
            if key in seen:
                union(i, seen[key])
            seen[key] = i
        for text in [r['text']] + [s['text'] for s in r['segments']]:
            texts.append(normalize(text)); owners.append(i)
    # Stateless hashing is ONLY for split diagnostics/grouping, never a fitted predictive feature.
    matrix = HashingVectorizer(analyzer='char_wb', ngram_range=(3,5), n_features=2**18,
                               alternate_sign=False, norm='l2').transform(texts)
    near_pairs = []
    for start in range(0, len(texts), 128):
        sim = (matrix[start:start+128] @ matrix.T).tocoo()
        for a, b, score in zip(sim.row + start, sim.col, sim.data):
            i, j = owners[a], owners[b]
            if a < b and i != j and score >= threshold:
                union(i, j)
                near_pairs.append([records[i]['id'], records[j]['id'], float(score)])
    groups = [f'group-{find(i):04d}' for i in range(n)]
    return groups, near_pairs


def label_sets(records, indices):
    result = {task: set(records[i][task] for i in indices if records[i][task] is not None)
              for task in ('sentiment','sarcasm')}
    result['emotion'] = {s['emotion'] for i in indices for s in records[i]['segments']
                         if s['emotion'] is not None and s['start'] is not None}
    result['combined'] = {records[i]['sarcasm']+' '+records[i]['sentiment'] for i in indices
                          if records[i]['sarcasm'] is not None and records[i]['sentiment'] is not None}
    return result


def create_splits(manifest_path, out, cfg, holdout='source'):
    records = read_json(manifest_path)['records']
    if not records:
        raise ValueError('No valid recordings')
    groups, near = make_groups(records, cfg['near_duplicate_threshold'], holdout)
    all_labels = label_sets(records, range(len(records)))
    authoritative = [r.get('split') for r in records]
    if any(authoritative):
        if not all(x in ('train','validation','test') for x in authoritative):
            raise ValueError('Incomplete or invalid authoritative split metadata')
        assignment = authoritative
    else:
        fractions = cfg['split_fractions']
        if len(fractions) != 3 or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1):
            raise ValueError('Three positive split fractions summing to one required')
        assignment = None
        for attempt in range(200):
            rng = cfg['seed'] + attempt
            try:
                train, rest = next(GroupShuffleSplit(n_splits=1, train_size=fractions[0], random_state=rng).split(records, groups=groups))
                val_i, test_i = next(GroupShuffleSplit(n_splits=1, train_size=fractions[1]/sum(fractions[1:]), random_state=rng).split(rest, groups=np.array(groups)[rest]))
            except ValueError:
                continue
            parts = [train, rest[val_i], rest[test_i]]
            if all(label_sets(records, p) == all_labels for p in parts):
                assignment = [''] * len(records)
                for name, part in zip(('train','validation','test'), parts):
                    for i in part:
                        assignment[i] = name
                break
        if assignment is None:
            raise ValueError('Cannot form group-disjoint splits with every required class. Obtain more independent groups; do not weaken grouping silently.')
    memberships = {}
    for group, split in zip(groups, assignment):
        if group in memberships and memberships[group] != split:
            raise ValueError('Authoritative splits leak a source/duplicate group')
        memberships[group] = split
    for name in ('train','validation','test'):
        if label_sets(records, [i for i,x in enumerate(assignment) if x == name]) != all_labels:
            raise ValueError(f'{name} lacks required classes')
    result = {'schema_version': 1, 'manifest_fingerprint': fingerprint(records), 'seed': cfg['seed'],
              'grouping': f'connected source/script/template/exact PCM/exact and near transcript; holdout={holdout}',
              'near_duplicate_threshold': cfg['near_duplicate_threshold'], 'near_pairs': near,
              'metadata_limitations': 'Missing source/template identifiers cannot be reconstructed reliably. Similarity grouping is a conservative heuristic, not proof of template independence. Signature and repetition suffix not interpreted.',
              'counts': dict(Counter(assignment)), 'group_count': len(set(groups)),
              'assignments': {r['id']: {'split': s, 'group': g} for r,s,g in zip(records, assignment, groups)}}
    write_json(out, result)
    return result


def load_partition(manifest, splits):
    m, s = read_json(manifest), read_json(splits)
    if fingerprint(m['records']) != s['manifest_fingerprint']:
        raise ValueError('Manifest changed; rebuild splits and retrain')
    rows = [{**r, **s['assignments'][r['id']]} for r in m['records']]
    return rows, m, s
