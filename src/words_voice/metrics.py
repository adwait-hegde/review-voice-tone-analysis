import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(y_true, y_pred, labels, groups=None, bootstraps=1000, seed=42):
    if not len(y_true):
        raise ValueError('No labelled evaluation examples')
    result = {'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
              'accuracy': float(accuracy_score(y_true, y_pred)),
              'per_class': classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
              'labels': labels, 'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(), 'n': len(y_true)}
    if groups is not None and bootstraps:
        unique = sorted(set(groups)); rng = np.random.default_rng(seed)
        group_index = {g: np.flatnonzero(np.array(groups) == g) for g in unique}
        values = []
        for _ in range(bootstraps):
            ix = np.concatenate([group_index[g] for g in rng.choice(unique, size=len(unique), replace=True)])
            values.append(f1_score(np.array(y_true)[ix], np.array(y_pred)[ix], labels=labels, average='macro', zero_division=0))
        result['macro_f1_group_bootstrap_95ci'] = np.quantile(values, [.025,.975]).tolist()
        result['independent_groups'] = len(unique)
        result['ci_caveat'] = 'Percentile group bootstrap; label set fixed even if resamples omit classes. Few groups yield unstable intervals.'
    return result
