"""Evaluate already selected checkpoints on a common cohort; does not train/tune."""
import argparse
from pathlib import Path
from words_voice.experiment import evaluate
from words_voice.data import write_json

p=argparse.ArgumentParser()
p.add_argument('--partition',choices=['validation','test'],required=True)
p.add_argument('--models-dir',default='artifacts/models')
p.add_argument('--manifest',default='artifacts/data/manifest.json')
p.add_argument('--splits',default='artifacts/data/splits.json')
p.add_argument('--out',required=True)
a=p.parse_args()
results=[]
for name in ('majority','tfidf','mfcc','neural_text','neural_audio','fusion'):
    r=evaluate(a.manifest,a.splits,Path(a.models_dir)/(name+'.joblib'),Path(a.out)/name,a.partition)
    results.append(r)
    print(name,{k:round(v['macro_f1'],4) for k,v in r['metrics'].items()},flush=True)
if len({r['cohort_sha256'] for r in results})!=1:
    raise ValueError('Models evaluated on different cohorts')
write_json(Path(a.out)/'comparison.json',results)
