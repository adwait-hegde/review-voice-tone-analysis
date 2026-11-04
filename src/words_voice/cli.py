import argparse
import json
from pathlib import Path
from .data import audit, read_json, write_json


def main():
    p = argparse.ArgumentParser(description='Words vs. Voice: reproducible review analysis')
    p.add_argument('--config',default='configs/default.json')
    sub = p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit'); a.add_argument('--data',default='wav'); a.add_argument('--labels',default='configs/labels.json'); a.add_argument('--out',default='artifacts/data')
    s=sub.add_parser('split'); s.add_argument('--manifest',default='artifacts/data/manifest.json'); s.add_argument('--out',default='artifacts/data/splits.json'); s.add_argument('--holdout',choices=['source','voice','speaker'],default='source')
    for name in ('train','evaluate'):
        a=sub.add_parser(name)
        a.add_argument('--manifest',default='artifacts/data/manifest.json'); a.add_argument('--splits',default='artifacts/data/splits.json')
        a.add_argument('--out',required=True)
        if name=='train':
            a.add_argument('--model',choices=['majority','tfidf','mfcc','neural_text','neural_audio','fusion'],required=True)
            a.add_argument('--task',choices=['review','emotion'],default='review'); a.add_argument('--seed',type=int)
            a.add_argument('--smoke',action='store_true'); a.add_argument('--allow-download',action='store_true')
        else:
            a.add_argument('--checkpoint',required=True); a.add_argument('--partition',choices=['validation','test'],default='test')
            a.add_argument('--noise-snr',type=float); a.add_argument('--transcripts')
    a=sub.add_parser('infer'); a.add_argument('audio'); a.add_argument('--checkpoint',action='append',default=[],help='NAME=PATH, repeat for comparison'); a.add_argument('--emotion-checkpoint'); a.add_argument('--text'); a.add_argument('--asr-model',default='tiny.en'); a.add_argument('--allow-download',action='store_true'); a.add_argument('--regions-config'); a.add_argument('--out',default='artifacts/inference.json')
    a=sub.add_parser('transcribe'); a.add_argument('--manifest',default='artifacts/data/manifest.json'); a.add_argument('--out',required=True); a.add_argument('--asr-model',default='tiny.en'); a.add_argument('--allow-download',action='store_true')
    a=sub.add_parser('suite'); a.add_argument('--models',nargs='+',default=['majority','tfidf','mfcc']); a.add_argument('--task',default='review',choices=['review','emotion']); a.add_argument('--manifest',default='artifacts/data/manifest.json'); a.add_argument('--splits',default='artifacts/data/splits.json'); a.add_argument('--out',default='artifacts/experiments'); a.add_argument('--allow-download',action='store_true'); a.add_argument('--final-test',action='store_true')
    a=sub.add_parser('challenge-template'); a.add_argument('--out',default='artifacts/challenge_template.json')
    a=sub.add_parser('challenge-evaluate'); a.add_argument('--annotations',required=True); a.add_argument('--checkpoint',required=True); a.add_argument('--out',required=True)
    a=sub.add_parser('temporal-evaluate'); a.add_argument('--annotations',required=True); a.add_argument('--checkpoint',required=True); a.add_argument('--out',required=True); a.add_argument('--regions-config')
    a=sub.add_parser('tune-regions'); a.add_argument('--annotations',required=True); a.add_argument('--checkpoint',required=True); a.add_argument('--out',required=True)
    a=sub.add_parser('shortcuts'); a.add_argument('--manifest',default='artifacts/data/manifest.json'); a.add_argument('--splits',default='artifacts/data/splits.json'); a.add_argument('--out',default='artifacts/shortcuts.json')
    args=p.parse_args(); cfg=read_json(args.config)
    try:
        if args.command=='audit':
            result=audit(args.data,args.labels,args.out)
            print(json.dumps(result,indent=2))
            if result['invalid'] or result['missing_pairs']:
                raise ValueError('Audit found invalid/missing pairs; see audit.json. Exclusions require review.')
        elif args.command=='split':
            from .splits import create_splits
            result=create_splits(args.manifest,args.out,cfg,args.holdout)
            print(json.dumps({k:result[k] for k in ('counts','group_count','metadata_limitations')},indent=2))
        elif args.command=='train':
            from .experiment import train
            if args.seed is not None: cfg['seed']=args.seed
            b=train(args.manifest,args.splits,args.model,args.task,args.out,cfg,args.allow_download,args.smoke)
            print(json.dumps({'checkpoint':args.out,'smoke_only':b['smoke_only'],'training_seconds':b['training_seconds'],'history':b['history']},indent=2))
        elif args.command=='evaluate':
            from .experiment import evaluate
            r=evaluate(args.manifest,args.splits,args.checkpoint,args.out,args.partition,args.noise_snr,args.transcripts)
            print(json.dumps({k:v['macro_f1'] for k,v in r['metrics'].items()},indent=2))
        elif args.command=='infer':
            from .inference import infer
            r=infer(args.audio,dict(c.split('=',1) for c in args.checkpoint),args.emotion_checkpoint,args.text,args.asr_model,args.allow_download,read_json(args.regions_config) if args.regions_config else None,cfg.get('asr_revision'))
            write_json(args.out,r); print(json.dumps(r,indent=2))
        elif args.command=='transcribe':
            from .inference import transcribe
            result={}
            for r in read_json(args.manifest)['records']:
                result[r['id']]=transcribe(r['audio'],args.asr_model,args.allow_download,cfg.get('asr_revision'))
                write_json(args.out,result)
        elif args.command=='suite':
            from .experiment import train,evaluate
            results=[]
            for seed in cfg['seeds']:
                for model in args.models:
                    run=Path(args.out)/f'{args.task}-{model}-seed{seed}'
                    path=run/'model.joblib'
                    train(args.manifest,args.splits,model,args.task,path,{**cfg,'seed':seed},args.allow_download)
                    results.append(evaluate(args.manifest,args.splits,path,run/('test' if args.final_test else 'validation'),'test' if args.final_test else 'validation'))
            if len({r['cohort_sha256'] for r in results})!=1:
                raise ValueError('Compared models did not use the same evaluation cohort')
            write_json(Path(args.out)/'summary.json',results)
        elif args.command in ('challenge-template','challenge-evaluate','temporal-evaluate','tune-regions','shortcuts'):
            from . import challenge
            if args.command=='challenge-template': write_json(args.out,challenge.template())
            elif args.command=='challenge-evaluate': challenge.evaluate_challenge(args.annotations,args.checkpoint,args.out)
            elif args.command=='temporal-evaluate': challenge.temporal_evaluate(args.annotations,args.checkpoint,args.out,read_json(args.regions_config) if args.regions_config else None)
            elif args.command=='tune-regions': challenge.tune(args.annotations,args.checkpoint,args.out)
            else: challenge.shortcuts(args.manifest,args.splits,args.out,cfg)
    except (ValueError,OSError,ImportError,RuntimeError) as exc:
        p.exit(2,f'Error: {exc}\n')

if __name__=='__main__': main()
