"""Human challenge protocol, temporal evaluation and artifact-only diagnostics."""
import itertools
from pathlib import Path
import numpy as np
import soundfile as sf
from .data import digest, read_json, write_json
from .experiment import load_checkpoint, evaluate
from .features import audio_load, FeatureStore
from .inference import emotion_regions
from .metrics import classification_metrics
from .splits import fingerprint, make_groups, load_partition


def template():
    return {'schema_version':1, 'purpose':'Fill with consented human recordings; do not reuse train scripts. Nulls mean unannotated.',
            'records':[{'id':'human-001','audio':'/absolute/path/to/review.wav','origin':'human',
                        'speaker_id':None,'source_id':None,'script_id':None,'template_id':None,
                        'partition':'test','text':None,'sentiment':None,'sarcasm':None,
                        'temporal_ground_truth':False,'regions':[],
                        'annotators':[],'consent_obtained':False}]}


def prepare(path, bundle, partition=None):
    records=read_json(path)['records']
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate challenge recording IDs')
    selected=[]
    for r in records:
        if partition and r.get('partition')!=partition: continue
        if r.get('origin')!='human' or not r.get('consent_obtained') or not r.get('speaker_id') or not r.get('source_id'):
            raise ValueError('Human challenge requires consent, speaker_id, source_id and origin=human')
        if not isinstance(r.get('text'),str) or not r['text'].strip():
            raise ValueError('Challenge requires a human supplied transcript')
        if r.get('sentiment') not in ('positive','negative',None) or r.get('sarcasm') not in ('genuine','sarcastic',None):
            raise ValueError('Invalid challenge review labels')
        ap=Path(r['audio'])
        if not ap.is_absolute(): ap=Path(path).resolve().parent/ap
        y,sr=sf.read(ap,always_2d=True)
        duration=len(y)/sr
        row={**r,'audio':str(ap),'duration':duration,'sample_rate':sr,'channels':y.shape[1],
             'audio_sha256':digest(ap),'pcm_sha256':digest(ap),'voice_id':None,
             'script_id':r.get('script_id'),'template_id':r.get('template_id'),
             'segments':[{'idx':0,'start':0.,'end':duration,'text':r['text'],'emotion':None}], 'split':'test'}
        selected.append(row)
    if not selected: raise ValueError('No eligible annotated challenge recordings')
    exposure=bundle.get('exposure_records',[])
    groups,_=make_groups(exposure+selected,bundle['config']['near_duplicate_threshold'])
    old=set(groups[:len(exposure)])
    if old & set(groups[len(exposure):]):
        raise ValueError('Challenge shares a source or near/exact transcript with training/validation')
    human_groups,_=make_groups(selected,bundle['config']['near_duplicate_threshold'],'speaker')
    for r,g in zip(selected,human_groups):
        r['group']=g
    return selected


def evaluate_challenge(path,checkpoint,out):
    b=load_checkpoint(checkpoint)
    if b['task']!='review': raise ValueError('Use temporal-evaluate for human perceived emotion')
    rows=prepare(path,b,'test')
    # Cluster repeated source AND speaker measurements for challenge bootstrap.
    groups,_=make_groups(rows,b['config']['near_duplicate_threshold'],'speaker')
    for r,g in zip(rows,groups): r['group']=g
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    m={'schema_version':1,'label_mapping':b['label_mapping'],'records':rows}
    s={'manifest_fingerprint':fingerprint(rows),'assignments':{r['id']:{'split':'test','group':r['group']} for r in rows}}
    write_json(out/'manifest.json',m); write_json(out/'splits.json',s)
    return evaluate(out/'manifest.json',out/'splits.json',checkpoint,out/'human_test',challenge=True)


def annotated_regions(r,classes):
    if not r.get('temporal_ground_truth') or len(r.get('annotators',[]))<2:
        raise ValueError('Boundary evaluation requires separate temporal ground truth and at least two annotators')
    regions=sorted(r['regions'],key=lambda x:x['start'])
    last=0.
    for q in regions:
        if abs(q['start']-last)>.001 or not q['start']<q['end']<=r['duration']+.001 or q['emotion'] not in classes+['silence']:
            raise ValueError('Temporal regions must cover audio contiguously with known labels (including silence)')
        last=q['end']
    if abs(last-r['duration'])>.01: raise ValueError('Temporal annotation must cover full recording')
    return regions


def temporal_scores(rows,b,overrides=None):
    classes=b['classes']['emotion']; labels=classes+['silence']
    results={}
    store=FeatureStore(b['config'])
    for fixed in (True,False):
        truth,pred,groups=[],[],[]
        matched=actual=proposed=0
        errors=[]
        for r in rows:
            gold=annotated_regions(r,classes)
            y=audio_load(r['audio'],b['config']['sample_rate'])
            regions=emotion_regions(b,y,store,overrides,fixed)
            for t in np.arange(.05,r['duration'],.1):
                truth.append(next(x['emotion'] for x in gold if x['start']<=t<x['end']))
                pred.append(next(x['emotion'] for x in regions if x['start']<=t<x['end']))
                groups.append(r['group'])
            def changes(rs):
                return [cur['start'] for prev,cur in zip(rs,rs[1:])
                        if prev['emotion'] != cur['emotion'] and 'silence' not in (prev['emotion'],cur['emotion'])]
            gt=changes(gold)
            pr=changes(regions)
            # Greedy nearest one-to-one boundary matching at prespecified 0.5-second tolerance.
            pairs=sorted((abs(a-c),i,j) for i,a in enumerate(gt) for j,c in enumerate(pr) if abs(a-c)<=.5)
            used_a,used_b=set(),set()
            for error,i,j in pairs:
                if i not in used_a and j not in used_b:
                    used_a.add(i); used_b.add(j); errors.append(error)
            matched+=len(used_a); actual+=len(gt); proposed+=len(pr)
        metrics=classification_metrics(truth,pred,labels,groups,b['config']['bootstrap_samples'],b['config']['seed'])
        metrics['boundary']={'tolerance_seconds':.5,'matched':matched,'actual':actual,'predicted':proposed,
                             'precision':matched/proposed if proposed else 0.,'recall':matched/actual if actual else 0.,
                             'f1':2*matched/(actual+proposed) if actual+proposed else 0.,
                             'mean_matched_error_seconds':float(np.mean(errors)) if errors else None}
        metrics['unit_note']='100ms bins for duration-weighted classification; bootstrap samples independent source groups, never bins/windows.'
        results['fixed' if fixed else 'dynamic']=metrics
    return results


def temporal_evaluate(path,checkpoint,out,overrides=None):
    b=load_checkpoint(checkpoint)
    if b['task']!='emotion': raise ValueError('Emotion checkpoint required')
    all_rows=prepare(path,b)
    memberships={}
    for r in all_rows:
        partition=r.get('partition')
        if partition not in ('validation','test'): raise ValueError('Specify validation or test partition')
        if r['group'] in memberships and memberships[r['group']] != partition:
            raise ValueError('Temporal validation/test share a speaker or source group')
        memberships[r['group']]=partition
    rows=[r for r in all_rows if r['partition']=='test']
    if not rows: raise ValueError('No temporal test recordings')
    write_json(out,{'origin':'human','partition':'test','metrics':temporal_scores(rows,b,overrides)})


def tune(path,checkpoint,out):
    b=load_checkpoint(checkpoint)
    if b['task']!='emotion': raise ValueError('Emotion checkpoint required')
    rows=prepare(path,b,'validation')
    all_records=read_json(path)['records']
    val_speakers={r['speaker_id'] for r in rows}
    if val_speakers & {r.get('speaker_id') for r in all_records if r.get('partition')=='test'}:
        raise ValueError('Temporal validation/test must be speaker disjoint')
    best,best_score=None,-1.
    for smooth,duration in itertools.product((1,3,5),(.5,1.,2.)):
        cfg={'smooth_windows':smooth,'min_region_seconds':duration}
        score=temporal_scores(rows,b,cfg)['dynamic']['macro_f1']
        if score>best_score: best,best_score=cfg,score
    write_json(out,{**best,'validation_macro_f1':best_score,'selection':'human validation only; frozen before test'})


def shortcuts(manifest,splits,out,cfg):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    rows,_,_=load_partition(manifest,splits)
    features=[]; table=[]
    for r in rows:
        y=audio_load(r['audio'],cfg['sample_rate']); sr=cfg['sample_rate']
        frame=np.array([np.sqrt(np.mean(y[i:i+320]**2)) for i in range(0,len(y),320)])
        silence=20*np.log10(frame+1e-10)<cfg['silence_db']
        boundaries=[abs(float(y[i]-y[i-1])) for i in range(10*sr,len(y),10*sr)]
        values=[float(silence.mean()),float(np.sqrt(np.mean(y*y))),float(np.max(abs(y))),float(np.mean(boundaries)) if boundaries else 0.,r['duration']]
        features.append(values); table.append({'id':r['id'],'split':r['split'],'silence_fraction':values[0],'rms':values[1],'peak':values[2],'slot_seam_jump':values[3],'duration':values[4]})
    x=np.array(features); metrics={}
    for task in ('sentiment','sarcasm'):
        tr=[i for i,r in enumerate(rows) if r['split']=='train' and r[task] is not None]
        va=[i for i,r in enumerate(rows) if r['split']=='validation' and r[task] is not None]
        if len({rows[i][task] for i in tr})<2: continue
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(x[tr],[rows[i][task] for i in tr])
        metrics[task]=classification_metrics([rows[i][task] for i in va],model.predict(x[va]),list(model.classes_),[rows[i]['group'] for i in va],cfg['bootstrap_samples'],cfg['seed'])
    write_json(out,{'purpose':'Artifact-only diagnostic, not a proposed sentiment model; validation partition only.',
                    'voice_identity_check':'Unavailable without voice IDs; cannot rule out voice shortcuts.',
                    'features':table,'validation_metrics':metrics})
