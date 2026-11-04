import copy
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from words_voice.data import audit, labels_for, read_json, write_json, segment_bounds
from words_voice.splits import create_splits, load_partition, make_groups
from words_voice.features import audio_load, masked_mean, FeatureStore, Encoders
from words_voice.models import masked_task_loss, pad_audio
from words_voice.experiment import train, evaluate, load_checkpoint, predict_features
from words_voice.inference import infer, postprocess, emotion_regions, color_transcript, transcript_html
from words_voice.metrics import classification_metrics

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def dataset(tmp_path):
    root=tmp_path/'wav'; root.mkdir()
    mapping=read_json(ROOT/'configs/labels.json')
    for i in range(32):
        prefix=['g-pos','g-neg','s-pos','s-neg'][i%4]
        name=f'{prefix}_{i}'
        y=.2*np.sin(2*np.pi*(100+i*20)*np.arange(3200)/16000)
        sf.write(root/(name+'.wav'),y,16000)
        # Explicit interval metadata is authoritative for the fixtures, not fixed-slot assumptions.
        text=f"{'excellent happy' if i%2==0 else 'awful broken'} {'honest plain' if i%4<2 else 'ironic kidding'} unique fixture {i}"
        write_json(root/(name+'.json'),{'id':name,'duration':.2,'sample_rate':16000,
                   'segments':[{'idx':0,'start':0,'end':.2,'text':text,'emotion':['ang','sad','hap','neu'][i%4]}]})
    mp=tmp_path/'labels.json'; write_json(mp,mapping)
    out=tmp_path/'data'; audit(root,mp,out)
    cfg=read_json(ROOT/'configs/default.json'); cfg.update({'near_duplicate_threshold':.9999,'bootstrap_samples':20,'epochs':2,'hidden_size':8})
    create_splits(out/'manifest.json',out/'splits.json',cfg)
    return root,out,cfg


def test_audit_pairing_schema_and_timing(dataset,tmp_path):
    root,out,cfg=dataset
    report=read_json(out/'audit.json')
    assert report['recording_count']==32 and not report['invalid']
    sf.write(root/'orphan.wav',np.zeros(100),16000)
    (root/'broken.json').write_text('{')
    sf.write(root/'broken.wav',np.zeros(100),16000)
    r=audit(root,tmp_path/'labels.json',out)
    assert r['missing_pairs']==['orphan'] and len(r['invalid'])==1
    mapping=read_json(tmp_path/'labels.json')
    assert segment_bounds({'idx':1,'speech_seconds':2},30,3,mapping)[:2]==(10,20)
    mapping['timing']['mode']='unresolved'
    assert segment_bounds({'idx':1},30,3,mapping)[:2]==(None,None)
    assert segment_bounds({'idx':0},8,1,mapping)[:2]==(0,8)


def test_verified_labels():
    m=read_json(ROOT/'configs/labels.json')
    assert labels_for('s-neg_3_hap_r1',m)=={'sentiment':'negative','sarcasm':'sarcastic'}
    assert labels_for('unknown',m)['sentiment'] is None
    m['review_prefixes']['s-neg']['verified']=False
    with pytest.raises(ValueError): labels_for('s-neg_1',m)


def test_splits_disjoint_and_fingerprint(dataset):
    _,out,cfg=dataset
    rows,_,s=load_partition(out/'manifest.json',out/'splits.json')
    groups={k:{r['group'] for r in rows if r['split']==k} for k in ['train','validation','test']}
    assert not(groups['train']&groups['test'] or groups['train']&groups['validation'] or groups['test']&groups['validation'])
    rows[1]['segments'][0]['text']=rows[0]['segments'][0]['text']
    g,_=make_groups(rows,.9999)
    assert g[0]==g[1]
    m=read_json(out/'manifest.json'); m['records'][0]['text']='modified'; write_json(out/'manifest.json',m)
    with pytest.raises(ValueError,match='Manifest changed'): load_partition(out/'manifest.json',out/'splits.json')


def test_slice_resampling_and_mask(tmp_path):
    sf.write(tmp_path/'x.wav',np.r_[np.ones(8000)*.1,np.ones(8000)*.4],8000)
    y=audio_load(tmp_path/'x.wav',16000,1,2)
    assert len(y)==16000 and y[200:-200].mean()==pytest.approx(.4,abs=.001)
    x,mask,lengths=pad_audio([np.ones(2),np.ones(4)*2])
    assert mask.tolist()==[[True,True,False,False],[True]*4]
    h=torch.tensor([[[2.],[4.],[999.]],[[1.],[3.],[5.]]])
    assert masked_mean(h,torch.tensor([[1,1,0],[1,1,1]])).tolist()==[[3.],[3.]]


def test_missing_task_loss():
    logits={'sentiment':torch.tensor([[1.,2.],[2.,1.]],requires_grad=True), 'sarcasm':torch.randn(2,2,requires_grad=True)}
    loss=masked_task_loss(logits,{'sentiment':torch.tensor([1,-1]),'sarcasm':torch.tensor([-1,-1])})
    expected=torch.nn.functional.cross_entropy(logits['sentiment'][:1],torch.tensor([1]))
    assert torch.allclose(loss,expected)
    loss.backward(); assert logits['sentiment'].grad[1].abs().sum()==0
    zero=masked_task_loss(logits,{t:torch.tensor([-1,-1]) for t in logits})
    assert zero.item()==0


def test_metrics_known():
    r=classification_metrics(['a','a','b','b'],['a','b','b','b'],['a','b'],['x','x','y','z'],20)
    assert r['accuracy']==.75 and r['macro_f1']==pytest.approx((2/3+.8)/2)
    assert r['confusion_matrix']==[[1,1],[0,2]]
    assert r['independent_groups']==3


def test_checkpoint_and_inference(dataset,tmp_path):
    root,out,cfg=dataset
    ck=tmp_path/'text.joblib'
    b=train(out/'manifest.json',out/'splits.json','tfidf','review',ck,cfg)
    assert load_checkpoint(ck)['kind']=='tfidf'
    result=evaluate(out/'manifest.json',out/'splits.json',ck,tmp_path/'evaluation')
    assert result['metrics']['combined']['n']>0
    audio=next(root.glob('*.wav'))
    r=infer(audio,{'text':str(ck),'missing':str(tmp_path/'absent')},text='excellent happy honest plain')
    assert r['predictions']['text']['sentiment']['label']=='positive'
    assert any('Checkpoint missing' in w for w in r['warnings'])
    assert r['colored_transcript']==[]  # no invented timing for supplied plain text


def test_dynamic_regions_silence_and_coloring(dataset,tmp_path):
    root,out,cfg=dataset
    ck=tmp_path/'emotion.joblib'
    b=train(out/'manifest.json',out/'splits.json','majority','emotion',ck,cfg)
    y=np.r_[np.ones(16000)*.2,np.zeros(16000),np.ones(16000)*.2].astype('float32')
    r=emotion_regions(b,y,FeatureStore(cfg,tmp_path/'cache'),{'hop_seconds':.25,'window_seconds':.5})
    assert r[0]['start']==0 and r[-1]['end']==3
    assert any(x['emotion']=='silence' for x in r)
    transcript={'segments':[{'start':0,'end':1,'text':'<script>','words':[]}]}
    units=color_transcript(transcript,r)
    assert '<script>' not in transcript_html(units)
    assert '&lt;script&gt;' in transcript_html(units)
    labels,_=postprocess(np.array([[.9,.1],[.1,.9],[.9,.1]]),np.zeros(3,dtype=bool),.5,{'smooth_windows':1,'min_region_seconds':1})
    assert len(set(labels))==1


def test_tiny_local_encoders_and_neural_roundtrip(dataset,tmp_path,monkeypatch):
    """Random tiny local models test engineering only; NEVER scientific pretrained results."""
    from transformers import BertConfig,BertModel,BertTokenizer,Wav2Vec2Config,Wav2Vec2Model,Wav2Vec2FeatureExtractor
    td=tmp_path/'tiny_text'; ad=tmp_path/'tiny_audio'; td.mkdir(); ad.mkdir()
    (td/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nexcellent\nawful\n')
    tokenizer=BertTokenizer(vocab_file=str(td/'vocab.txt')); tokenizer.save_pretrained(td)
    BertModel(BertConfig(vocab_size=tokenizer.vocab_size,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,max_position_embeddings=512)).save_pretrained(td)
    config=Wav2Vec2Config(hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,conv_dim=(8,8),conv_stride=(5,2),conv_kernel=(10,3),num_conv_pos_embedding_groups=2,num_conv_pos_embeddings=8,mask_time_prob=0.)
    Wav2Vec2Model(config).save_pretrained(ad)
    Wav2Vec2FeatureExtractor(sampling_rate=16000,return_attention_mask=False).save_pretrained(ad)
    root,out,cfg=dataset
    cfg={**cfg,'text_encoder':str(td),'audio_encoder':str(ad),'audio_chunk_seconds':.1,'epochs':1}
    enc=Encoders(cfg,['audio','text'])
    assert enc.text('excellent').shape==(8,)
    assert enc.audio(np.ones(5000,dtype='float32')).shape==(8,)
    # Isolate cache from repository test output.
    monkeypatch.chdir(tmp_path)
    ck=tmp_path/'fusion.joblib'
    b=train(out/'manifest.json',out/'splits.json','fusion','review',ck,cfg,smoke=True)
    assert b['smoke_only']
    r=infer(next(root.glob('*.wav')),{'fusion':str(ck)},text='excellent')
    assert 'combined' in r['predictions']['fusion']


def test_unknown_emotion_duration_and_audio_duplicates(dataset,tmp_path):
    root,out,cfg=dataset
    files=sorted(root.glob('*.json'))
    d=read_json(files[0]); d['segments'][0]['emotion']='unknown'; write_json(files[0],d)
    d=read_json(files[1]); d['duration']=999; write_json(files[1],d)
    files[3].with_suffix('.wav').write_bytes(files[2].with_suffix('.wav').read_bytes())
    report=audit(root,tmp_path/'labels.json',out)
    assert len(report['invalid'])==2
    assert len(report['duplicate_audio'])==1


def test_partial_label_training_and_authoritative_leak(dataset,tmp_path):
    root,out,cfg=dataset
    m=read_json(out/'manifest.json')
    for i,r in enumerate(m['records']):
        if i%5==0: r['sentiment']=None
    write_json(out/'manifest.json',m)
    create_splits(out/'manifest.json',out/'splits.json',cfg)
    train(out/'manifest.json',out/'splits.json','tfidf','review',tmp_path/'partial.joblib',cfg)
    rows=m['records']; rows[1]['segments'][0]['text']=rows[0]['segments'][0]['text']
    for i,r in enumerate(rows): r['split']=['train','validation','test'][i%3]
    write_json(out/'manifest.json',m)
    with pytest.raises(ValueError,match='leak'):
        create_splits(out/'manifest.json',out/'splits.json',cfg)


def test_demo_starts_without_models():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=20)
    assert not app.exception
    assert app.title[0].value=='Words vs. Voice'


def test_manifest_portability_and_schema(dataset):
    from words_voice.splits import fingerprint
    import jsonschema
    _,out,_=dataset
    m=read_json(out/'manifest.json')
    jsonschema.validate(m,read_json(ROOT/'docs/manifest.schema.json'))
    relocated=copy.deepcopy(m['records'])
    for r in relocated:
        r['audio']='/other/machine/'+Path(r['audio']).name
        r['metadata']='/other/machine/'+Path(r['metadata']).name
    assert fingerprint(relocated)==fingerprint(m['records'])


def test_human_challenge_refuses_blank_template(tmp_path):
    from words_voice.challenge import template,prepare
    path=tmp_path/'challenge.json';write_json(path,template())
    with pytest.raises(ValueError,match='requires consent'):
        prepare(path,{'config':read_json(ROOT/'configs/default.json')})


def test_asr_receives_waveform_and_preserves_timestamps(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import faster_whisper
    from words_voice.inference import transcribe
    path=tmp_path/'wave.wav';sf.write(path,np.zeros(16000),16000)
    class FakeASR:
        def __init__(self,*args,**kwargs): assert kwargs['local_files_only']
        def transcribe(self,audio,**kwargs):
            assert isinstance(audio,np.ndarray) and audio.shape==(16000,)
            assert kwargs['word_timestamps']
            w=SimpleNamespace(start=.2,end=.8,word='test')
            return iter([SimpleNamespace(start=.1,end=.9,text='test',words=[w])]),SimpleNamespace(language='en')
    monkeypatch.setattr(faster_whisper,'WhisperModel',FakeASR)
    result=transcribe(path)
    assert result['segments'][0]['words'][0]['start']==.2
