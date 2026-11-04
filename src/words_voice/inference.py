import html
import time
import numpy as np
from scipy.ndimage import uniform_filter1d
from .experiment import load_checkpoint, predict_features, hardware
from .features import FeatureStore, audio_load

COLORS = {'ang':'#f2a6a6','sad':'#a9c8ef','neu':'#d8d8d8','hap':'#f8df83', 'silence':'#eeeeee'}


def transcribe(path, model='tiny.en', allow_download=False, revision=None):
    from faster_whisper import WhisperModel
    engine = WhisperModel(model, device='cpu', compute_type='int8', local_files_only=not allow_download, revision=revision)
    segments, info = engine.transcribe(audio_load(path, 16000), word_timestamps=True, vad_filter=True)
    output = []
    for s in segments:
        output.append({'start':s.start,'end':s.end,'text':s.text,
                       'words':[{'start':w.start,'end':w.end,'text':w.word} for w in (s.words or [])]})
    return {'text':' '.join(s['text'].strip() for s in output), 'segments':output,
            'source':'automatic', 'model':model, 'language':info.language}


def waveform_features(bundle, y, text, store):
    kind = bundle['kind']
    if kind == 'majority':
        return np.ones((1,1),dtype='float32')
    if kind == 'tfidf':
        return [text]
    modes = {'mfcc':['mfcc'],'neural_text':['text'],'neural_audio':['audio'],'fusion':['text','audio']}[kind]
    return np.concatenate([store.encode(m,text=text,y=y) for m in modes])[None,:]


def predict_review(bundle, y, text, store):
    if bundle['task'] != 'review':
        raise ValueError('Expected a review checkpoint')
    if bundle['kind'] in ('tfidf','neural_text','fusion') and not text.strip():
        raise ValueError('Text/fusion requires a supplied or automatic transcript')
    scores = predict_features(bundle, waveform_features(bundle,y,text,store))
    result = {t:{'label':bundle['classes'][t][v[0].argmax()],
                  'scores':dict(zip(bundle['classes'][t],map(float,v[0])))} for t,v in scores.items()}
    if 'sentiment' in result and 'sarcasm' in result:
        result['combined'] = result['sarcasm']['label'].capitalize()+' '+result['sentiment']['label']
    result['smoke_only'] = bundle['smoke_only']
    return result


def runs(labels):
    if not len(labels):
        return []
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1]
    ends = np.r_[starts[1:],len(labels)]
    return [(int(a),int(b),int(labels[a])) for a,b in zip(starts,ends)]


def postprocess(probabilities, silence, step, cfg):
    smoothed = uniform_filter1d(np.asarray(probabilities), size=max(1,int(cfg['smooth_windows'])), axis=0, mode='nearest')
    labels = smoothed.argmax(1)
    labels[silence] = -1
    # Reassign short emotion runs, never across a silence barrier. Iterate to stability.
    for _ in range(len(labels)):
        changed = False
        current = runs(labels)
        for i,(a,b,c) in enumerate(current):
            if c == -1 or (b-a)*step >= cfg['min_region_seconds']:
                continue
            neighbors = []
            if i > 0 and current[i-1][2] != -1:
                neighbors.append(current[i-1][2])
            if i+1 < len(current) and current[i+1][2] != -1:
                neighbors.append(current[i+1][2])
            if neighbors:
                selected = max(neighbors,key=lambda k:float(smoothed[a:b,k].mean()))
                labels[a:b] = selected; changed = True
                break
        if not changed:
            break
    return labels, smoothed


def emotion_regions(bundle, y, store, overrides=None, fixed=False):
    if bundle['task'] != 'emotion':
        raise ValueError('Expected an audio emotion checkpoint')
    cfg = {**bundle['config'], **(overrides or {})}
    sr, duration = cfg['sample_rate'], len(y)/cfg['sample_rate']
    width = 10. if fixed else float(cfg['window_seconds'])
    hop = width if fixed else float(cfg['hop_seconds'])
    if not (0 < hop <= width) or cfg['min_region_seconds'] < 0:
        raise ValueError('Require 0 < hop <= window and nonnegative minimum region duration')
    classes = bundle['classes']['emotion']
    if fixed:
        result = []
        for a in np.arange(0,duration,width):
            b = min(duration,a+width)
            p = predict_features(bundle,waveform_features(bundle,y[round(a*sr):round(b*sr)],'',store))['emotion'][0]
            result.append({'start':float(a),'end':float(b),'emotion':classes[int(p.argmax())], 'scores':dict(zip(classes,map(float,p)))})
        return result
    # Overlap-add window probabilities onto a nonoverlapping time grid; windows are not evaluation units.
    n = max(1,int(np.ceil(duration/hop)))
    total, weights = np.zeros((n,len(classes))), np.zeros(n)
    for a in np.arange(0,duration,hop):
        b = min(duration,a+width)
        p = predict_features(bundle,waveform_features(bundle,y[round(a*sr):max(round(a*sr)+1,round(b*sr))],'',store))['emotion'][0]
        for i in range(int(a/hop+1e-6),min(n,int(np.ceil(b/hop)))):
            overlap = max(0,min(b,(i+1)*hop)-max(a,i*hop))
            total[i] += overlap*p; weights[i] += overlap
    probs = total / np.maximum(weights[:,None],1e-12)
    # Explicit energy speech-activity detector. Threshold is a validation parameter, not an emotion label.
    silence = np.array([20*np.log10(np.sqrt(np.mean(y[round(i*hop*sr):min(len(y),round((i+1)*hop*sr))]**2))+1e-10) < cfg['silence_db'] for i in range(n)])
    labels, smooth = postprocess(probs,silence,hop,cfg)
    result = []
    for a,b,c in runs(labels):
        result.append({'start':float(a*hop),'end':float(min(duration,b*hop)),
                       'emotion':'silence' if c == -1 else classes[c],
                       'scores':{} if c == -1 else dict(zip(classes,map(float,smooth[a:b].mean(0))))})
    return result


def color_transcript(transcript, regions):
    units = []
    for segment in transcript['segments']:
        for unit in segment.get('words') or [segment]:
            overlap = lambda r: max(0,min(unit['end'],r['end'])-max(unit['start'],r['start']))
            best = max(regions,key=overlap) if regions else None
            emotion = best['emotion'] if best and overlap(best)>0 else 'unknown'
            units.append({**{k:unit[k] for k in ('start','end','text')},'predicted_emotion':emotion})
    return units


def transcript_html(units):
    return ' '.join(f'<span style="background:{COLORS.get(u["predicted_emotion"],"#eee")};color:#171717;padding:3px;border-radius:4px" title="{u["start"]:.2f}–{u["end"]:.2f}s · {html.escape(u["predicted_emotion"])}">{html.escape(u["text"])}</span>' for u in units)


def infer(path, review_checkpoints, emotion_checkpoint=None, text=None, asr_model='tiny.en', allow_download=False, region_config=None, asr_revision=None):
    started = time.perf_counter()
    transcript = None
    warnings = []
    if text is None:
        try:
            transcript = transcribe(path,asr_model,allow_download,asr_revision)
        except (OSError,ValueError,RuntimeError,ImportError) as exc:
            warnings.append(f'Transcription unavailable: {exc}. Cache the ASR model or supply a transcript.')
    if transcript is None:
        transcript = {'text':text or '', 'source':'supplied_unaligned' if text else 'unavailable','segments':[]}
    predictions = {}
    for name,checkpoint in review_checkpoints.items():
        try:
            b = load_checkpoint(checkpoint)
            y = audio_load(path,b['config']['sample_rate'])
            predictions[name] = predict_review(b,y,transcript['text'],FeatureStore(b['config'],allow_download=allow_download))
        except (OSError,ValueError,RuntimeError,ImportError) as exc:
            warnings.append(f'{name}: {exc}')
    regions = []
    if emotion_checkpoint:
        try:
            b = load_checkpoint(emotion_checkpoint)
            y = audio_load(path,b['config']['sample_rate'])
            regions = emotion_regions(b,y,FeatureStore(b['config'],allow_download=allow_download),region_config)
            if b['smoke_only']:
                warnings.append('Emotion checkpoint was trained in smoke mode; not a final experiment.')
        except (OSError,ValueError,RuntimeError,ImportError) as exc:
            warnings.append(f'Emotion: {exc}')
    # No fabricated word or sentence timing for a plain supplied text.
    colored = color_transcript(transcript,regions)
    return {'transcript':transcript,'emotion_regions':regions,'colored_transcript':colored,
            'predictions':predictions,'warnings':warnings,'latency_seconds':time.perf_counter()-started,
            'hardware':hardware(),'score_note':'Scores are uncalibrated. Regions are postprocessed window predictions, not validated natural emotion boundaries.'}
