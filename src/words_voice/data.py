"""Auditable metadata adapter; no scientific labels inferred from filenames."""
import collections
import hashlib
import json
import math
import re
import wave
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    return ' '.join(re.findall(r"\w+", text.lower()))


def labels_for(identifier, mapping):
    entry = mapping['review_prefixes'].get(identifier.split('_')[0])
    if entry is None:
        return {'sentiment': None, 'sarcasm': None}
    if not entry.get('verified') or not entry.get('evidence'):
        raise ValueError('Label mapping requires verified=true and nonempty evidence')
    if entry['sentiment'] not in ('positive', 'negative') or entry['sarcasm'] not in ('genuine', 'sarcastic'):
        raise ValueError('Invalid intended sentiment or sarcasm mapping')
    return {k: entry[k] for k in ('sentiment', 'sarcasm')}


def segment_bounds(segment, duration, count, mapping):
    if 'start' in segment and 'end' in segment:
        start, end = float(segment['start']), float(segment['end'])
        provenance = 'explicit_metadata'
    elif mapping['timing']['mode'] == 'fixed_slots_verified':
        if not mapping['timing'].get('evidence'):
            raise ValueError('Verified slot timing requires evidence')
        size = float(mapping['timing']['slot_seconds'])
        start, end = segment['idx'] * size, min((segment['idx'] + 1) * size, duration)
        provenance = 'verified_slot'
    elif count == 1 and segment['idx'] == 0:
        start, end, provenance = 0.0, duration, 'whole_recording_single_label'
    else:
        return None, None, 'unresolved'
    if not 0 <= start < end <= duration + 0.001:
        raise ValueError('Invalid segment boundaries')
    return start, end, provenance


def audit(root, mapping_path, out):
    root, out = Path(root).resolve(), Path(out)
    mapping = read_json(mapping_path)
    files = {'json': {}, 'wav': {}}
    issues, rows = [], []
    for p in sorted(root.rglob('*')):
        if p.suffix.lower()[1:] in files:
            kind = p.suffix.lower()[1:]
            key = str(p.relative_to(root).with_suffix(''))
            if key in files[kind]:
                issues.append({'file': str(p), 'error': 'duplicate pairing key'})
            files[kind][key] = p
    missing = sorted(set(files['json']) ^ set(files['wav']))
    seen_ids = set()
    for key in sorted(set(files['json']) & set(files['wav'])):
        jp, ap = files['json'][key], files['wav'][key]
        try:
            d = read_json(jp)
            if not isinstance(d.get('id'), str) or not isinstance(d.get('segments'), list) or not d['segments']:
                raise ValueError('id string and nonempty segments list required')
            if d['id'] in seen_ids:
                raise ValueError('Duplicate metadata id')
            seen_ids.add(d['id'])
            with wave.open(str(ap), 'rb') as f:
                sr, frames, channels = f.getframerate(), f.getnframes(), f.getnchannels()
                width = f.getsampwidth()
                pcm = f.readframes(frames)
                if sr <= 0 or frames == 0 or len(pcm) != frames * channels * width:
                    raise ValueError('Empty or truncated PCM audio')
                duration = frames / sr
            if not math.isfinite(float(d['duration'])) or abs(duration - float(d['duration'])) > .05:
                raise ValueError('Audio/metadata duration mismatch')
            if sr != d['sample_rate']:
                raise ValueError('Audio/metadata sample-rate mismatch')
            segs = []
            indices = [s.get('idx') for s in d['segments']]
            if sorted(indices) != list(range(len(indices))) or any(type(i) is not int for i in indices):
                raise ValueError('Segment indices must be unique consecutive integers starting at zero')
            for s in sorted(d['segments'], key=lambda s: s['idx']):
                if not isinstance(s.get('text'), str) or not s['text'].strip():
                    raise ValueError('Each segment needs a nonempty transcript string')
                emotion = s.get('emotion')
                if emotion is not None and emotion not in mapping['emotions']:
                    raise ValueError(f'Unknown emotion code: {emotion}')
                start, end, timing = segment_bounds(s, duration, len(indices), mapping)
                segs.append({'idx': s['idx'], 'text': s['text'], 'emotion': mapping['emotions'].get(emotion),
                             'start': start, 'end': end, 'timing_provenance': timing,
                             'speech_seconds_reported': s.get('speech_seconds'),
                             'slot_sum_evidence': sum(s.get(k, 0) for k in ('lead_in','speech_seconds','tail')) if 'tail' in s else None})
            row = {'id': d['id'], 'audio': str(ap), 'metadata': str(jp), 'audio_sha256': digest(ap),
                   'pcm_sha256': hashlib.sha256(pcm).hexdigest(), 'sample_rate': sr, 'channels': channels,
                   'duration': duration, 'text': ' '.join(s['text'] for s in segs), 'segments': segs,
                   'origin': d.get('origin', mapping['data_origin']), **labels_for(d['id'], mapping),
                   'generation_evidence': any('tts_overlong' in s for s in d['segments'])}
            for name in ('speaker_id', 'voice_id', 'script_id', 'source_id', 'template_id', 'split'):
                row[name] = d.get(name)
            rows.append(row)
        except (ValueError, KeyError, TypeError, wave.Error, EOFError, OSError) as exc:
            issues.append({'file': str(jp), 'error': str(exc)})
    def duplicates(items):
        groups = collections.defaultdict(list)
        for k, v in items:
            groups[k].append(v)
        return [v for v in groups.values() if len(v) > 1]
    report = {'recording_count': len(rows), 'total_duration_seconds': sum(r['duration'] for r in rows),
              'sample_rates': dict(collections.Counter(r['sample_rate'] for r in rows)),
              'emotion_distribution': dict(collections.Counter(s['emotion'] for r in rows for s in r['segments'])),
              'verified_overall_classes': dict(collections.Counter(f"{r['sarcasm']} {r['sentiment']}" for r in rows if r['sarcasm'] and r['sentiment'])),
              'unlabelled_reviews': sum(r['sentiment'] is None or r['sarcasm'] is None for r in rows),
              'prefix_counts_NOT_labels': dict(collections.Counter(r['id'].split('_')[0] for r in rows)),
              'id_patterns': sorted(set(re.sub(r'\d+', '#', r['id']) for r in rows)),
              'emotion_codes': list(mapping['emotions']), 'origin': mapping['data_origin'],
              'origin_evidence': mapping['data_origin_evidence'], 'missing_pairs': missing, 'invalid': issues,
              'duplicate_audio': duplicates((r['pcm_sha256'], r['id']) for r in rows),
              'duplicate_transcripts': duplicates((normalize(r['text']), r['id']) for r in rows),
              'duplicate_segment_transcripts': duplicates((normalize(s['text']), r['id']) for r in rows for s in r['segments']),
              'missing_identifiers': {k: sum(not r[k] for r in rows) for k in ('speaker_id','voice_id','script_id','source_id','template_id')},
              'unresolved_segment_boundaries': sum(s['start'] is None for r in rows for s in r['segments']),
              'ten_second_sum_evidence': sum(s['slot_sum_evidence'] is not None and abs(s['slot_sum_evidence']-10)<.01 for r in rows for s in r['segments']),
              'generation_sources': [str(p) for p in root.parent.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.txt') and any(word in p.stem.lower() for word in ('generat','tts','dataset')) and not any(x in p.parts for x in ('.venv','src','tests','docs','artifacts','.git'))]}
    write_json(out / 'audit.json', report)
    write_json(out / 'manifest.json', {'schema_version': 1, 'label_mapping': mapping, 'records': rows})
    return report
