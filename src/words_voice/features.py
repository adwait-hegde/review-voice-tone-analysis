"""Predictive features take waveform and text ONLY; cache keys are not features."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def audio_load(path, target_sr=16000, start=None, end=None):
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        a = 0 if start is None else round(start * sr)
        b = len(f) if end is None else min(len(f), round(end * sr))
        if a < 0 or b <= a:
            raise ValueError('Empty/invalid audio slice')
        f.seek(a)
        y = f.read(b-a, dtype='float32', always_2d=True).mean(axis=1)
    if sr != target_sr:
        divisor = math.gcd(sr, target_sr)
        y = resample_poly(y, target_sr // divisor, sr // divisor).astype('float32')
    if not len(y) or not np.isfinite(y).all():
        raise ValueError('Empty/nonfinite waveform')
    return y


def add_noise(y, snr_db, seed):
    if snr_db is None:
        return y
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(y))
    scale = np.sqrt(np.mean(y*y) / (10**(snr_db/10) * (np.mean(noise*noise)+1e-12)))
    return (y + scale * noise).astype('float32')


def mfcc(y, sr=16000, count=20):
    import librosa
    y = np.pad(y, (0, max(0, 512-len(y))))
    c = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=count, n_fft=512, hop_length=160)
    delta = np.diff(c, axis=1)
    if not delta.shape[1]:
        delta = np.zeros_like(c)
    return np.concatenate([c.mean(1), c.std(1), delta.mean(1), delta.std(1)]).astype('float32')


def masked_mean(hidden, mask):
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)


class Encoders:
    """Frozen models; process unpadded audio chunks singly to avoid group-norm padding artifacts."""
    def __init__(self, cfg, modalities, allow_download=False):
        import torch
        from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
        self.cfg, self.device = cfg, cfg['device']
        self.torch = torch
        kw = {'local_files_only': not allow_download}
        if 'text' in modalities:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg['text_encoder'], revision=cfg.get('text_revision'), **kw)
            self.text_model = AutoModel.from_pretrained(cfg['text_encoder'], revision=cfg.get('text_revision'), **kw).to(self.device).eval()
            self.text_model.requires_grad_(False)
        if 'audio' in modalities:
            self.processor = AutoFeatureExtractor.from_pretrained(cfg['audio_encoder'], revision=cfg.get('audio_revision'), **kw)
            self.audio_model = AutoModel.from_pretrained(cfg['audio_encoder'], revision=cfg.get('audio_revision'), **kw).to(self.device).eval()
            self.audio_model.requires_grad_(False)
            if self.processor.sampling_rate != cfg['sample_rate']:
                raise ValueError('Encoder/config sample-rate mismatch')

    def text(self, text):
        torch = self.torch
        size = min(self.cfg['text_chunk_tokens'], self.text_model.config.max_position_embeddings)
        encoded = self.tokenizer(text, truncation=True, max_length=size,
                                 return_overflowing_tokens=True, padding=True, return_tensors='pt')
        chunks, weights = [], []
        with torch.inference_mode():
            for i in range(encoded['input_ids'].shape[0]):
                batch = {k: v[i:i+1].to(self.device) for k,v in encoded.items()
                         if k in self.tokenizer.model_input_names}
                h = self.text_model(**batch).last_hidden_state
                chunks.append(masked_mean(h, batch['attention_mask']).cpu().numpy()[0])
                weights.append(int(batch['attention_mask'].sum().item()))
        return np.average(chunks, axis=0, weights=weights).astype('float32')

    def audio(self, y):
        torch, sr = self.torch, self.cfg['sample_rate']
        size = int(sr * self.cfg['audio_chunk_seconds'])
        values, lengths = [], []
        with torch.inference_mode():
            for start in range(0, len(y), size):
                part = y[start:start+size]
                # wav2vec convolution requires >=400 samples. Short tails are explicitly padded and masked.
                valid = len(part)
                part = np.pad(part, (0, max(0, 400-valid)))
                inputs = self.processor(part, sampling_rate=sr, return_tensors='pt')
                batch = {k: v.to(self.device) for k,v in inputs.items()}
                h = self.audio_model(**batch).last_hidden_state
                valid_frames = max(1, int(self.audio_model._get_feat_extract_output_lengths(torch.tensor(valid)).item()))
                mask = torch.arange(h.shape[1], device=self.device)[None,:] < valid_frames
                values.append(masked_mean(h, mask).cpu().numpy()[0]); lengths.append(valid)
        return np.average(values, axis=0, weights=lengths).astype('float32')


class FeatureStore:
    def __init__(self, cfg, cache='artifacts/cache', allow_download=False):
        self.cfg, self.cache, self.allow_download = cfg, Path(cache), allow_download
        self.cache.mkdir(parents=True, exist_ok=True)
        self.encoders = {}

    def encode(self, modality, text=None, y=None):
        if modality == 'mfcc':
            return mfcc(y, self.cfg['sample_rate'], self.cfg['mfcc_count'])
        if modality not in self.encoders:
            self.encoders[modality] = Encoders(self.cfg, [modality], self.allow_download)
        return getattr(self.encoders[modality], modality)(text if modality == 'text' else y)

    def get(self, r, kind, start=None, end=None):
        key = {'version': 1, 'audio': r['audio_sha256'] if kind != 'text' else None,
               'text': r['text'] if kind == 'text' else None, 'kind': kind, 'start': start, 'end': end,
               'config': {k:self.cfg.get(k) for k in ('sample_rate','mfcc_count','text_encoder','text_revision','audio_encoder','audio_revision','audio_chunk_seconds','text_chunk_tokens','noise_snr_db')},
               'noise_seed': self.cfg['seed'] if self.cfg.get('noise_snr_db') is not None else None}
        token = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        path = self.cache / (token + '.npy')
        if path.exists():
            return np.load(path, allow_pickle=False)
        y = None
        if kind != 'text':
            y = audio_load(r['audio'], self.cfg['sample_rate'], start, end)
            noise_seed = int(r['audio_sha256'][:8],16) ^ self.cfg['seed']
            y = add_noise(y, self.cfg.get('noise_snr_db'), noise_seed)
        x = self.encode(kind, text=r['text'], y=y)
        np.save(path, x)
        return x
