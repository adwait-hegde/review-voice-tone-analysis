"""Small numpy-only audio helpers shared by synthesize (index-tts venv) and assemble/qc (tools venv)."""
import numpy as np

from synth import config

FRAME_SECONDS = 0.02


def frame_rms_db(wav, sr, frame=FRAME_SECONDS):
    n = int(sr * frame)
    k = len(wav) // n
    if k == 0:
        return np.array([-120.0])
    frames = wav[: k * n].reshape(k, n)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def voiced_mask(wav, sr, silence_db=config.SILENCE_DB):
    """Per-20 ms frame: True where RMS is within silence_db of the loudest frame."""
    db = frame_rms_db(wav, sr)
    return db > db.max() + silence_db


def speech_bounds(wav, sr):
    """(start, end) sample indices of the first and last voiced frame."""
    idx = np.flatnonzero(voiced_mask(wav, sr))
    if len(idx) == 0:
        return 0, 0
    n = int(sr * FRAME_SECONDS)
    return int(idx[0] * n), int(min(len(wav), (idx[-1] + 1) * n))


def speech_seconds(wav, sr):
    start, end = speech_bounds(wav, sr)
    return (end - start) / sr
