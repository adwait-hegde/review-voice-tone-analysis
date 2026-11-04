"""Pipeline settings shared by every stage.

Edit values here, then rerun run.ps1. build_cases is deterministic, so changing
MAX_SEGMENTS or REDUNDANCY only appends cases; finished audio is reused.
This module must stay stdlib-only: it is imported from both venvs.
"""
import os
from pathlib import Path

# --- Paths ------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
BANK_DIR = PROJECT_DIR / "bank"

# Models, venvs and audio (tens of GB): keep outside the repo and any synced folder.
# Set ENT_SYNTH_ROOT (or run.ps1 -Root) to put them elsewhere; default <home>\ent_synth.
ROOT = Path(os.environ.get("ENT_SYNTH_ROOT") or Path.home() / "ent_synth")
INDEX_TTS_DIR = ROOT / "index-tts"
MODEL_DIR = INDEX_TTS_DIR / "checkpoints"  # IndexTTS-2.5 weights
ENT_DIR = ROOT / "ENT"
VOICES_CSV = ROOT / "voices" / "voices.csv"
EMO_REFS_DIR = ROOT / "emo_refs"  # <voice_id>/<emotion>.wav per voice, or shared <emotion>/*.wav
CREMAD_DIR = ROOT / "datasets" / "CREMA-D"

DATA_DIR = ROOT / "data"
CASES_JSONL = DATA_DIR / "cases.jsonl"
CASES_CSV = DATA_DIR / "cases.csv"
PILOT_IDS = DATA_DIR / "pilot_ids.txt"
SEGMENTS_DIR = DATA_DIR / "segments"
WAV_DIR = DATA_DIR / "wav"
QC_DIR = DATA_DIR / "qc"
QC_CSV = DATA_DIR / "qc.csv"
RERENDER_TXT = DATA_DIR / "rerender.txt"
PAUSE_FLAG = DATA_DIR / "PAUSE"  # created by run.ps1 -Pause, removed by run.ps1 -Resume
MANIFEST_JSONL = DATA_DIR / "manifest.jsonl"
ENT_PICKLE = DATA_DIR / "ent_synth.pkl"

# --- Case space -------------------------------------------------------------
EMOTIONS = ["neu", "hap", "ang", "sad"]  # ENT / IEMOCAP 4-class codes
TONES = ["sarcastic", "genuine"]
POLARITIES = ["positive", "negative"]  # recommends buying / not buying
MAX_SEGMENTS = 3  # clips of 10..MAX_SEGMENTS*10 seconds
REDUNDANCY = 1  # copies of every (tone, polarity, emotion sequence)
SEGMENT_SECONDS = 10.0
PILOT_PER_GROUP = 2  # pilot cases per (tone, polarity, n_segments)

# Words per 10 s segment, by emotion, aiming at ~9 s of speech. Calibrated on the
# CREMA-D pilot (words/s at normal speed: neu 2.5, hap 2.8, ang 2.3, sad 2.6; the
# angry references give slow, emphatic delivery, so angry gets the fewest words).
WORD_RANGE = {"neu": (19, 23), "hap": (20, 25), "ang": (18, 22), "sad": (18, 22)}

# --- TTS (IndexTTS-2.5) -----------------------------------------------------
TTS_MODEL = "IndexTTS-2.5"
TTS_LANG = "EN"
USE_BF16 = True
SAMPLE_RATE = 16000
# IndexTTS's w2v-BERT feature model (~2.3 GB fp32) only runs when the voice prompt
# changes; keeping it on CPU leaves room on an 8 GB GPU for other apps (Chrome ~3 GB).
W2V_BERT_ON_CPU = True

# "audio_ref": the voice's own reference recording for the emotion (EMO_REFS_DIR,
# built by prepare_cremad), blended in with EMO_ALPHA. "vector": the emotion vectors
# below. In the pilot, vectors gave no recognizable neutral or sad (emotion2vec+ heard
# happy/angry for everything); CREMA-D references did much better, and alpha 0.8 vs
# 1.0 was best for neutral vs the rest.
EMOTION_MODE = "audio_ref"
EMO_ALPHA = {"neu": 0.8, "hap": 1.0, "ang": 1.0, "sad": 1.0}
# Order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm].
# Passed through IndexTTS's normalize_emo_vec (bias + sum<=0.8), as its web UI does.
EMO_VECTORS = {
    "neu": [0, 0, 0, 0, 0, 0, 0, 0.8],
    "hap": [0.8, 0, 0, 0, 0, 0, 0, 0],
    "ang": [0, 0.8, 0, 0, 0, 0, 0, 0],
    "sad": [0, 0, 0.8, 0, 0, 0, 0, 0],
}
# Optional extra weight added to sarcastic segments, e.g. {"disgusted": 0.1}.
# None keeps sarcasm purely lexical; try a blend in the pilot before enabling.
SARCASM_BLEND = None
EMO_DIMS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

# Over-long segments: speech (edge silence trimmed) up to MAX_OVERRUN over 10 s is
# accepted and tightened in assemble; longer ones are regenerated faster via
# duration_factor (not below MIN_DURATION_FACTOR), then with the shorter text_alt.
MAX_OVERRUN = 0.08
MIN_DURATION_FACTOR = 0.8
MAX_ATTEMPTS = 4

# --- Assembly ---------------------------------------------------------------
TARGET_LUFS = -23.0  # per clip, so emotion-related loudness differences survive
PEAK_CEILING_DBFS = -1.0
PAD_NOISE_DBFS = -65.0  # padding is a low noise floor, not digital zero
LEAD_IN_SECONDS = (0.15, 0.35)  # silence before speech in a single-segment clip
# Multi-segment clips keep emotion changes flowing like ordinary sentence breaks: each
# side of a boundary gets this much silence, and the spare time in each 10 s window
# goes to the clip's start and end instead (speech never leaves its own window).
BOUNDARY_SIDE_SECONDS = (0.15, 0.3)
MAX_START_SILENCE = 1.5  # spare time beyond this at the clip start moves to the boundary
MAX_PAUSE_SECONDS = 0.35  # internal pauses longer than this may be shortened
SILENCE_DB = -40.0  # frame RMS below peak-relative threshold counts as silence

# --- QC ---------------------------------------------------------------------
SER_MODEL = "emotion2vec/emotion2vec_plus_large"
WHISPER_MODEL = "small.en"
QC_DEVICE = "cpu"  # GPU is busy with TTS; set "cuda" if running QC alone
MAX_WER = 0.25  # segments above this are listed for re-rendering
# ECAPA cosine of each segment to the rest of its clip. A change of emotion alone
# lowers it (pilot range 0.37-0.81), so only flag clear voice drift.
MIN_SPEAKER_SIM = 0.45
RERENDER_ON_SER_MISMATCH = False  # report SER disagreement; don't filter on it yet
