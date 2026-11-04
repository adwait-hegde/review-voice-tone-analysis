"""Helpers shared by every stage (stdlib only; imported from both venvs)."""
import hashlib
import json
import os
import random
import re
import signal
import time
from pathlib import Path

from synth import config

EMOTION_ID = {"hap": 0, "ang": 1, "sad": 2, "neu": 3}  # ENT's emotion_id (exc merged into hap)
TONE_CODE = {"sarcastic": "s", "genuine": "g"}
POLARITY_CODE = {"positive": "pos", "negative": "neg"}


# --- Determinism ------------------------------------------------------------
def stable_int(*parts) -> int:
    """Process-independent hash (Python's hash() is salted per run)."""
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:16], 16)


def stable_rng(*parts) -> random.Random:
    return random.Random(stable_int(*parts))


def case_id(tone, polarity, emotions, r) -> str:
    return f"{TONE_CODE[tone]}-{POLARITY_CODE[polarity]}_{len(emotions)}_{'-'.join(emotions)}_r{r}"


def roles_for(k):
    if k == 1:
        return ["solo"]
    return ["open"] + ["body"] * (k - 2) + ["close"]


# --- Labels -----------------------------------------------------------------
def merge_intervals(emotions, seg_seconds=config.SEGMENT_SECONDS):
    """Per-10 s emotions -> ENT-style intervals, merging consecutive repeats."""
    intervals = []
    for i, emo in enumerate(emotions):
        start, end = i * seg_seconds, (i + 1) * seg_seconds
        if intervals and intervals[-1]["emo"] == emo:
            intervals[-1]["end"] = end
        else:
            intervals.append({"emo": emo, "start": start, "end": end})
    return intervals


_ENT_PUNC = re.compile(r'[\[\]\.\,\?\!\-\:\;"]+')


def normalize_transcript(text):
    """ENT's transcript normalization (dataset/IEMOCAP/preprocess_IEMOCAP.py)."""
    return " ".join(_ENT_PUNC.sub(" ", text.upper()).split())


# --- Files ------------------------------------------------------------------
def tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj):
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1))


def write_jsonl(path: Path, rows):
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean_tmp(directory: Path):
    """Delete leftovers from a stage that was killed mid-write."""
    if directory.exists():
        for p in directory.rglob("*.tmp*"):
            p.unlink(missing_ok=True)


# --- Cases ------------------------------------------------------------------
def load_cases():
    if not config.CASES_JSONL.exists():
        raise SystemExit(f"{config.CASES_JSONL} not found; run build_cases first.")
    return read_jsonl(config.CASES_JSONL)


def select_cases(cases, pilot=False, ids_file=None, shard=0, num_shards=1):
    if pilot:
        ids_file = config.PILOT_IDS
    if ids_file:
        wanted = {line.strip() for line in open(ids_file, encoding="utf-8") if line.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if num_shards > 1:
        cases = [c for c in cases if stable_int(c["id"]) % num_shards == shard]
    return cases


def emotion_settings(case, seg):
    """What the TTS is asked to do for one segment (also part of its cache key)."""
    emo = seg["emotion"]
    if config.EMOTION_MODE == "audio_ref":
        return {"mode": "audio_ref", "alpha": config.EMO_ALPHA[emo], "ref": emo_ref_for(case, seg)}
    vec = list(config.EMO_VECTORS[emo])
    if case["sarcastic"] and config.SARCASM_BLEND:
        for dim, w in config.SARCASM_BLEND.items():
            vec[config.EMO_DIMS.index(dim)] += w
    return {"mode": "vector", "vector": vec}


def emo_ref_for(case, seg):
    """The voice's own reference for this emotion if present, else one from the shared pool."""
    own = config.EMO_REFS_DIR / case["voice_id"] / f"{seg['emotion']}.wav"
    if own.exists():
        return str(own)
    refs = sorted((config.EMO_REFS_DIR / seg["emotion"]).glob("*.wav"))
    if not refs:
        raise SystemExit(f"EMOTION_MODE='audio_ref' but neither {own} nor {config.EMO_REFS_DIR / seg['emotion']}\\*.wav exists")
    return str(refs[stable_int(case["id"], seg["idx"]) % len(refs)])


def segment_key(case, seg):
    """Cache key: changes whenever the text, voice or emotion request changes."""
    spec = [config.TTS_MODEL, config.TTS_LANG, case["voice_id"], seg["text"], seg["text_alt"],
            emotion_settings(case, seg)]
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    return f"{case['id']}_s{seg['idx']}_{digest}"


def segment_paths(case, seg):
    stem = config.SEGMENTS_DIR / segment_key(case, seg)
    return stem.with_suffix(".flac"), stem.with_suffix(".json")


def segment_done(case, seg):
    audio, meta = segment_paths(case, seg)
    return meta.exists() and audio.exists()


def wav_path(case):
    return config.WAV_DIR / f"{case['id']}.wav"


def clip_signature(case):
    """Identifies the exact segments a clip was built from; a mismatch means it is stale."""
    return hashlib.sha1("|".join(segment_key(case, s) for s in case["segments"]).encode()).hexdigest()[:12]


def clip_current(case, sidecar: Path):
    """True if sidecar (a clip's assemble or QC json) exists and matches the case as it is now."""
    return sidecar.exists() and read_json(sidecar).get("signature") == clip_signature(case)


# --- Power ------------------------------------------------------------------
class KeepAwake:
    """Ask Windows not to idle-sleep while a stage runs (lid close / manual sleep still pause it)."""

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

    def __enter__(self):
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)
        return self

    def __exit__(self, *exc):
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        return False


def pin_to_performance_cores():
    """Keep this process on a hybrid CPU's performance cores (Windows; e.g. 13th-gen Intel).

    Synthesis is bound by the one Python thread feeding the GPU, and Windows kept moving it
    to efficiency cores: pinning halved the time per segment. Returns the logical processors
    used, or None if nothing changed (not Windows, not a hybrid CPU, or the call failed).
    """
    if os.name != "nt":
        return None
    import ctypes
    k32 = ctypes.WinDLL("kernel32")
    k32.GetSystemCpuSetInformation.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                                               ctypes.c_void_p, ctypes.c_ulong]
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    size = ctypes.c_ulong()
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetSystemCpuSetInformation(buf, size, ctypes.byref(size), None, 0):
        return None
    efficiency, off = {}, 0  # SYSTEM_CPU_SET_INFORMATION: Size@0, Group@12, LogicalProcessorIndex@14, EfficiencyClass@18
    while off < size.value:
        if int.from_bytes(buf.raw[off + 12:off + 14], "little") == 0:  # processor group 0 (up to 64 logical CPUs)
            efficiency[buf.raw[off + 14]] = buf.raw[off + 18]
        off += int.from_bytes(buf.raw[off:off + 4], "little")
    fast = sorted(cpu for cpu, cls in efficiency.items() if cls == max(efficiency.values()))
    if len(fast) == len(efficiency):
        return None
    if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), sum(1 << cpu for cpu in fast)):
        return None
    return fast


# --- Pause / interrupts -----------------------------------------------------
def wait_while_paused(stop=None):
    """Block while run.ps1 -Pause's flag file exists; returns the seconds spent paused."""
    if not config.PAUSE_FLAG.exists():
        return 0.0
    started = time.time()
    print(">> paused; run.ps1 -Resume continues (the model stays loaded)", flush=True)
    while config.PAUSE_FLAG.exists() and not (stop and stop.requested):
        time.sleep(2)
    print(">> resumed", flush=True)
    return time.time() - started



class GracefulInterrupt:
    """First Ctrl+C: finish the current item, then stop. Second Ctrl+C: stop now."""

    def __init__(self):
        self.requested = False

    def __enter__(self):
        self._previous = signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._previous)
        return False

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\n>> Ctrl+C: stopping after the current item (press again to stop immediately)", flush=True)
