"""Quality checks per clip and per 10 s segment. Run with venv-tools.

Per clip: 16 kHz mono, exact length, no clipping. Per segment: emotion predicted by
emotion2vec+ large (9 classes, scored on ENT's 4), Whisper WER against the text
actually spoken, and ECAPA speaker similarity to the rest of the clip and to the
voice prompt. (SpeechBrain's IEMOCAP classifier was tried first and labelled every
IndexTTS segment 'ang' at p=1.0, so it is not used.) Results go to qc/<id>.json (resumable), then qc.csv
and qc_summary.txt are rebuilt. Segments failing WER/speaker/format checks are
listed in rerender.txt; apply with `run.ps1 -Rerender`.

    python -m synth.qc [--pilot] [--ids-file F]
"""
import argparse
import csv
import io
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

from synth import config
from synth.common import (GracefulInterrupt, KeepAwake, atomic_write_json, atomic_write_text, clean_tmp,
                          clip_current, clip_signature, load_cases, normalize_transcript, read_json, select_cases,
                          wait_while_paused, wav_path)

SR = config.SAMPLE_RATE
SEG = int(config.SEGMENT_SECONDS * SR)
MODELS_DIR = config.ROOT / "qc_models"

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_words(rest))
    return str(n)


def asr_normalize(text):
    """Whisper writes '10/10' and '$40'; spell digits so they compare with the script."""
    text = re.sub(r"\d+", lambda m: f" {number_words(int(m.group()))} ", text.replace("$", " ").replace("/", " out of "))
    return normalize_transcript(text).replace("'", "").split()


def wer(ref_words, hyp_words):
    d = list(range(len(hyp_words) + 1))
    for i, r in enumerate(ref_words, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp_words, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(1, len(ref_words))


E2V_TO_ENT = {"neutral": "neu", "happy": "hap", "angry": "ang", "sad": "sad"}


class Models:
    def __init__(self):
        import torch
        from faster_whisper import WhisperModel
        from funasr import AutoModel
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.torch = torch
        self.ser = AutoModel(model=config.SER_MODEL, hub="hf", device=config.QC_DEVICE, disable_update=True)
        self.spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                  savedir=str(MODELS_DIR / "ecapa"),
                                                  local_strategy=LocalStrategy.COPY,  # Windows can't symlink
                                                  run_opts={"device": config.QC_DEVICE})
        self.asr = WhisperModel(config.WHISPER_MODEL, device=config.QC_DEVICE,
                                compute_type="int8" if config.QC_DEVICE == "cpu" else "float16",
                                download_root=str(MODELS_DIR / "whisper"))
        self.prompt_emb = {}

    def emotion(self, wav):
        """-> (best of ENT's 4 classes, their scores, emotion2vec's top label over all 9)."""
        rec = self.ser.generate(wav, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        scores = {label.split("/")[-1]: float(s) for label, s in zip(rec["labels"], rec["scores"])}
        four = {code: round(scores.get(name, 0.0), 3) for name, code in E2V_TO_ENT.items()}
        return max(four, key=four.get), four, max(scores, key=scores.get)

    def embed(self, wav):
        emb = self.spk.encode_batch(self.torch.from_numpy(wav)[None]).squeeze()
        return (emb / emb.norm()).numpy()

    def prompt_embedding(self, voice_id, path):
        if voice_id not in self.prompt_emb:
            import torchaudio
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torchaudio.functional.resample(self.torch.from_numpy(wav.mean(axis=1)), sr, SR).numpy()
            self.prompt_emb[voice_id] = self.embed(wav[: 15 * SR])
        return self.prompt_emb[voice_id]

    def transcribe(self, wav):
        segments, _ = self.asr.transcribe(wav, language="en", beam_size=5, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments)


def check_case(models, case, voices):
    path = wav_path(case)
    info = sf.info(path)
    wav, _ = sf.read(path, dtype="float32")
    clip_ok = {"sample_rate": info.samplerate == SR, "mono": info.channels == 1,
               "exact_length": info.frames == case["n_segments"] * SEG,
               "no_clipping": int((np.abs(wav) >= 0.999).sum()) == 0}
    assembled = read_json(path.with_suffix(".json"))
    seg_wavs = [wav[i * SEG:(i + 1) * SEG] for i in range(case["n_segments"])]
    embs = [models.embed(w) for w in seg_wavs]
    prompt = models.prompt_embedding(case["voice_id"], voices[case["voice_id"]])

    rows = []
    for i, (seg, w) in enumerate(zip(case["segments"], seg_wavs)):
        pred, probs, pred9 = models.emotion(w)
        spoken = assembled["segments"][i]["text"]  # text or text_alt, whichever was synthesized
        hyp = models.transcribe(w)
        others = [e for j, e in enumerate(embs) if j != i]
        clip_sim = float(np.dot(embs[i], np.mean(others, axis=0) / np.linalg.norm(np.mean(others, axis=0)))) \
            if others else None
        rows.append({"idx": i, "target": seg["emotion"], "pred": pred, "probs": probs, "pred9": pred9,
                     "wer": round(wer(asr_normalize(spoken), asr_normalize(hyp)), 3), "ref": spoken, "hyp": hyp,
                     "spk_sim_clip": None if clip_sim is None else round(clip_sim, 3),
                     "spk_sim_prompt": round(float(np.dot(embs[i], prompt)), 3),
                     "stretch": assembled["segments"][i]["stretch"]})
    return {"id": case["id"], "signature": clip_signature(case), "voice_id": case["voice_id"], "tone": case["tone"],
            "polarity": case["polarity"], "n_segments": case["n_segments"], "clip_checks": clip_ok, "segments": rows}


def segment_fails(clip, s):
    reasons = []
    if not all(clip["clip_checks"].values()):
        reasons.append("clip_format")
    if s["wer"] > config.MAX_WER:
        reasons.append("wer")
    if s["spk_sim_clip"] is not None and s["spk_sim_clip"] < config.MIN_SPEAKER_SIM:
        reasons.append("speaker")
    if config.RERENDER_ON_SER_MISMATCH and s["pred"] != s["target"]:
        reasons.append("ser")
    return reasons


def summarize(results):
    out = io.StringIO()
    segs = [(c, s) for c in results for s in c["segments"]]
    if not segs:
        return "no QC results yet\n"
    emos = config.EMOTIONS
    conf = Counter((s["target"], s["pred"]) for _, s in segs)
    out.write(f"QC over {len(results)} clips / {len(segs)} segments\n\n")
    out.write("SER confusion (rows = target, cols = best of ENT's 4 by emotion2vec+)\n")
    out.write("        " + "".join(f"{e:>6}" for e in emos) + "   agree\n")
    for t in emos:
        n = sum(conf[(t, p)] for p in emos)
        out.write(f"  {t:<6}" + "".join(f"{conf[(t, p)]:>6}" for p in emos) +
                  (f"   {conf[(t, t)] / n:.0%}\n" if n else "\n"))
    agree = sum(s["pred"] == s["target"] for _, s in segs) / len(segs)
    out.write(f"  overall agreement {agree:.0%}\n")
    for tone in config.TONES:
        ts = [s for c, s in segs if c["tone"] == tone]
        if ts:
            out.write(f"  {tone}: {sum(s['pred'] == s['target'] for s in ts) / len(ts):.0%}\n")

    out.write("\nWER / speaker similarity by voice\n")
    by_voice = defaultdict(list)
    for c, s in segs:
        by_voice[c["voice_id"]].append(s)
    for v, ss in sorted(by_voice.items()):
        sims = [s["spk_sim_clip"] for s in ss if s["spk_sim_clip"] is not None]
        out.write(f"  {v}: n={len(ss):>3}  mean WER {np.mean([s['wer'] for s in ss]):.3f}  "
                  f"sim(clip) {np.mean(sims) if sims else float('nan'):.3f}  "
                  f"sim(prompt) {np.mean([s['spk_sim_prompt'] for s in ss]):.3f}\n")
    out.write(f"  all: mean WER {np.mean([s['wer'] for _, s in segs]):.3f}\n")

    bad_clips = [c["id"] for c in results if not all(c["clip_checks"].values())]
    out.write(f"\nclips failing format checks: {len(bad_clips)} {bad_clips[:10]}\n")
    out.write(f"segments time-compressed >10%: {sum(s['stretch'] > 1.10 for _, s in segs)}\n")
    fails = Counter(r for c, s in segs for r in segment_fails(c, s))
    out.write(f"segments flagged for re-render: {sum(bool(segment_fails(c, s)) for c, s in segs)} {dict(fails)}\n")
    return out.getvalue()


def consolidate(cases):
    results = [read_json(config.QC_DIR / f"{c['id']}.json") for c in cases
               if clip_current(c, config.QC_DIR / f"{c['id']}.json")]
    buf = io.StringIO()
    cols = ["id", "idx", "voice_id", "tone", "polarity", "target", "pred", "wer", "spk_sim_clip",
            "spk_sim_prompt", "stretch", "clip_ok", "flags", "ref", "hyp"]
    writer = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    writer.writeheader()
    rerender = []
    for c in results:
        for s in c["segments"]:
            flags = segment_fails(c, s)
            if flags:
                rerender.append(f"{c['id']} {s['idx']}")
            writer.writerow({"id": c["id"], "voice_id": c["voice_id"], "tone": c["tone"], "polarity": c["polarity"],
                             "clip_ok": all(c["clip_checks"].values()), "flags": "|".join(flags),
                             **{k: s[k] for k in ("idx", "target", "pred", "wer", "spk_sim_clip", "spk_sim_prompt",
                                                  "stretch", "ref", "hyp")}})
    atomic_write_text(config.QC_CSV, buf.getvalue())
    if rerender:
        atomic_write_text(config.RERENDER_TXT, "\n".join(rerender) + "\n")
    else:
        config.RERENDER_TXT.unlink(missing_ok=True)
    summary = summarize(results)
    atomic_write_text(config.DATA_DIR / "qc_summary.txt", summary)
    print(summary)
    if rerender:
        print(f"{len(rerender)} segments listed in {config.RERENDER_TXT}; apply with: run.ps1 -Rerender")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--ids-file")
    args = ap.parse_args()

    config.QC_DIR.mkdir(parents=True, exist_ok=True)
    clean_tmp(config.QC_DIR)
    cases = select_cases(load_cases(), args.pilot, args.ids_file)
    todo = [c for c in cases if clip_current(c, wav_path(c).with_suffix(".json"))
            and not clip_current(c, config.QC_DIR / f"{c['id']}.json")]
    print(f"{len(todo)} clips to check")
    if todo:
        with open(config.VOICES_CSV, encoding="utf-8", newline="") as f:
            voices = {row["voice_id"]: row["path"] for row in csv.DictReader(f)}
        t0 = time.time()
        models = Models()
        print(f">> QC models loaded in {time.time() - t0:.0f}s", flush=True)
        with GracefulInterrupt() as stop, KeepAwake():
            for n, case in enumerate(todo, 1):
                wait_while_paused(stop)
                if stop.requested:
                    break
                atomic_write_json(config.QC_DIR / f"{case['id']}.json", check_case(models, case, voices))
                print(f"[{n}/{len(todo)}] {case['id']}", flush=True)
    consolidate(cases)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n>> stopped immediately; unchecked clips will be checked next run")
        sys.exit(130)
