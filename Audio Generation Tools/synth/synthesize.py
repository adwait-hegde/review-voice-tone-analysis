"""Synthesize every 10 s segment with IndexTTS-2.5. Run with the index-tts venv's python.

Resumable: a segment is finished once its .json sidecar exists (written after its
.flac, both via tmp + rename), so it is skipped on the next run. Ctrl+C finishes
the current segment and stops; a second Ctrl+C stops immediately. While
data\PAUSE exists (run.ps1 -Pause), it waits between segments with the model loaded. Whole cases are
done one at a time, grouped by voice (the voice prompt is encoded once per voice)
and shuffled within each voice; voices are spread evenly over tone, polarity and
emotion, so partial runs stay balanced on the labels.

    python -m synth.synthesize [--pilot] [--max-minutes N] [--shard I --num-shards N] [--rerender]
"""
import argparse
import contextlib
import csv
import hashlib
import io
import sys
import time

import soundfile as sf
import torch
import torchaudio

from synth import config
from synth.audio import speech_seconds
from synth.common import (GracefulInterrupt, KeepAwake, atomic_write_json, clean_tmp, emotion_settings, load_cases,
                          pin_to_performance_cores, segment_done, segment_paths, select_cases, stable_int, tmp_path,
                          wait_while_paused, wav_path)


def load_voices():
    with open(config.VOICES_CSV, encoding="utf-8", newline="") as f:
        return {row["voice_id"]: row["path"] for row in csv.DictReader(f)}


def memoize_embeddings(get_emb):
    """Cache w2v-BERT features per input clip.

    IndexTTS keeps only the last voice/emotion prompt, but every segment switches to a
    different emotion reference, so it would re-encode one (on CPU, tens of seconds)
    before almost every segment. There are only voices x emotions distinct clips.
    """
    cache = {}

    def cached(input_features, attention_mask):
        key = hashlib.sha1(input_features.cpu().numpy().tobytes()).hexdigest()
        if key not in cache:
            cache[key] = get_emb(input_features, attention_mask)
        return cache[key].clone()

    return cached


def load_tts():
    from indextts.infer_v2_5 import IndexTTS2

    with contextlib.redirect_stdout(io.StringIO()):
        tts = IndexTTS2(cfg_path=str(config.MODEL_DIR / "config.yaml"), model_dir=str(config.MODEL_DIR),
                        use_bf16=config.USE_BF16, use_qwen_emo=False)
    # On GPUs under 10 GB IndexTTS-2.5 splits any text over 40 characters into separately
    # generated chunks joined by silence. Our segments are ~10 s, so generate each in one pass.
    tts.low_vram = False
    if config.W2V_BERT_ON_CPU:
        tts.semantic_model.to("cpu")

        @torch.no_grad()
        def get_emb_on_cpu(input_features, attention_mask):  # same as IndexTTS2.get_emb, model on CPU
            out = tts.semantic_model(input_features=input_features.cpu(), attention_mask=attention_mask.cpu(),
                                     output_hidden_states=True)
            return (out.hidden_states[17].to(tts.device) - tts.semantic_mean) / tts.semantic_std

        tts.get_emb = get_emb_on_cpu
    tts.get_emb = memoize_embeddings(tts.get_emb)
    torch.cuda.empty_cache()  # also releases the fp32 GPT copy left over from the bf16 cast
    return tts


def tts_once(tts, voice_path, text, emo_kwargs, duration_factor, out_tmp):
    """One IndexTTS call -> 16 kHz mono float32. Its per-call chatter is discarded."""
    with contextlib.redirect_stdout(io.StringIO()):
        tts.infer(spk_audio_prompt=voice_path, text=text, output_path=str(out_tmp), lang=config.TTS_LANG,
                  duration_factor=duration_factor, verbose=False, **emo_kwargs)
    wav, sr = sf.read(out_tmp, dtype="float32", always_2d=True)
    out_tmp.unlink()
    wav = torch.from_numpy(wav.mean(axis=1))
    return torchaudio.functional.resample(wav, sr, config.SAMPLE_RATE).numpy()


def synthesize_segment(tts, case, seg, voice_path):
    """Generate one segment, retrying faster (duration_factor) then with text_alt if too long."""
    settings = emotion_settings(case, seg)
    if settings["mode"] == "vector":
        emo_kwargs = {"emo_vector": tts.normalize_emo_vec(settings["vector"], apply_bias=True)}
    else:
        emo_kwargs = {"emo_audio_prompt": settings["ref"], "emo_alpha": settings["alpha"]}
    audio_path, _ = segment_paths(case, seg)
    raw_tmp = audio_path.with_name(audio_path.stem + ".raw.tmp.wav")
    limit = config.SEGMENT_SECONDS * (1 + config.MAX_OVERRUN)

    attempts, best = [], None
    for variant in ("text", "text_alt"):
        factor = 1.0
        while len(attempts) < config.MAX_ATTEMPTS:
            t0 = time.perf_counter()
            wav = tts_once(tts, voice_path, seg[variant], emo_kwargs, factor, raw_tmp)
            speech = speech_seconds(wav, config.SAMPLE_RATE)
            attempts.append({"variant": variant, "duration_factor": round(factor, 3), "speech_seconds": round(speech, 3),
                             "audio_seconds": round(len(wav) / config.SAMPLE_RATE, 3),
                             "gen_seconds": round(time.perf_counter() - t0, 2)})
            if best is None or speech < best[2]:
                best = (wav, attempts[-1], speech)
            if speech <= limit:
                return best[0], best[1], attempts
            if factor <= config.MIN_DURATION_FACTOR + 1e-6:
                break  # already as fast as allowed: fall back to the shorter text
            factor = max(config.MIN_DURATION_FACTOR, factor * 0.97 * config.SEGMENT_SECONDS / speech)
    return best[0], best[1], attempts  # still over-long; assemble compresses it and flags it


def save_segment(case, seg, wav, chosen, attempts):
    audio_path, meta_path = segment_paths(case, seg)
    tmp = tmp_path(audio_path)
    sf.write(tmp, wav, config.SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    tmp.replace(audio_path)
    atomic_write_json(meta_path, {
        "case_id": case["id"], "idx": seg["idx"], "emotion": seg["emotion"], "voice_id": case["voice_id"],
        "model": config.TTS_MODEL, "settings": emotion_settings(case, seg),
        "variant": chosen["variant"], "text": seg[chosen["variant"]], "duration_factor": chosen["duration_factor"],
        "speech_seconds": chosen["speech_seconds"],
        "overlong": chosen["speech_seconds"] > config.SEGMENT_SECONDS * (1 + config.MAX_OVERRUN),
        "attempts": attempts, "gen_seconds": round(sum(a["gen_seconds"] for a in attempts), 2),
    })


def apply_rerender(cases):
    """Delete segments listed in rerender.txt (lines: '<case_id> <idx>') so they regenerate, once."""
    if not config.RERENDER_TXT.exists():
        print(f"no {config.RERENDER_TXT}; nothing to re-render")
        return
    by_id = {c["id"]: c for c in cases}
    n = 0
    for line in config.RERENDER_TXT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cid, idx = line.split()
        case = by_id.get(cid)
        if case is None:
            continue
        for p in segment_paths(case, case["segments"][int(idx)]):
            p.unlink(missing_ok=True)
        for p in (wav_path(case), wav_path(case).with_suffix(".json"), config.QC_DIR / f"{cid}.json"):
            p.unlink(missing_ok=True)
        n += 1
    config.RERENDER_TXT.replace(config.RERENDER_TXT.with_suffix(".applied.txt"))  # never re-apply
    print(f"re-render: cleared {n} segments")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true", help="only the cases in pilot_ids.txt")
    ap.add_argument("--ids-file", help="only the case ids listed in this file")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after the current segment once exceeded")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--rerender", action="store_true", help=f"first clear the segments listed in {config.RERENDER_TXT.name}")
    args = ap.parse_args()

    started = time.time()
    config.SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_tmp(config.SEGMENTS_DIR)
    all_cases = load_cases()
    if args.rerender:
        apply_rerender(all_cases)
    cases = select_cases(all_cases, args.pilot, args.ids_file, args.shard, args.num_shards)
    cases.sort(key=lambda c: (c["voice_id"], stable_int("order", c["id"])))
    todo = [(c, s) for c in cases for s in c["segments"] if not segment_done(c, s)]
    total = sum(len(c["segments"]) for c in cases)
    print(f"{total - len(todo)}/{total} segments already done, {len(todo)} to go")
    if not todo:
        return

    voices = load_voices()
    fast = pin_to_performance_cores()
    if fast:
        print(f">> pinned to performance cores {fast[0]}-{fast[-1]}")
    print(">> loading IndexTTS-2.5 ...", flush=True)
    tts = load_tts()
    torch.cuda.reset_peak_memory_stats()
    deadline = started + args.max_minutes * 60 if args.max_minutes else None
    done, gen_total, audio_total, failures = total - len(todo), 0.0, 0.0, 0

    with GracefulInterrupt() as stop, KeepAwake():
        for case, seg in todo:
            paused = wait_while_paused(stop)
            if deadline:
                deadline += paused  # time spent paused doesn't count against --max-minutes
            if stop.requested or (deadline and time.time() > deadline):
                print(">> time limit reached" if not stop.requested else ">> stopped on request")
                break
            try:
                wav, chosen, attempts = synthesize_segment(tts, case, seg, voices[case["voice_id"]])
            except (RuntimeError, ValueError) as e:  # keep unattended runs going past a bad segment
                failures += 1
                print(f"!! {case['id']} seg {seg['idx']} failed: {e}", flush=True)
                if failures >= 3:
                    raise
                continue
            failures = 0
            save_segment(case, seg, wav, chosen, attempts)
            done += 1
            gen = sum(a["gen_seconds"] for a in attempts)
            gen_total += gen
            audio_total += sum(a["audio_seconds"] for a in attempts)
            print(f"[{done}/{total}] {case['id']} seg {seg['idx']} {seg['emotion']} {case['voice_id']}: "
                  f"speech {chosen['speech_seconds']:.2f}s, {len(attempts)} try, df {chosen['duration_factor']}, "
                  f"{chosen['variant']}, gen {gen:.1f}s", flush=True)

    if audio_total:
        print(f">> session: {done - (total - len(todo))} segments, generation RTF {gen_total / audio_total:.2f}, "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
              f"{(time.time() - started) / 60:.1f} min wall")
    print(f">> {done}/{total} segments done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n>> stopped immediately; the unfinished segment will be redone next run")
        sys.exit(130)
