"""Fit each segment to exactly 10.0 s and join them into one 16 kHz mono WAV per clip.

Per segment: trim edge silence; if the speech is too long, shorten long internal
pauses, then WSOLA-compress what remains; pad with a low noise floor (lead-in and
tail). Loudness is normalized per clip, not per segment, so the louder delivery of
angry segments survives. Resumable: a clip is finished once its .json sidecar
exists (written after the WAV, both via tmp + rename) and its signature matches the
current segments; clips whose text or voice changed are rebuilt. Clips whose
segments are not all synthesized yet are skipped.

    python -m synth.assemble [--pilot] [--ids-file F]
"""
import argparse
import sys

import numpy as np
import pyloudnorm
import soundfile as sf

from synth import config
from synth.audio import FRAME_SECONDS, speech_bounds, voiced_mask
from synth.common import (GracefulInterrupt, atomic_write_json, clean_tmp, clip_current, clip_signature, load_cases,
                          read_json, segment_done, segment_paths, select_cases, stable_int, stable_rng, tmp_path,
                          wav_path)

SR = config.SAMPLE_RATE
SEG_SAMPLES = int(config.SEGMENT_SECONDS * SR)
FADE = int(0.01 * SR)
MIN_TAIL = 0.1  # seconds of padding kept after the speech


def shorten_pauses(wav, max_pause):
    """Cut internal silent runs longer than max_pause down to max_pause."""
    n = int(SR * FRAME_SECONDS)
    mask = voiced_mask(wav, SR)
    keep = np.ones(len(wav), dtype=bool)
    max_frames = int(round(max_pause / FRAME_SECONDS))
    i = 0
    while i < len(mask):
        if mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and not mask[j]:
            j += 1
        if i > 0 and j < len(mask) and j - i > max_frames:
            keep[(i + max_frames // 2) * n:(j - (max_frames - max_frames // 2)) * n] = False
        i = j
    return wav[keep], float((~keep).sum()) / SR


def time_compress(wav, ratio):
    from audiotsm import wsola
    from audiotsm.io.array import ArrayReader, ArrayWriter

    writer = ArrayWriter(channels=1)
    wsola(channels=1, speed=ratio).run(ArrayReader(wav[np.newaxis, :].astype(np.float32)), writer)
    return writer.data[0]


def place_speech(speech_seconds, position, rng):
    """Silence before the speech inside its 10 s window, by the segment's place in the clip.

    Speech is pushed toward the neighbouring segments so emotion changes sound like
    ordinary sentence breaks: the first segment is right-aligned, the last
    left-aligned, middle ones centred; spare time collects at the clip's start and end.
    """
    spare = max(0.0, config.SEGMENT_SECONDS - speech_seconds - MIN_TAIL)
    if position == "only":
        return min(rng.uniform(*config.LEAD_IN_SECONDS), spare)
    if position == "first":
        tail = rng.uniform(*config.BOUNDARY_SIDE_SECONDS)
        return min(max(0.0, spare + MIN_TAIL - tail), config.MAX_START_SILENCE, spare)
    if position == "last":
        return min(rng.uniform(*config.BOUNDARY_SIDE_SECONDS), spare)
    return min((spare + MIN_TAIL) / 2, spare)  # middle: centred


def fit_segment(wav, rng, position="only"):
    """-> (exactly SEG_SAMPLES of audio, mask of padded samples, stats)."""
    start, end = speech_bounds(wav, SR)
    speech = wav[max(0, start - int(0.03 * SR)):min(len(wav), end + int(0.05 * SR))].copy()
    stats = {"speech_seconds_raw": round(len(speech) / SR, 3), "pause_cut_seconds": 0.0, "stretch": 1.0}
    budget = config.SEGMENT_SECONDS - config.LEAD_IN_SECONDS[0] - MIN_TAIL  # longest speech that fits
    if len(speech) / SR > budget:
        speech, cut = shorten_pauses(speech, config.MAX_PAUSE_SECONDS)
        stats["pause_cut_seconds"] = round(cut, 3)
    if len(speech) / SR > budget:
        ratio = len(speech) / SR / budget
        speech = time_compress(speech, ratio)
        stats["stretch"] = round(ratio, 4)
    speech = speech[:int(budget * SR)]
    fade = min(FADE, len(speech) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        speech[:fade] *= ramp
        speech[-fade:] *= ramp[::-1]

    lead = place_speech(len(speech) / SR, position, rng)
    a = min(int(lead * SR), SEG_SAMPLES - len(speech))
    out = np.zeros(SEG_SAMPLES, dtype=np.float32)
    pad = np.ones(SEG_SAMPLES, dtype=bool)
    out[a:a + len(speech)] = speech
    pad[a:a + len(speech)] = False
    stats.update(lead_in=round(a / SR, 3), tail=round((SEG_SAMPLES - a - len(speech)) / SR, 3),
                 speech_seconds=round(len(speech) / SR, 3),
                 compressed=stats["stretch"] > 1 + config.MAX_OVERRUN + 0.02)
    return out, pad, stats


def assemble_case(case):
    rng = stable_rng("assemble", case["id"])
    parts, pads, seg_stats = [], [], []
    last = len(case["segments"]) - 1
    for seg in case["segments"]:
        audio_path, meta_path = segment_paths(case, seg)
        wav, sr = sf.read(audio_path, dtype="float32")
        assert sr == SR, f"{audio_path} is {sr} Hz"
        position = "only" if last == 0 else "first" if seg["idx"] == 0 else "last" if seg["idx"] == last else "middle"
        out, pad, stats = fit_segment(wav, rng, position)
        meta = read_json(meta_path)
        stats.update(idx=seg["idx"], emotion=seg["emotion"], variant=meta["variant"], text=meta["text"],
                     duration_factor=meta["duration_factor"], tts_overlong=meta["overlong"])
        parts.append(out)
        pads.append(pad)
        seg_stats.append(stats)

    clip, pad = np.concatenate(parts), np.concatenate(pads)
    loudness = pyloudnorm.Meter(SR).integrated_loudness(clip)
    clip *= 10 ** ((config.TARGET_LUFS - loudness) / 20)
    ceiling = 10 ** (config.PEAK_CEILING_DBFS / 20)
    peak = float(np.abs(clip).max())
    if peak > ceiling:
        clip *= ceiling / peak
    noise = np.random.default_rng(stable_int("noise", case["id"])).standard_normal(int(pad.sum()))
    clip[pad] += (noise * 10 ** (config.PAD_NOISE_DBFS / 20)).astype(np.float32)

    out_path = wav_path(case)
    tmp = tmp_path(out_path)
    sf.write(tmp, clip, SR, format="WAV", subtype="PCM_16")
    tmp.replace(out_path)
    atomic_write_json(out_path.with_suffix(".json"), {
        "id": case["id"], "signature": clip_signature(case), "duration": len(clip) / SR, "sample_rate": SR,
        "input_lufs": round(loudness, 2),
        "peak_limited": peak > ceiling, "segments": seg_stats,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--ids-file")
    args = ap.parse_args()

    config.WAV_DIR.mkdir(parents=True, exist_ok=True)
    clean_tmp(config.WAV_DIR)
    cases = select_cases(load_cases(), args.pilot, args.ids_file)
    finished = {c["id"] for c in cases if clip_current(c, wav_path(c).with_suffix(".json"))}
    ready = [c for c in cases if c["id"] not in finished and all(segment_done(c, s) for s in c["segments"])]
    print(f"{len(finished)}/{len(cases)} clips already assembled, {len(ready)} ready, "
          f"{len(cases) - len(finished) - len(ready)} waiting for synthesis")
    n = 0
    with GracefulInterrupt() as stop:
        for case in ready:
            if stop.requested:
                break
            assemble_case(case)
            n += 1
    print(f">> assembled {n} clips; {len(finished) + n}/{len(cases)} done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n>> stopped immediately; the unfinished clip will be redone next run")
        sys.exit(130)
