"""Export assembled clips as manifest.jsonl and an ENT-style pickle. Run with venv-tools.

The pickle is a list of utterance maps with the fields ENT's IEMOCAP_MIX builder
(dataset/IEMOCAP/mix.py) produces: audio, audio_length (wav2vec2 frames), text,
tlabel, frame_tlabel, label, frame_label and their run lengths, plus id,
duration, interval, emotion_seq, sarcasm and polarity. ENT's get_labels only
handles one emotional interval; get_labels below is its loop generalized to any
number of merged intervals. --check verifies it against ENT's own get_labels
(extracted from mix.py) on every clip with a single non-neutral run.

    python -m synth.export_ent [--pilot] [--require-qc] [--no-audio] [--check]
"""
import argparse
import ast
import pickle

import soundfile as sf

from synth import config
from synth.common import (EMOTION_ID, clip_current, load_cases, merge_intervals, normalize_transcript, read_json,
                          select_cases, tmp_path, wav_path, write_jsonl)

# wav2vec2-base feature extractor (Wav2Vec2Config defaults), as in ENT's get_feat_extract_output_lengths
CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)


def feat_extract_output_length(n):
    for kernel, stride in zip(CONV_KERNEL, CONV_STRIDE):
        n = (n - kernel) // stride + 1
    return n


def get_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def get_labels(intervals, labels, duration, length, win_len=0.02, stride=0.02):
    """ENT's frame labelling (mix.py get_labels), for any list of intervals."""
    number_frames = int(duration / stride) + 1
    frame_labels = []
    for i in range(number_frames):
        win_start = i * stride
        win_end = win_start + win_len
        if win_end >= duration:
            win_end = duration
            win_start = max(duration - win_len, 0)
        for j in range(len(intervals)):
            if get_overlap([win_start, win_end], intervals[j]) >= 0.5 * (win_end - win_start):
                emo_frame = labels[j]
                break
        frame_labels.append(emo_frame)
        if win_end >= duration:
            break
    while len(frame_labels) < length:
        frame_labels.append(emo_frame)
    return frame_labels[:length]


def get_length(frame_label):
    """ENT's getLength, kept byte-for-byte (including its off-by-one) for compatibility."""
    frame_label_length = []
    length = 0
    now_label = frame_label[0]
    for label in frame_label:
        if label == now_label:
            length += 1
        else:
            frame_label_length.append(length)
            length = 0
            now_label = label
    frame_label_length.append(length)
    return frame_label_length


def read_symbol_table(path):
    table = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            sym, idx = line.strip().split()
            table[" " if sym == "▁" else sym] = int(idx)
    return table


def utterance_map(case, symbols, with_audio):
    import torch

    path = wav_path(case)
    assembled = read_json(path.with_suffix(".json"))
    wav, sr = sf.read(path, dtype="float32")
    assert sr == config.SAMPLE_RATE
    seg_texts = [normalize_transcript(s["text"]) for s in assembled["segments"]]
    text = " ".join(seg_texts)
    unknown = sorted({ch for ch in text if ch not in symbols})
    if unknown:
        raise SystemExit(f"{case['id']}: characters missing from ENT's char units: {unknown}")

    intervals = merge_intervals(case["emotions"])
    labels = [EMOTION_ID[iv["emo"]] for iv in intervals]
    audio_length = feat_extract_output_length(len(wav))
    frame_label = get_labels([[iv["start"], iv["end"]] for iv in intervals], labels, assembled["duration"], audio_length)
    # Per-character emotion, len(text) + 1 like mix.py: the leading slot and the
    # separating space go with the segment that follows.
    frame_tlabel = []
    for emo, t in zip(case["emotions"], seg_texts):
        frame_tlabel += [EMOTION_ID[emo]] * (len(t) + 1)
    item = {
        "id": case["id"],
        "audio": torch.from_numpy(wav) if with_audio else str(path),
        "audio_length": audio_length,
        "duration": assembled["duration"],
        "text": text,
        "tlabel": [symbols[ch] for ch in text],
        "frame_tlabel": frame_tlabel,
        "label": labels,
        "frame_label": frame_label,
        "frame_tlabel_length": get_length(frame_tlabel),
        "frame_label_length": get_length(frame_label),
        "interval": intervals,
        "emotion_seq": [EMOTION_ID[e] for e in case["emotions"]],
        "sarcasm": int(case["sarcastic"]),
        "polarity": int(case["recommend"]),
    }
    assert len(item["frame_tlabel_length"]) == len(labels) == len(item["frame_label_length"]), case["id"]
    return item


def ent_get_labels():
    """ENT's own get_labels/getOverlap, pulled out of mix.py without running its imports."""
    src = (config.ENT_DIR / "dataset" / "IEMOCAP" / "mix.py").read_text(encoding="utf-8")
    funcs = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ("get_labels", "getOverlap")]
    namespace = {}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "ENT/mix.py", "exec"), namespace)
    return namespace["get_labels"]


def check_against_ent(items):
    ent = ent_get_labels()
    checked = 0
    for it in items:
        non_neutral = [iv for iv in it["interval"] if iv["emo"] != "neu"]
        if len(non_neutral) != 1:
            continue  # ENT's version asserts a single emotional interval
        iv = non_neutral[0]
        data = {"duration": it["duration"], "emotion": [{"emo": EMOTION_ID[iv["emo"]], "start": iv["start"], "end": iv["end"]}]}
        labels, frames = ent(data, it["audio_length"])
        assert labels == it["label"], (it["id"], labels, it["label"])
        assert frames == it["frame_label"], it["id"]
        checked += 1
    print(f">> --check: frame labels identical to ENT's get_labels on {checked} single-emotion clips")


def manifest_row(case, item, qc):
    path = wav_path(case)
    row = {"id": case["id"], "audio": str(path), "duration": item["duration"], "sample_rate": config.SAMPLE_RATE,
           "text": item["text"], "text_raw": " ".join(s["text"] for s in read_json(path.with_suffix(".json"))["segments"]),
           "tone": case["tone"], "sarcastic": case["sarcastic"], "polarity": case["polarity"],
           "emotion_seq": case["emotions"], "emotion": item["interval"], "product": case["product"],
           "brand": case["brand"], "voice_id": case["voice_id"]}
    if qc:
        row["qc"] = [{k: s[k] for k in ("pred", "wer", "spk_sim_clip")} for s in qc["segments"]]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--ids-file")
    ap.add_argument("--require-qc", action="store_true", help="only clips that were QC'd with nothing flagged")
    ap.add_argument("--no-audio", action="store_true", help="store WAV paths instead of audio tensors in the pickle")
    ap.add_argument("--check", action="store_true", help="compare frame labels with ENT's own get_labels")
    args = ap.parse_args()

    from synth.qc import segment_fails

    symbols = read_symbol_table(config.ENT_DIR / "dataset" / "IEMOCAP" / "iemocap_char_units.txt")
    cases = [c for c in select_cases(load_cases(), args.pilot, args.ids_file)
             if clip_current(c, wav_path(c).with_suffix(".json"))]
    items, rows, skipped = [], [], 0
    for case in cases:
        qc_path = config.QC_DIR / f"{case['id']}.json"
        qc = read_json(qc_path) if clip_current(case, qc_path) else None
        if args.require_qc and (qc is None or any(segment_fails(qc, s) for s in qc["segments"])):
            skipped += 1
            continue
        item = utterance_map(case, symbols, not args.no_audio)
        items.append(item)
        rows.append(manifest_row(case, item, qc))
    if args.check:
        check_against_ent(items)

    write_jsonl(config.MANIFEST_JSONL, rows)
    tmp = tmp_path(config.ENT_PICKLE)
    with open(tmp, "wb") as f:
        pickle.dump(items, f)
    tmp.replace(config.ENT_PICKLE)
    print(f">> exported {len(items)} clips ({skipped} skipped by --require-qc) to {config.ENT_PICKLE} and {config.MANIFEST_JSONL}")


if __name__ == "__main__":
    main()
