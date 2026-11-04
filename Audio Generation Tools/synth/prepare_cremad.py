"""Build voice prompts and per-voice emotion references from CREMA-D. Run once with venv-tools.

CREMA-D (Cao et al. 2014, ODbL) has 91 actors reading 12 sentences in six
emotions, each clip rated by ~10 listeners. For every clip whose intended emotion
is one of ENT's 4, its score is the listeners' agreement when their majority vote
matches that emotion (else 0). Actors are ranked by how well all four of their
emotions were recognized; the best N_PER_SEX women and men are kept. For each:

  voices/cremad_<actor>.wav               neutral prompt: best neutral clips joined (~8-10 s)
  emo_refs/cremad_<actor>/<emotion>.wav   emotion reference: best clips of that emotion joined

and voices.csv is rewritten to list them (the previous file is kept as voices_previous.csv).
Only the clips used are downloaded, into datasets/cremad_wav/.

    python -m synth.prepare_cremad
"""
import argparse
import csv
import time
import urllib.request
from collections import defaultdict

import numpy as np
import pyloudnorm
import soundfile as sf

from synth import config

MEDIA_URL = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/{}.wav"
CODES = {"NEU": ("neu", "N"), "HAP": ("hap", "H"), "ANG": ("ang", "A"), "SAD": ("sad", "S")}
N_PER_SEX = 12
PROMPT_CLIPS = 4  # neutral clips joined into the voice prompt
REF_CLIPS = 3  # clips joined into each emotion reference
MIN_AGREEMENT = 0.6
GAP_SECONDS = 0.3
SR = 16000


def clip_scores():
    """{actor: {emotion: [(score, fileName), ...] best first}}"""
    by_actor = defaultdict(lambda: defaultdict(list))
    with open(config.CREMAD_DIR / "processedResults" / "tabulatedVotes.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            actor, _, emo, level = row["fileName"].split("_")
            if emo not in CODES:
                continue
            code, vote = CODES[emo]
            score = float(row["agreement"]) if row["emoVote"] == vote else 0.0
            score += 0.05 if level == "HI" else 0.0  # prefer the clearest delivery
            by_actor[actor][code].append((score, row["fileName"]))
    for emos in by_actor.values():
        for clips in emos.values():
            clips.sort(reverse=True)
    return by_actor


def choose_actors(by_actor):
    with open(config.CREMAD_DIR / "VideoDemographics.csv", encoding="utf-8", newline="") as f:
        demo = {row["ActorID"]: row for row in csv.DictReader(f)}
    ranked = defaultdict(list)
    for actor, emos in by_actor.items():
        need = {"neu": PROMPT_CLIPS, "hap": REF_CLIPS, "ang": REF_CLIPS, "sad": REF_CLIPS}
        if all(sum(s >= MIN_AGREEMENT for s, _ in emos[e]) >= n for e, n in need.items()):
            quality = np.mean([s for e in config.EMOTIONS for s, _ in emos[e][:REF_CLIPS]])
            ranked[demo[actor]["Sex"]].append((quality, actor))
    chosen = []
    for sex in ("Female", "Male"):
        chosen += [(a, demo[a]) for _, a in sorted(ranked[sex], reverse=True)[:N_PER_SEX]]
    return chosen


def fetch(name):
    out = config.ROOT / "datasets" / "cremad_wav" / f"{name}.wav"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                tmp = out.with_suffix(".wav.tmp")
                urllib.request.urlretrieve(MEDIA_URL.format(name), tmp)
                tmp.replace(out)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)
    wav, sr = sf.read(out, dtype="float32")
    assert sr == SR and wav.ndim == 1, f"{out}: expected 16 kHz mono"
    return wav


def join(names, path):
    gap = np.zeros(int(GAP_SECONDS * SR), dtype=np.float32)
    parts = []
    for name in names:
        parts += [fetch(name), gap]
    wav = np.concatenate(parts[:-1])
    wav *= 10 ** ((config.TARGET_LUFS - pyloudnorm.Meter(SR).integrated_loudness(wav)) / 20)
    ceiling = 10 ** (config.PEAK_CEILING_DBFS / 20)
    wav *= min(1.0, ceiling / float(np.abs(wav).max()))  # loud angry clips: limit instead of clipping
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, SR, subtype="PCM_16")
    return len(wav) / SR


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    by_actor = clip_scores()
    chosen = choose_actors(by_actor)
    rows = []
    for actor, demo in chosen:
        vid = f"cremad_{actor}"
        emos = by_actor[actor]
        prompt = config.VOICES_CSV.parent / f"{vid}.wav"
        secs = join([n for _, n in emos["neu"][:PROMPT_CLIPS]], prompt)
        for emo in config.EMOTIONS:
            join([n for _, n in emos[emo][:REF_CLIPS]], config.EMO_REFS_DIR / vid / f"{emo}.wav")
        rows.append({"voice_id": vid, "path": str(prompt), "gender": demo["Sex"].lower(),
                     "source": f"CREMA-D actor {actor} ({demo['Age']}, {demo['Race']})"})
        print(f"{vid}: {demo['Sex']:<6} prompt {secs:4.1f}s  agreement "
              + " ".join(f"{e}={np.mean([s for s, _ in emos[e][:REF_CLIPS]]):.2f}" for e in config.EMOTIONS))

    if config.VOICES_CSV.exists():
        config.VOICES_CSV.replace(config.VOICES_CSV.with_name("voices_previous.csv"))
    with open(config.VOICES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["voice_id", "path", "gender", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f">> {len(rows)} voices written to {config.VOICES_CSV}; references in {config.EMO_REFS_DIR}")


if __name__ == "__main__":
    main()
