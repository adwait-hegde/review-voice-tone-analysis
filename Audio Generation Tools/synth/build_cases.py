"""Enumerate every case and render its per-segment review text from the bank.

Deterministic: per-case choices are seeded from the case id, and voices/products
are assigned by position inside each (tone, polarity, n_segments, r) group, so
raising MAX_SEGMENTS or REDUNDANCY only appends rows. Writes cases.jsonl (source
of truth), cases.csv (flattened, for browsing) and pilot_ids.txt.

    python -m synth.build_cases
"""
import argparse
import csv
import io
import itertools
import re
from collections import Counter, defaultdict

from synth import config
from synth.common import (atomic_write_text, case_id, read_json, roles_for, stable_int, stable_rng,
                          write_jsonl)

SLOT = re.compile(r"\{(\w+)\}")
ALLOWED_CHARS = re.compile(r"^[A-Za-z' .,?!\-:;\"]+$")  # ENT char units after normalization
SENTENCE_START = re.compile(r"(^|[.!?]\s+)([a-z])")


def load_bank():
    brands = read_json(config.BANK_DIR / "brands.json")
    return {
        "products": read_json(config.BANK_DIR / "products.json"),
        "brands": [f"{a} {b}" for a in brands["first"] for b in brands["second"]],
        "fillers": read_json(config.BANK_DIR / "fillers.json"),
        "templates": read_json(config.BANK_DIR / "templates.json"),
    }


def load_voice_ids():
    if not config.VOICES_CSV.exists():
        raise SystemExit(f"{config.VOICES_CSV} not found (columns: voice_id,path,gender,source)")
    with open(config.VOICES_CSV, encoding="utf-8", newline="") as f:
        ids = sorted(row["voice_id"] for row in csv.DictReader(f))
    if not ids:
        raise SystemExit(f"{config.VOICES_CSV} lists no voices")
    return ids


def fill(template, slots):
    text = SLOT.sub(lambda m: slots[m.group(1)], template)
    return SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def render_segment(rng, bank, case, role, emotion, slots, used):
    """Pick a (content, reaction) pair whose word count suits a 10 s segment."""
    tone, polarity = case["tone"], case["polarity"]
    contents = bank["templates"]["content"][role][tone][polarity]
    reactions = bank["templates"]["reaction"][emotion][tone][polarity]
    reaction_first = role == "close" or (role == "body" and rng.random() < 0.5)
    pairs = []
    for ci, ct in enumerate(contents):
        if (role, ci) in used:
            continue
        for ri, rt in enumerate(reactions):
            if (emotion, ri) in used:
                continue
            parts = [fill(rt, slots), fill(ct, slots)] if reaction_first else [fill(ct, slots), fill(rt, slots)]
            text = " ".join(parts)
            pairs.append((ci, ri, text, len(text.split())))

    lo, hi = config.WORD_RANGE[emotion]
    in_range = [p for p in pairs if lo <= p[3] <= hi]
    primary = rng.choice(in_range) if in_range else min(pairs, key=lambda p: max(lo - p[3], p[3] - hi))
    others = [p for p in pairs if p[0] != primary[0] and p[1] != primary[1]] or [p for p in pairs if p != primary]
    shorter = [p for p in others if p[3] <= primary[3] - 3]
    alt = rng.choice(shorter) if shorter else min(others, key=lambda p: p[3])
    used.update({(role, primary[0]), (emotion, primary[1])})

    for text in (primary[2], alt[2]):
        if not ALLOWED_CHARS.match(text):
            raise SystemExit(f"bank text has characters outside ENT's char units: {text!r}")
    return {
        "role": role,
        "content_id": f"{role}.{tone}.{polarity}.{primary[0]}",
        "reaction_id": f"{emotion}.{tone}.{polarity}.{primary[1]}",
        "text": primary[2], "words": primary[3],
        "text_alt": alt[2], "words_alt": alt[3],
    }


def build_case(bank, voices, tone, polarity, emotions, r, seq_index):
    k = len(emotions)
    cid = case_id(tone, polarity, emotions, r)
    rng = stable_rng("case", cid)
    products, brands, fillers = bank["products"], bank["brands"], bank["fillers"]

    # Round-robin within the group keeps voices/products balanced across emotion
    # sequences; shifting by r gives each redundancy copy a different voice/product.
    group = (tone, polarity, k)
    voice = voices[(seq_index + stable_int("voice", *group) + r) % len(voices)]
    product = products[(seq_index + stable_int("product", *group) + 13 * r) % len(products)]
    brand = brands[stable_int("brand", cid) % len(brands)]

    case = {"id": cid, "tone": tone, "polarity": polarity, "sarcastic": tone == "sarcastic",
            "recommend": polarity == "positive", "n_segments": k, "duration": k * config.SEGMENT_SECONDS,
            "emotions": list(emotions), "r": r, "product": product["name"], "brand": brand, "voice_id": voice}
    base_slots = {"brand": brand, "product": product["name"], "time": rng.choice(fillers["time"]),
                  "person": rng.choice(fillers["person"]), "price": rng.choice(fillers["price"])}
    features = rng.sample(product["features"], len(product["features"]))
    flaws = rng.sample(product["flaws"], len(product["flaws"]))

    used, segments = set(), []
    for i, (role, emotion) in enumerate(zip(roles_for(k), emotions)):
        slots = dict(base_slots, feature=features[i % len(features)], flaw=flaws[i % len(flaws)])
        seg = render_segment(rng, bank, case, role, emotion, slots, used)
        segments.append({"idx": i, "emotion": emotion, **seg})
    case["segments"] = segments
    return case


def pilot_ids(cases):
    """PILOT_PER_GROUP cases per (tone, polarity, k), cycling the first emotion so all appear."""
    groups = defaultdict(list)
    for c in cases:
        if c["r"] == 1:
            groups[(c["tone"], c["polarity"], c["n_segments"])].append(c)
    chosen = []
    for gi, key in enumerate(sorted(groups)):
        members = sorted(groups[key], key=lambda c: stable_int("pilot", c["id"]))
        for j in range(config.PILOT_PER_GROUP):
            want = config.EMOTIONS[(gi * config.PILOT_PER_GROUP + j) % len(config.EMOTIONS)]
            pick = next((c for c in members if c["emotions"][0] == want and c["id"] not in chosen), members[j])
            chosen.append(pick["id"])
    return chosen


def to_csv(cases):
    cols = ["id", "tone", "polarity", "n_segments", "duration", "emotions", "r", "product", "brand", "voice_id"]
    for i in range(1, config.MAX_SEGMENTS + 1):
        cols += [f"seg{i}_emotion", f"seg{i}_text", f"seg{i}_text_alt"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    writer.writeheader()
    for c in cases:
        row = {k: c[k] for k in cols[:10]}
        row["emotions"] = "|".join(c["emotions"])
        for s in c["segments"]:
            n = s["idx"] + 1
            row.update({f"seg{n}_emotion": s["emotion"], f"seg{n}_text": s["text"], f"seg{n}_text_alt": s["text_alt"]})
        writer.writerow(row)
    return buf.getvalue()


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    bank, voices = load_bank(), load_voice_ids()
    cases = []
    for tone in config.TONES:
        for polarity in config.POLARITIES:
            for k in range(1, config.MAX_SEGMENTS + 1):
                for r in range(1, config.REDUNDANCY + 1):
                    for seq_index, emotions in enumerate(itertools.product(config.EMOTIONS, repeat=k)):
                        cases.append(build_case(bank, voices, tone, polarity, emotions, r, seq_index))

    write_jsonl(config.CASES_JSONL, cases)
    atomic_write_text(config.CASES_CSV, to_csv(cases))
    atomic_write_text(config.PILOT_IDS, "\n".join(pilot_ids(cases)) + "\n")

    segs = [s for c in cases for s in c["segments"]]
    off = sum(not (config.WORD_RANGE[s["emotion"]][0] <= s["words"] <= config.WORD_RANGE[s["emotion"]][1])
              for s in segs)
    print(f"{len(cases)} cases, {len(segs)} segments, {sum(c['duration'] for c in cases) / 3600:.2f} h of audio")
    print(f"segments outside their word range: {off}")
    by_emo = defaultdict(list)
    for s in segs:
        by_emo[s["emotion"]].append(s["words"])
    for emo, words in sorted(by_emo.items()):
        print(f"  {emo}: words min {min(words)} mean {sum(words) / len(words):.1f} max {max(words)}")
    print(f"voices used: {dict(sorted(Counter(c['voice_id'] for c in cases).items()))}")
    print(f"wrote {config.CASES_JSONL}, {config.CASES_CSV}, {config.PILOT_IDS}")


if __name__ == "__main__":
    main()
