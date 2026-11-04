# Implementation and measured results

Completed 2026-11-03 (America/Chicago), Python 3.12, Apple M5 / 16 GB RAM / CPU. All 336 original WAV SHA256 hashes were unchanged in the final audit.

## Verification

- 15 tests passed: schema/pairing, verified and unknown labels, slot/audio slicing, grouped splits, portable manifests, padding/masks, missing labels, classical and neural checkpoint inference, known metrics, challenge-template rejection, ASR timestamp preservation and Streamlit startup.
- Tiny random local encoder fixtures test engineering only; they are never scientific results.
- All six review models were fully trained. MFCC emotion was fully trained; pretrained audio emotion passed a smoke run only.
- Real MiniLM, wav2vec2 and Whisper weights were used at recorded immutable revisions. A 10-second real inference probe returned 23 timestamped/coloured words, six predicted emotion/silence regions and all three neural predictions without application warnings in 4.53 seconds including model loading. One probe is not a representative latency benchmark.

## Single-seed held-out test

Seed 42; all review models used identical 66 test recordings / 50 heuristic source groups. Selection used 63 validation recordings; test did not drive subsequent model changes. These are **likely synthetic-data results**, not evidence of human generalization.

| Model | Sentiment macro-F1 [95% CI] | Sarcasm macro-F1 [95% CI] | Combined macro-F1 [95% CI] |
|---|---|---|---|
| majority | 0.283 [0.200, 0.353] | 0.275 [0.183, 0.344] | 0.054 [0.015, 0.094] |
| tfidf | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| mfcc | 0.375 [0.267, 0.492] | 0.548 [0.423, 0.664] | 0.241 [0.140, 0.336] |
| neural_text | 0.802 [0.700, 0.884] | 0.906 [0.816, 0.970] | 0.752 [0.610, 0.842] |
| neural_audio | 0.741 [0.615, 0.842] | 0.871 [0.788, 0.937] | 0.633 [0.491, 0.734] |
| fusion | 0.861 [0.764, 0.940] | 0.905 [0.816, 0.969] | 0.797 [0.662, 0.879] |

MFCC segment emotion macro-F1: **0.725**, 95% group CI [0.651, 0.799], 186 labelled test slots. This measures metadata/requested emotion agreement, not perceived human emotion or natural boundary accuracy.

Fusion has a higher combined-category point estimate than the pretrained unimodal heads on this single-seed test, but TF-IDF is perfect on this corpus. This does not establish a general fusion advantage or statistical significance. Unknown templates/voices and strong synthetic textual cues limit interpretation. TF-IDF’s [1,1] bootstrap interval describes this finite perfectly predicted cohort, not certainty about future performance.

Intervals use 1,000 source-group bootstrap resamples with fixed label vocabulary, never overlapping windows as independent samples. True source identities are missing, so heuristic grouping cannot eliminate unidentified dependence. Per-class precision/recall/F1, confusion matrices, scores and error tables are under `artifacts/final-test/`. Scores are uncalibrated.

## Additional checks

MFCC with post-split 10 dB Gaussian noise: sentiment macro-F1 0.377, sarcasm 0.566, combined 0.217; same test cohort, supplied-transcript condition held fixed.

Classical training was repeated at seeds 42, 123, 2026 on the same split, evaluated on validation only. See `artifacts/repeated-classical/summary.json` and `aggregate.json`. An artifact-only validation classifier using silence fraction, RMS, peak, slot-seam jump and duration produced sentiment macro-F1 0.440 and sarcasm 0.424. This does not rule out voice/synthesis shortcuts.

## Remaining measurements and limitations

- Batch supplied-versus-ASR evaluation has not run; real ASR and word colouring were verified on an inference probe.
- Final neural test results use one seed; repeated-seed neural comparisons and paired significance tests remain future experiments. The CLI supports configurable repeated training seeds.
- Voice/speaker/template generalization cannot be established without authoritative identities. Unsupported speaker/voice holdouts fail explicitly.
- No human challenge recordings, perceived-emotion labels or natural-boundary annotations were provided. Annotation templates and guarded evaluation/tuning commands are implemented; no human metrics are fabricated.
- Dynamic smoothing/minimum-duration defaults are provisional. Slot-to-window distribution shift and the noise-sensitive energy speech detector limit segmentation.
- Encoder fine-tuning is deferred; the small likely synthetic corpus does not currently justify it.

## Reproduce

Exact setup, audit, train, evaluation, inference and demo commands are in `README.md`. Labels and owner timing confirmation are in `configs/labels.json`. The saved split is in `artifacts/data/splits.json` and `configs/splits-seed42.json`. Dependencies are pinned in `requirements-lock.txt`; model revisions are recorded in `artifacts/pretrained-config.json` and defaults. Private data, caches, checkpoints and generated predictions remain local and gitignored.
