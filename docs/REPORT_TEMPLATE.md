# Words vs. Voice: Emotion-Aware Customer Review Analysis

## Abstract
State the question, primary dataset provenance, evaluation unit, main observed results and uncertainty. Do not claim that fusion improves anything until the controlled results support it.

## Research questions and hypotheses
Does audio add information about intended sentiment/sarcasm beyond transcript text? Does fusion outperform each unimodal model on the same held-out examples? How robust are predictions to ASR errors and noise? Do fixed-slot emotion classifiers support useful postprocessed timelines, without claiming validated natural boundary detection?

## Dataset and labels
Report counts, duration, label distributions and audit exclusions. Cite `configs/labels.json` for owner-verified class/timing meanings. Explain why requested TTS emotion differs from perceived emotion. State evidence of synthetic origin and what remains unverified. Report missing speaker/voice/script/source metadata and implications for independence.

## Methods
- Pretrained components: MiniLM text encoder; wav2vec2-base audio encoder; Whisper tiny ASR. Record exact snapshot revisions, model licenses, training provenance, and library lock file.
- Trained here: TF-IDF logistic heads, MFCC logistic heads, frozen-representation MLP heads, fusion MLP and segment emotion classifier. Optional unfreezing was deferred unless separately justified.
- Engineering contribution: audit, evidence-based metadata adapter, grouping, masked tasks, reproducible checkpoints, controlled comparisons, timeline postprocessing and demo. Do not describe these as established research novelty.
- Explain training-only vectorizer/normalizer fitting, masks, mean pooling, chunking, cache keys, separate task heads, complete-case evaluation, validation selection, seed list and fixed test protocol.

## Experimental protocol
Record split fingerprint, group construction, threshold, class coverage, fraction counts, metadata limitations, known duplicates, and unused test status. List hardware, elapsed time, cached/uncached latency and resource use. Preserve exact commands/configs.

## Results (fill only from exported measurements)
| Condition | Model | Seed | N / groups | Sentiment macro-F1 (95% CI) | Sarcasm macro-F1 (95% CI) | Combined macro-F1 (95% CI) | Accuracy |
|---|---|---|---|---|---|---|---|

Add per-class precision/recall/F1 and labelled confusion matrices. Aggregate repeated seeds with mean and standard deviation; group-bootstrap CI describes sampling uncertainty conditional on a trained model, not all uncertainty. Report same-ID cohort fingerprints. For incomplete labels, report task-specific exclusions and the complete-case comparison cohort. Do not call raw softmax or logistic scores reliable confidence.

Separately report segment emotion, ASR condition, noisy-audio condition, identifiable voice/template holdouts, human challenge set, and temporal boundary evaluation (only if annotated). Label smoke runs and fixture tests as engineering verification, never scientific evidence.

## Error and shortcut analysis
Use `errors.csv`: investigate sarcasm direction, emotion/sentiment disagreement, negation, transcript errors, low-energy speech, silence/stitching, templates, device/voice, and known near duplicates. Inspect false positives and negatives without selecting new hyperparameters on test. Explain the artifact-only diagnostic and residual confounding.

## Limitations
Synthetic-to-human gap; unknown generation and voice identities; scripted language; limited effective group count; heuristic near-duplicate detection; only requested emotion supervision; ten-second slot labels and window mismatch; energy detector noise sensitivity; uncalibrated scores; ASR timestamp uncertainty; chunk mean pooling loses order; no validated perceived emotion/boundaries without human labels.

## Conclusion
Summarize measured findings with uncertainty and scope. No human generalization claim without human evaluation, no unsupported improvement or novelty claims.
