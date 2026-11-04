# Human challenge annotation protocol

Keep human recordings separate from the primary likely synthetic dataset. Obtain informed consent for recording, local processing, annotation, retention, and any sharing. Do not publish identifiable recordings without explicit permission. Recruit multiple speakers and record varied devices, rooms, pacing, and natural emotion transitions. This is a suggested protocol, not a claim that human data has been collected.

Use `words-voice challenge-template --out artifacts/challenge.json`. Replace the example with real records; paths may be absolute or relative to the annotation file. Assign stable anonymized speaker IDs, source IDs, script IDs, and template IDs where known. Use genuinely new scripts; avoid paraphrases of training reviews. Keep every speaker/source in one partition. Have a validation portion for postprocessing selection and a locked test portion. Include both intended opinions and both sarcasm classes; the review evaluation command fails if any required class is absent.

Each record needs `origin: "human"`, `consent_obtained: true`, a nonempty human transcript, and `partition: "validation"` or `"test"`. Set sentiment to `positive` or `negative` based on the intended opinion. Set sarcasm to `genuine` or `sarcastic`. “Great, it broke after one day” is sarcastic negative. Do not force ambiguous/mixed opinions into a binary label: use null, document the ambiguity, and exclude it from complete-case comparison. Annotators must not see model predictions. Collect independent labels from at least two annotators, record disagreements, and adjudicate before evaluation. Report agreement before adjudication (e.g. Cohen's kappa for categorical labels) and cohort exclusions.

For perceived emotion, use the codes `ang`, `sad`, `neu`, `hap` to correspond to the primary dataset vocabulary, while documenting perceived emotion separately from requested TTS emotion. Discuss the operational interpretation of each code with annotators before data collection. Mark uncertain/out-of-vocabulary cases and do not silently force them into this vocabulary. Temporal evaluation accepts only fully adjudicated recordings within the defined vocabulary.

Add `temporal_ground_truth: true`, `annotators: ["annotator-A", "annotator-B"]`, and a contiguous, nonoverlapping list:

```json
"regions": [
  {"start": 0.0, "end": 0.8, "emotion": "silence"},
  {"start": 0.8, "end": 3.6, "emotion": "neu"},
  {"start": 3.6, "end": 7.2, "emotion": "ang"}
]
```

Cover the entire waveform, including silence. Boundaries are seconds in original audio; record annotation uncertainty in a separate note. Natural emotion changes must be manually annotated, never copied from ten-second training slots. Boundary evaluation uses one-to-one nearest matches within a preregistered 0.5 s tolerance; report precision/recall and matched error, including missed and extra boundaries. Duration-weighted classification uses 100 ms bins but group bootstrap resamples sources, never bins. Fixed-slot output is the comparison baseline.

```bash
words-voice challenge-evaluate --annotations artifacts/challenge.json --checkpoint artifacts/models/tfidf.joblib --out artifacts/human/text
words-voice tune-regions --annotations artifacts/challenge.json --checkpoint artifacts/models/emotion.joblib --out artifacts/regions.json
words-voice temporal-evaluate --annotations artifacts/challenge.json --checkpoint artifacts/models/emotion.joblib --regions-config artifacts/regions.json --out artifacts/human/temporal.json
```

The commands reject missing annotations and detected training/validation source/transcript overlap. A similarity check cannot establish independence when source metadata are absent. Analyze results by speaker, microphone, script, and recording condition; do not pool human and synthetic metrics.
