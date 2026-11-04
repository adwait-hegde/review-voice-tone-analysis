# Words vs. Voice: Emotion-Aware Customer Review Analysis

A reproducible Python/PyTorch coursework project testing whether transcript and acoustic representations help classify **intended** opinion and sarcasm. It includes a separate audio emotion classifier, postprocessed emotion regions, timestamped ASR, and a Streamlit demo. Original JSON/WAV files are read-only inputs; all derived data go under `artifacts/`.

## Dataset findings

The supplied `wav/` directory contains 336 paired recordings, 9,120 seconds (2 h 32 min), and 912 labelled segments. Four emotion codes (`ang`, `sad`, `neu`, `hap`) each occur 228 times. Each of the four owner-verified review categories has 84 recordings. No missing pairs or invalid metadata/audio were found in the initial audit. The owner confirmed that `g`/`s` means genuine/sarcastic, `pos`/`neg` means intended opinion, and `idx` denotes consecutive 10-second slots. This evidence is recorded centrally in `configs/labels.json`.

Metadata (`tts_overlong`, stretch, compression, loudness adjustment) strongly suggests TTS generation. Treat primary results as **likely synthetic-data results**, not evidence of human-speech generalization. The generation code and speaker/voice/script/source/template identities were not supplied. The emotion codes are retained verbatim and represent supplied metadata conditions, not human-verified perceptions. No prefix is silently guessed; unknown prefixes have null review labels and unknown emotion codes are rejected.

The saved split has 207/63/66 recordings in train/validation/test, forming 248 connected groups. These include shared identifiers when available, exact PCM, exact segment/review text and near-duplicate segment/review text. Grouping uses stateless character-ngram cosine similarity at 0.85; it does not fit a predictive vocabulary to test data. Signature and repetition suffix are not interpreted as identity. Missing generation identifiers mean that unseen templates/voices cannot be guaranteed. The split fractions are group-based, so recording fractions can differ from 60/20/20.

## Setup

The existing `.venv` is ready to use: `source .venv/bin/activate`. For a fresh environment, follow the commands below with an installed Python 3.12 interpreter.

Python 3.11+; development was verified on Python 3.12 / Apple Silicon / 16 GB RAM. CPU is the default; set `device` in the JSON config to `cuda` for an available GPU. Deterministic algorithms are enabled; unsupported deterministic device operations fail rather than quietly changing behavior.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e '.[dev,asr]' --no-deps
export HF_HOME="$PWD/.cache/huggingface"
pytest -q
```

`requirements-lock.txt` pins the tested environment. On another platform, install `pip install -e '.[dev,asr]'` to resolve compatible wheels and save a new freeze for the experiment. No credentials are needed for the public model weights. Checkpoints are trusted-local joblib/PyTorch objects; do not load untrusted pickle files.

## Audit → manifest → split

```bash
words-voice audit --data wav --labels configs/labels.json --out artifacts/data
words-voice split --manifest artifacts/data/manifest.json --out artifacts/data/splits.json
```

For a relocated dataset, pass its path to `audit`; manifest fingerprints ignore absolute paths. The audit emits schema-validated records and reports malformed pairs, duration/sample-rate disagreements, duplicate PCM/transcripts, missing identifiers, ID patterns, class codes and timing evidence. The adapter checks semantic constraints beyond `docs/manifest.schema.json`. Invalid pairs cause a CLI failure after writing the report; review exclusions before training. Do not change split/mapping files after training: checkpoint evaluation checks fingerprints and assignments.

Slot boundaries are `idx * 10` to `min((idx+1)*10, recording duration)`. `speech_seconds` is preserved for inspection only; it never defines the full slot or a true speech/emotion boundary. Without verified timing, only explicit boundaries or single-label whole recordings are usable for emotion supervision; unresolved segments are excluded, visibly counted. Review tasks can continue independently.

## Lightweight vertical slice

```bash
words-voice train --model majority --out artifacts/models/majority.joblib
words-voice train --model tfidf --out artifacts/models/tfidf.joblib
words-voice train --model mfcc --smoke --out artifacts/models/mfcc-smoke.joblib
words-voice train --model mfcc --out artifacts/models/mfcc.joblib
words-voice train --model mfcc --task emotion --out artifacts/models/emotion.joblib
words-voice evaluate --checkpoint artifacts/models/tfidf.joblib --partition validation --out artifacts/results/tfidf-validation
words-voice evaluate --checkpoint artifacts/models/mfcc.joblib --partition validation --out artifacts/results/mfcc-validation
words-voice evaluate --checkpoint artifacts/models/emotion.joblib --partition validation --out artifacts/results/emotion-validation
```

Smoke mode limits training to approximately 24 examples (preserving classes) and neural training to two epochs; its checkpoints and exported metrics are marked `smoke_only`. It is not a full experiment. Majority, TF-IDF and MFCC training use separate heads for sentiment and sarcasm. TF-IDF uses unigram/bigram features. MFCC uses 20 coefficients with means/stds plus first-difference means/stds (80 dimensions), with train-only standardization. Logistic C is selected from 0.1, 1, 10 on validation macro-F1. Missing task labels are masked; no classifier creates ground truth.

## Pretrained representations and fusion

Estimated weights: MiniLM ~90 MB, wav2vec2-base ~380 MB, Whisper tiny ~75 MB; allow ~0.6 GB download and several GB temporary/runtime RAM. Audio is processed one five-second chunk at a time. Start with a smoke run to measure actual CPU cost. Downloads are explicit; inference otherwise uses only cached weights.

```bash
python scripts/prepare_models.py --allow-download
words-voice --config artifacts/pretrained-config.json train --model neural_text --smoke --out artifacts/models/neural_text-smoke.joblib
words-voice --config artifacts/pretrained-config.json train --model neural_audio --smoke --out artifacts/models/neural_audio-smoke.joblib
words-voice --config artifacts/pretrained-config.json train --model fusion --smoke --out artifacts/models/fusion-smoke.joblib
# Full heads over frozen encoders, after smoke verification:
words-voice --config artifacts/pretrained-config.json train --model neural_text --out artifacts/models/neural_text.joblib
words-voice --config artifacts/pretrained-config.json train --model neural_audio --out artifacts/models/neural_audio.joblib
words-voice --config artifacts/pretrained-config.json train --model fusion --out artifacts/models/fusion.joblib
words-voice --config artifacts/pretrained-config.json train --model neural_audio --task emotion --out artifacts/models/neural_emotion.joblib
```

The preparation script resolves immutable Hub revisions into the experiment config. The default config also records the revisions used during implementation. For reproducible reruns, retain this config and cache. MiniLM token representations are attention-mask mean pooled; long texts are chunked and chunk means are weighted by valid token count. Audio chunks are encoded individually without batch padding (appropriate for wav2vec2 group normalization), valid output frames are mean pooled, and chunk means are weighted by original sample count. Short convolution tails are padded with an explicit output mask. These are order-insensitive aggregate representations. Fusion concatenates text and audio vectors, applies a train-only standardizer, then a shared MLP and separate task heads. Neural epochs are selected on validation macro-F1. No IDs, emotion labels, generation metadata or split information enter features.

Encoders remain frozen. Optional end-to-end fine-tuning is intentionally deferred: the small likely synthetic dataset and unknown voices/templates do not yet justify it. The implemented contribution is task-head training and controlled multimodal comparison, not new pretrained encoders.

## Evaluation and experiments

```bash
# Repeat training seeds on the SAME saved split; validation by default:
words-voice suite --models majority tfidf mfcc --out artifacts/repeated-classical
words-voice --config artifacts/pretrained-config.json suite --models neural_text neural_audio fusion --out artifacts/repeated-neural
# Evaluate the six already-selected checkpoints on an identical test cohort:
python scripts/evaluate_all.py --partition test --out artifacts/final-test
# Or evaluate an individual model:
words-voice evaluate --checkpoint artifacts/models/fusion.joblib --partition test --out artifacts/results/fusion-test
# Or run a predeclared repeated-seed final protocol:
words-voice suite --models majority tfidf mfcc --final-test --out artifacts/final-classical
# Noisy test audio: noise is added after split, retaining original identity/group:
words-voice evaluate --checkpoint artifacts/models/mfcc.joblib --noise-snr 10 --out artifacts/results/mfcc-noisy-test
# Supplied vs ASR transcripts: identical eligible test IDs:
words-voice transcribe --out artifacts/asr.json --asr-model tiny.en
words-voice evaluate --checkpoint artifacts/models/tfidf.joblib --transcripts artifacts/asr.json --out artifacts/results/tfidf-asr-test
# Shortcut analysis is validation-only:
words-voice shortcuts --out artifacts/shortcuts.json
# These fail clearly for this dataset because identifiers are missing:
words-voice split --holdout voice --out artifacts/data/voice-splits.json
words-voice split --holdout speaker --out artifacts/data/speaker-splits.json
```

The same complete-case eligible cohort is used across compared review models. A wholly missing task has no trained head; partially missing training labels use masked losses. Evaluation fails for required missing classes. Outputs contain macro-F1, accuracy, per-class precision/recall/F1, confusion matrices, group-bootstrap 95% intervals, subgroup metrics when identifiers exist, hardware/latency, predictions and `errors.csv` with IDs/true labels/predictions/scores. Cohort fingerprints permit comparison checks. CI resampling uses independent source groups, never overlapping windows. Cached feature timing is explicitly distinguished from uncached end-to-end inference. Scores are **uncalibrated**, not reliable confidence estimates.

`configs/experiments.json` records the comparison matrix. Noise is seeded Gaussian noise at requested SNR and is not clipped after addition. This experiment changes acoustic input while holding transcript condition fixed; ASR robustness to noisy speech requires separately transcribing the noisy waveform. Fixed/dynamic natural emotion boundary comparison requires separately annotated temporal ground truth; do not use slot labels as boundary truth.

## Inference and demo

```bash
words-voice infer wav/s-pos_1_sad_r1.wav \
  --checkpoint text-only=artifacts/models/tfidf.joblib \
  --checkpoint audio-only=artifacts/models/mfcc.joblib \
  --checkpoint fusion=artifacts/models/fusion.joblib \
  --emotion-checkpoint artifacts/models/emotion.joblib \
  --asr-model tiny.en --out artifacts/inference.json
streamlit run app.py
```

For offline classification without ASR, pass `--text "your supplied transcript"`. Plain text has no timestamps and is not word-coloured. ASR returns actual word timestamps; where only ASR segment timestamps exist, colouring applies to segments. Colour is assigned from predicted emotion overlap, never ground-truth emotion. Missing checkpoints or ASR weights give explicit messages while other available models continue. The demo includes audio playback, predictions and combined category, a comparison table, an emotion legend, timeline, timestamped transcript and JSON download. A separate clearly labelled dataset-inspection view contains metadata ground truth.

Dynamic inference uses overlapping short audio windows, overlap-weighted probability aggregation, smoothing and merging of short same-emotion regions. An energy-based speech-activity detector marks silence separately. Its threshold is sensitive to noise; this is **postprocessed segmentation**, not a validated natural emotion-boundary detector. Smoothing/minimum duration defaults are provisional, and `tune-regions` selects them only with a separate annotated validation set. Training on 10-second slots and predicting shorter windows introduces a distribution shift.

## Human challenge and reporting

See [annotation guidelines](docs/ANNOTATION.md), [report template](docs/REPORT_TEMPLATE.md), and [implementation results](docs/IMPLEMENTATION_RESULTS.md). Human data and natural boundary annotations have not been provided. Challenge evaluation rejects missing consent/identity, unannotated boundaries and detected source/transcript overlap with training/validation. Keep human and synthetic results separate. No human generalization or fusion improvement is claimed by this implementation.

## Repository map

- `src/words_voice/data.py`, `splits.py`: audit, mapping, manifest, grouping and immutable split checks.
- `features.py`, `models.py`, `experiment.py`: cached features, frozen encoders, masked heads, training, checkpoints and evaluation.
- `inference.py`: ASR, waveform-only inference, dynamic emotion regions and timestamp colouring.
- `challenge.py`: human challenge validation/evaluation, temporal tuning and shortcut diagnostics.
- `cli.py`, `app.py`: command-line and Streamlit entry points.
- `configs/`: verified labels, hardware-friendly defaults and experiment protocol.
- `tests/`: consequential behavior tests, including tiny randomly initialized local encoder fixtures. Fixtures are engineering tests, never evaluation results.

Official API references checked during implementation: [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [Transformers wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Streamlit upload](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader), [audio playback](https://docs.streamlit.io/develop/api-reference/media/st.audio).
