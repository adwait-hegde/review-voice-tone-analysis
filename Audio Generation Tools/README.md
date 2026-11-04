# Synthetic emotion + sarcasm review speech (ENT-compatible)

Generates spoken consumer-product reviews labelled with:

- **emotion per 10 s segment**: ENT's 4 IEMOCAP classes `neu`, `hap`, `ang`, `sad`
- **tone per clip**: sarcastic or genuine
- **polarity per clip**: positive (recommends buying) or negative

Clips are 10 to `MAX_SEGMENTS`×10 s long. Each case is one (tone, polarity, emotion sequence, redundancy copy). The first run uses `MAX_SEGMENTS = 3` and `REDUNDANCY = 1`: **336 clips, 912 segments, about 2.5 h of audio**.

## Running it

```powershell
powershell -ExecutionPolicy Bypass -File run.ps1                  # run until done
powershell -ExecutionPolicy Bypass -File run.ps1 -Pause           # running stages wait after the current segment
powershell -ExecutionPolicy Bypass -File run.ps1 -Resume          # ...and carry on
powershell -ExecutionPolicy Bypass -File run.ps1 -MaxMinutes 60   # a time-boxed session instead
powershell -ExecutionPolicy Bypass -File run.ps1 -Pilot           # 24-clip pilot
```

**Pause and resume:**
- `-Pause` creates `data\PAUSE`. Synthesis and QC finish their current item and then wait, with the model still loaded, so resuming is instant.
- `-Resume` deletes the flag. While paused, the GPU is idle but its memory stays reserved.
- If you start `run.ps1` while the flag exists, it waits too.

**Progress:** each run logs to `data\run_<date>_<time>.log`. To watch it live, use `Get-Content <log> -Wait -Tail 5`.

`run.ps1` runs five stages in order: build cases → synthesize → assemble → QC → export.

**Data folder (`ROOT`):**
- **What's in it:** the models, both Python environments and all generated audio, which comes to tens of GB. It must not be inside the repo or a synced folder such as OneDrive.
- **Location:** the default is `%USERPROFILE%\ent_synth`. Every `data\...` path in this README is inside it.
- **Choosing another location:**
  - for one run: `run.ps1 -Root D:\ent_synth`
  - for every run: `setx ENT_SYNTH_ROOT D:\ent_synth`, then open a new terminal
- **Using `-Root`:** pass the same `-Root` to `-Pause` and `-Resume` too, or they look for the pause flag in the default folder.
- **Running a stage by hand:** if you run a `python -m synth.<stage>` command directly, set `ENT_SYNTH_ROOT` in that terminal first.

**One-time setup** (Windows, an NVIDIA GPU with 8 GB or more, [uv](https://docs.astral.sh/uv/), git). Below, `<ROOT>` is the data folder.
1. **IndexTTS:** clone [index-tts](https://github.com/index-tts/index-tts) into `<ROOT>\index-tts`, run `uv sync` there, and download `IndexTeam/IndexTTS-2.5` from Hugging Face into `<ROOT>\index-tts\checkpoints\`.
2. **Tools environment:** create `<ROOT>\venv-tools` with Python 3.11 (`uv venv --python 3.11 <ROOT>\venv-tools`) and install [requirements-tools.txt](requirements-tools.txt) into it. The QC models download into `<ROOT>\qc_models\` on first use.
3. **ENT:** clone [ENT](https://github.com/ECNU-Cross-Innovation-Lab/ENT) into `<ROOT>\ENT`. The export uses its character table and its own `get_labels` for `--check`.
4. **CREMA-D ratings:** clone [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) into `<ROOT>\datasets\CREMA-D` without its Git LFS audio: `$env:GIT_LFS_SKIP_SMUDGE = "1"` before `git clone`. Only the ratings and demographics are used. `prepare_cremad` downloads the few clips it needs on its own.
5. **Voices and emotion references:** from this folder, run `<ROOT>\venv-tools\Scripts\python.exe -m synth.prepare_cremad`. It builds `<ROOT>\voices\` and `<ROOT>\emo_refs\`.

**Stopping and resuming:**
- You can stop it any time: Ctrl+C, closing the window, killing the process, or sleep/hibernate.
- Rerun the same command to continue. Each stage skips finished work.
- All writes go to a `*.tmp` file and are renamed into place, so a hard stop never leaves a half-written file that looks finished.
- The most you lose is the one segment in progress.
- Ctrl+C once finishes the current segment and stops. Twice stops immediately.

**Background sessions:** to run minimized across the week, start sessions like this from this folder:

```powershell
Start-Process powershell -WindowStyle Minimized -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\run.ps1`" -MaxMinutes 90"
```

**Re-rendering:** after QC, `run.ps1 -Rerender` regenerates the segments listed in `data\rerender.txt` (failed WER, speaker or format checks).

**Speed:** plugged in, a 10 s segment takes about 13 s to generate, so all 912 segments take about 3.5 h. On battery it is much slower: about 60 s per segment, measured before the pinning fix. Synthesis is limited by the one CPU thread that feeds the GPU, not by total CPU, RAM or the GPU itself, so low overall CPU use is normal. What it does to keep that thread fast:
- **Fast cores:** it pins itself to the CPU's performance cores. Windows had been moving it onto the efficiency cores, which made each segment take twice as long.
- **No repeated analysis:** each voice and emotion recording is analyzed once and then reused.

Other things that help:
- Plug in.
- Set Windows power mode to *Best performance*.
- Keep the laptop on a hard surface.
- Avoid heavy Chrome use or Windows Defender scans while it runs.

A second generator in parallel doesn't fit: one uses 4.7 GB of the 8 GB GPU.

The stages ask Windows not to idle-sleep while they work. Closing the lid still pauses the run, and it resumes when the laptop wakes.

## Pilot results (24 clips, 48 segments, CREMA-D voices)

| Check | Result |
|---|---|
| Format | All clips 16 kHz mono, exactly k×10 s; 0 failures |
| ENT compatibility | `export_ent --check`: frame labels identical to ENT's `get_labels` on all 12 single-emotion clips |
| Transcript (Whisper WER) | 5.2% mean |
| Emotion (emotion2vec+, ENT's 4 classes) | 69% overall: hap 100%, ang 75%, sad 67%, neu 43%; genuine 79%, sarcastic 58% |
| Retries | 9 of 48 segments regenerated faster for running past 10.8 s; word ranges recalibrated since |
| Stop/resume | Synthesis killed twice and assembly killed mid-run; each resumed with no lost or corrupted files, and the resumed WAVs are byte-identical to an uninterrupted run |

For comparison, the first pilot with IndexTTS demo voices and emotion vectors scored 40% (neutral 0%), which is why emotion now comes from references.

**Still to do yourself:** listen to about 20 clips in `data\wav\` and label sarcastic vs. genuine and the emotions blind. The file name holds the answer, e.g. `s-neg_3_neu-ang-hap_r1` = sarcastic, negative, neutral → angry → happy. The classifier numbers are only a proxy.

**Growing the dataset:** raise `MAX_SEGMENTS` or `REDUNDANCY` in [synth/config.py](synth/config.py) and rerun. Case IDs are stable, so only new cases are added. Existing audio is reused.

## Layout

| Path | What |
|---|---|
| [synth/config.py](synth/config.py) | All settings: case space, word-count targets, emotion mode/strength, QC thresholds |
| [synth/prepare_cremad.py](synth/prepare_cremad.py) | One-time: CREMA-D actors → `voices\` prompts and `emo_refs\` emotion references |
| [synth/build_cases.py](synth/build_cases.py) | Enumerates cases and renders each segment's text from the bank → `data\cases.jsonl` / `cases.csv` |
| [synth/synthesize.py](synth/synthesize.py) | IndexTTS-2.5, one call per 10 s segment (index-tts venv) |
| [synth/assemble.py](synth/assemble.py) | Fits segments to exactly 10.0 s and joins them. Speech is placed toward its neighbours, so each emotion change sounds like an ordinary ~0.7 s sentence break (spare time goes to the clip's start and end, never outside the segment's labelled window). Normalizes loudness per clip. |
| [synth/qc.py](synth/qc.py) | Emotion classifier agreement, Whisper WER, speaker similarity, format checks |
| [synth/export_ent.py](synth/export_ent.py) | `data\manifest.jsonl` + `data\ent_synth.pkl` in ENT's mixed-data field layout |
| [bank/](bank/) | Products, fictional brands, fillers, content and reaction sentences |

Heavy files live in the data folder (`<ROOT>`, see [Running it](#running-it)), not in the repo:
- `index-tts\`: the model and its venv
- `venv-tools\`
- `ENT\`: ENT's code, for its character table and label check
- `qc_models\`: the QC classifiers
- `datasets\`: CREMA-D ratings and the clips used
- `voices\`
- `emo_refs\`
- `data\`: cases, segments, clips (`wav\`), QC results, `manifest.jsonl`, `ent_synth.pkl` and run logs

## How the text is built

Each segment is a **content sentence** plus a **reaction sentence**, which keeps tone, polarity and emotion as separate, reusable parts:

- **Content sentences** are chosen by role (`solo` for 10 s clips; otherwise `open`, `body`…, `close`), tone and polarity.
- **Reaction sentences** are chosen by emotion, tone and polarity.

Slots are filled with 150 product types (each with its own good parts and failure points), 625 fictional two-word brands, and time/person/price fillers. Pairs are picked to fit each emotion's word-count window.

Sarcasm comes from the wording, such as ironic praise and mock complaints, because IndexTTS has no sarcasm control. (`SARCASM_BLEND` only applies in vector mode.)

## Output format

`ent_synth.pkl` is a list of dicts with the fields ENT's `dataset/IEMOCAP/mix.py` produces:
- `audio`, `audio_length` (wav2vec2 frames), `text`, `tlabel`
- `frame_tlabel`, `label` (merged interval labels), `frame_label`
- their run lengths

It also adds:
- `id`, `duration`
- `interval` (`[{emo, start, end}]`), `emotion_seq` (per 10 s)
- `sarcasm` (0/1), `polarity` (1 = recommends)

Emotion IDs follow ENT: `hap 0, ang 1, sad 2, neu 3`. `export_ent --check` confirms the frame labels match ENT's own `get_labels` on every clip ENT's version can represent.

Things to know when training with ENT:
- **Multiple emotions per clip:** ENT's released `get_labels` and IEMOCAP collate assume one emotional interval, while these clips can have up to `MAX_SEGMENTS`.
- **Length filter:** `IEMOCAP.Merge` drops clips whose `audio_length` ≥ 300000. Here `audio_length` is in wav2vec2 frames, as in `mix.py`, so the filter keeps them.

## Voices and emotion

Voices and emotion both come from **CREMA-D**: 91 English actors, each recorded in six emotions, every clip rated by about 10 listeners. [synth/prepare_cremad.py](synth/prepare_cremad.py) is run once and does the following:

- **Picks actors:** the 12 women and 12 men whose neutral, happy, angry and sad clips listeners recognized best. It downloads only those clips.
- **Builds voice prompts:** joins each actor's best neutral clips into a ~10 s prompt in `voices\`.
- **Builds emotion references:** joins their best clips of each emotion into `emo_refs\<voice>\<emotion>.wav`.
- **Writes `voices\voices.csv`.**

Each segment is then synthesized in that actor's voice, using that actor's own reference for the segment's emotion (`EMOTION_MODE = "audio_ref"`, `EMO_ALPHA` per emotion).

**Why references instead of IndexTTS's emotion vectors:** in the pilot, vectors with IndexTTS's demo voices gave no recognizable neutral or sad. emotion2vec+ heard happy or angry for every setting, including a zero vector. CREMA-D references fixed neutral and kept happy and angry distinct. **Sad is still the weakest**: the classifier mostly hears it as neutral, although it is clearly slower. The pilot's demo-voice outputs are kept in `data\pilot_demo_voices\` for comparison.

**Other sources:** to swap in IEMOCAP references later, put `<emotion>.wav` files under `emo_refs\<voice_id>\` (and the voices in `voices.csv`).
- Don't use IEMOCAP speakers as voices if you evaluate on IEMOCAP.
- Don't use ZED clips at all, since ZED is ENT's diarization test set.

## QC

QC uses three models:
- **Emotion:** emotion2vec+ large, scored on ENT's 4 classes. SpeechBrain's IEMOCAP wav2vec2 classifier was tried first, but it labelled every IndexTTS segment `ang` at p = 1.0, so it isn't used.
- **WER:** Whisper `small.en`.
- **Speaker similarity:** SpeechBrain ECAPA.

Classifier agreement is reported, not used to filter (`RERENDER_ON_SER_MISMATCH = False`), because these classifiers are themselves unreliable on synthetic speech. The listening check is the real test.

## Licenses

- **IndexTTS weights:** bilibili Model Use License. See `<ROOT>\index-tts\INDEX_MODEL_LICENSE*` before redistributing generated audio.
- **CREMA-D:** Open Database License; contents under the Database Contents License. Cite Cao et al., "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset", IEEE Trans. Affective Computing, 2014.
- **IEMOCAP:** needs a license request to USC.
