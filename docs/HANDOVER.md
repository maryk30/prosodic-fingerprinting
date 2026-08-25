# VoiceGuard MVP — Handover Document

Technical reference for anyone picking up this project after today's build
session. For the live day-of task tracking, see [`CLAUDE.md`](../CLAUDE.md);
this document is the stable "what exists and how it works" reference.

## 1. What VoiceGuard is

VoiceGuard detects AI-generated voice clones by checking a candidate clip's
**prosody** — habitual rhythm, pause placement, speaking rate, and
pitch/energy dynamics — against an enrolled speaker's known prosodic habits,
instead of relying on spectral/timbre artifacts (the usual approach).

**Hypothesis**: cloning systems reproduce timbre convincingly but don't
reliably reproduce a target speaker's fine-grained prosodic habits. A
mismatch between a speaker's enrolled prosody and a candidate clip's
prosody is evidence the clip is a clone.

## 2. What was built (deliverables)

| Deliverable | Location | Status |
|---|---|---|
| Genuine voice recordings | `data/genuine/<speaker>/*.m4a` | Done — 10 clips × 2 speakers (`krishiv`, `mary`) |
| Synthetic clone recordings | `data/synthetic/<speaker>/<tts_system>/` | **Not done** — needs a TTS API key |
| Preprocessing pipeline | `src/preprocessing.py` | Done |
| Prosodic feature extraction | `src/features.py` | Done |
| Prosodic fingerprint construction | `src/fingerprint.py` | Done |
| Spectral (MFCC) fingerprint construction | `src/spectral_features.py`, `src/spectral_fingerprint.py` | Done |
| Voice fingerprint visuals (spectrogram, MFCC profile, prosodic comparison) | `src/visualize_fingerprints.py` | Done |
| Detection model (One-Class SVM) | `src/models/oneclass.py` | Done (genuine-only training; real eval pending) |
| Evaluation metrics | `src/eval.py` | Done, wired up, pending synthetic data to produce real numbers |
| Results notebook | `notebooks/results.ipynb` | Done, runs clean end-to-end |
| CLI | `cli.py` (`enroll`, `score`) | Done, verified against real clips |
| Supervised model (stretch) | `src/models/supervised.py` | Not started |
| Spectral baseline (stretch) | `src/baseline_spectral.py` | Not started |

Everything above the two stretch rows is implemented, tested against real
recordings, and pushed to `main`.

## 3. Architecture — the pipeline

```
raw clip (.m4a/.wav/...)
        │
        ▼
┌───────────────────────┐
│  preprocessing.py      │  load → mono 16kHz → RMS loudness normalize
│                        │  → webrtcvad speech/silence segmentation
│                        │  → pause list (incl. leading/trailing silence)
└───────────┬────────────┘
            ▼
┌───────────────────────┐
│  features.py           │  per clip, extract 11 prosodic features:
│                        │  F0 (semitone-normalized), energy, speaking
│                        │  rate, pause stats, nPVI rhythm, jitter, shimmer
└───────────┬────────────┘
            ▼
     features.csv  (one row per clip; the Track A → Track B data contract)
            │
   ┌────────┴─────────┐
   ▼                   ▼
┌────────────┐   ┌──────────────────────┐
│fingerprint  │   │  models/oneclass.py   │
│.py          │   │  per-speaker One-     │
│             │   │  Class SVM, trained   │
│per-speaker  │   │  on that speaker's    │
│mean/var/    │   │  genuine feature      │
│skew summary │   │  vectors only         │
│vector       │   └──────────┬───────────┘
└─────────────┘              │
                              ▼
                    genuine / synthetic + confidence
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                cli.py              eval.py + results.ipynb
           (enroll / score)      (accuracy/precision/recall/
                                   F1/ROC-AUC/EER, plots)
```

### 3.1 Preprocessing (`src/preprocessing.py`)

- Loads any librosa/ffmpeg-decodable audio (the team recorded in `.m4a`;
  `ffmpeg` must be installed — `brew install ffmpeg`).
- Resamples to mono 16kHz (required by `webrtcvad`).
- **Loudness normalization**: scales each clip so its RMS level matches a
  fixed target (-20 dBFS), so volume differences between recordings don't
  leak into the energy features.
- **VAD segmentation**: `webrtcvad` (30ms frames, aggressiveness 2) finds
  speech vs. silence regions. Pauses are kept, not discarded — pause
  structure is itself a prosodic feature.
- `discover_clips()` guards against stray long recordings (>15s) — one
  105-second outlier clip was caught and excluded this way rather than
  silently corrupting downstream stats.

### 3.2 Feature extraction (`src/features.py`)

Per clip, 11 numeric features are computed:

| Feature | How | Notes |
|---|---|---|
| `f0_mean`, `f0_std` | librosa `pyin` on voiced frames, converted to **semitones relative to the speaker's own median pitch** | Reference = median voiced F0 across that speaker's *genuine* clips only. Makes the fingerprint comparable across speakers regardless of natural register. |
| `energy_mean`, `energy_std` | RMS trajectory (`librosa.feature.rms`) | |
| `speaking_rate_mean` | onset count (`librosa.onset.onset_detect`) ÷ total speech time | Proxy for syllable rate — no forced alignment available for a true syllable count |
| `pause_count`, `pause_mean_dur`, `pause_var_dur` | gaps between VAD speech segments | Leading/trailing silence and <50ms VAD noise excluded — only "real" internal pauses count |
| `npvi` | Normalized Pairwise Variability Index over inter-onset intervals | Rhythm metric; proxy for syllable-nucleus intervals |
| `jitter`, `shimmer` | Praat via `parselmouth` (local jitter/shimmer, periodic point process) | Standard voice-quality measures |

Output: `data/features/features.csv`, matching the fixed **data contract**:
```
speaker, clip_id, label (genuine|synthetic), tts_system (na for genuine),
f0_mean, f0_std, energy_mean, energy_std, speaking_rate_mean,
pause_count, pause_mean_dur, pause_var_dur, npvi, jitter, shimmer
```
`build_features_table()` walks `data/genuine/` and `data/synthetic/`
automatically — dropping real TTS clones into `data/synthetic/<speaker>/<tts_system>/`
and re-running `python src/features.py` is all that's needed to pick them up.

The per-speaker F0 reference (Hz) is persisted to
`data/features/f0_reference.json` so later `cli.py score` calls can convert
a fresh candidate clip's pitch the same way it was done at enrollment time.

### 3.3 Fingerprint construction (`src/fingerprint.py`)

Per the MVP scope decision, a speaker's fingerprint is **not** a learned
embedding — it's a statistical summary vector: for each of the 11 features,
the **mean, variance, and skew** across that speaker's genuine enrollment
clips (33 numbers total). Saved as `data/features/fingerprints/<speaker>.json`.

### 3.3b Spectral fingerprint (`src/spectral_features.py`, `src/spectral_fingerprint.py`)

Added on request as a companion deliverable to the prosodic fingerprint —
"fingerprints (spectral and prosodic)" for both speakers. Where the
prosodic fingerprint captures habits, the spectral one captures raw
timbral texture, via **MFCCs** (Mel-Frequency Cepstral Coefficients), the
standard representation behind most speaker-ID and spectral anti-spoofing
systems:

1. `spectral_features.py` computes 13 MFCCs per clip (on the same
   preprocessed audio as the prosodic pipeline), reduces each to a
   mean/std per coefficient (26 numbers), and writes
   `data/features/spectral_features.csv`.
2. `spectral_fingerprint.py` collapses a speaker's genuine clips into
   mean/var/skew of those 26 numbers (**78-dim**), reusing
   `fingerprint.py`'s `build_fingerprint()` — it was generalized to take a
   `numeric_features` column list instead of being prosodic-only, so both
   fingerprint types share one implementation.
3. Saved as `data/features/fingerprints/<speaker>_spectral.json`.

This is a **lightweight MFCC summary, not the full pretrained ECAPA-TDNN
embedding** scoped under Stage 6 (stretch) in `CLAUDE.md` — same
no-heavy-dependency approach as the rest of the MVP. `cli.py enroll` now
builds both fingerprint types for a speaker in one call.

**Voice fingerprint visuals** (`src/visualize_fingerprints.py`): renders a
mel-spectrogram and MFCC coefficient profile per speaker, plus a
cross-speaker prosodic comparison. Note: comparing exactly two speakers by
min-max normalizing each feature to [0,1] was tried first and rejected —
with only two points, that trivially puts one speaker at 0 and the other
at 1 on *every* feature regardless of how close the real values are. The
comparison instead uses small multiples, each in its own real units.

### 3.4 Detection model (`src/models/oneclass.py`)

A **per-speaker One-Class SVM** (`sklearn.svm.OneClassSVM`), trained only
on that speaker's genuine per-clip feature vectors (standardized). This
matches realistic deployment: you enroll a known speaker's genuine voice;
you never need synthetic examples at training time to detect an anomaly
later.

- `SpeakerAnomalyDetector.train(speaker, clip_rows, f0_ref_hz)` — fits a
  `StandardScaler` + `OneClassSVM(kernel="linear", nu=0.1)`.
- `.score(feature_row)` — returns `{"prediction": "genuine"|"synthetic", "confidence": float}` (SVM decision function; >0 = inlier).
- `.score_clip(path)` — runs the full preprocessing → features pipeline on
  a raw audio path and scores it directly.
- `.save()` / `.load()` — persists scaler + model + f0 reference via
  `joblib`, to `data/features/models/<speaker>_oneclass.joblib`.

**Why linear, not RBF**: with only ~9 training clips per leave-one-out fold
against an 11-dimensional feature space, RBF badly overfit (0-20%
self-acceptance on held-out genuine clips). Linear does noticeably better
(50-60%). Hyperparameters were **not** grid-searched further against this
same tiny holdout — that would just be fitting noise, not a defensible
tuning process. More enrollment clips is the real fix, not more tuning.

### 3.5 Evaluation (`src/eval.py`, `notebooks/results.ipynb`)

`evaluate(y_true, y_score)` computes accuracy/precision/recall/F1 always,
and ROC-AUC/EER only when both classes (genuine + synthetic) are present —
otherwise it reports them as unavailable (`nan`) rather than a misleading
number. `compute_eer()` finds the ROC threshold where false-positive rate
equals false-negative rate.

The notebook currently shows genuine-only diagnostics (feature
distributions per speaker, leave-one-out self-consistency) because no
synthetic clips exist yet. The real evaluation cell (ROC-AUC, EER, ROC
curve per speaker) is fully implemented and executes without error — it
just has nothing to score against yet, and prints a clear message saying so
instead of silently doing nothing.

### 3.6 CLI (`cli.py`)

```bash
python cli.py enroll <speaker> <clip1> [<clip2> ...]
python cli.py score <speaker> <candidate_clip>
```

- `enroll`: extracts features from the given clips, **replaces** that
  speaker's prior genuine rows in `features.csv`, rebuilds their
  fingerprint, retrains and saves their One-Class SVM model.
- `score`: loads the speaker's saved model and prints a genuine/synthetic
  verdict with a confidence score for the given clip.

Both commands were run against the real `krishiv`/`mary` recordings and
verified to work, including a cross-speaker sanity check (scoring one
speaker's clip against the other's model).

## 4. Known limitations (be upfront about these to the panel)

1. **No synthetic data yet.** The whole point of the system — distinguishing
   genuine from cloned voices — is unmeasured. Everything built so far is
   the genuine-side half of the pipeline, fully working, waiting for the
   negative class.
2. **Small sample size.** 10 genuine clips per speaker is workable for an
   MVP demo but thin for a statistically confident anomaly boundary — the
   linear-vs-RBF kernel finding above is a direct symptom of this.
3. **Speaking-rate and rhythm are proxies, not ground truth.** Without
   forced alignment or a syllabifier, onset detection stands in for
   syllable timing. Reasonable for an MVP, a known simplification for a
   fuller system.
4. **Stretch goals unstarted**: supervised SVM/RF (Stage 5b) and an
   ECAPA-TDNN spectral-embedding baseline (Stage 6) for comparison against
   the prosodic approach — both need labeled synthetic data to be
   meaningful, so intentionally deferred rather than half-built.

## 5. How to resume work

1. Get a free-tier TTS/voice-cloning API key (e.g. ElevenLabs) and generate
   clones of `krishiv` and `mary` reading the same or similar sentences.
2. Drop the clone audio into `data/synthetic/<speaker>/<tts_system>/`.
3. Run `python src/features.py` — regenerates `features.csv` and
   `f0_reference.json` including the new synthetic rows, no code changes.
4. Re-run `notebooks/results.ipynb` end-to-end — the evaluation cell will
   now produce real ROC-AUC/EER/accuracy numbers and ROC curve plots.
5. If numbers look bad, prefer collecting more genuine enrollment clips
   over further hyperparameter tuning (see §3.4).
6. Optionally pick up Stage 5b/6 (stretch) now that a real negative class
   exists.

## 6. Environment setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg   # needed to decode .m4a recordings
```
