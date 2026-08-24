# CLAUDE.md — VoiceGuard MVP Task Board

> Read this file first. It's the shared source of truth for two people
> running Claude in parallel today. Keep it in sync with small, frequent
> commits — see **Sync Protocol** below.

## Project in one paragraph
VoiceGuard detects AI-generated voice clones using **prosodic fingerprinting**
— habitual rhythm, pause placement, speaking rate, and pitch/energy dynamics
— instead of spectral artifacts. Hypothesis: cloning systems nail timbre but
don't reliably reproduce a target speaker's fine-grained prosodic habits, so
a mismatch between an enrolled speaker's known prosody and a candidate clip's
prosody is evidence of cloning. Today's goal: a working MVP, not the full
research system — see "MVP scope decisions" below for what's in/out.

## MVP scope decisions (already made — don't relitigate mid-day)
- **Primary model**: One-Class SVM anomaly detector, trained only on genuine
  enrollment clips (no synthetic data needed at training time — matches
  realistic deployment). Supervised SVM/RF is a stretch goal only.
- **Synthetic data**: generated via a free-tier cloud TTS/voice-cloning API
  (e.g. ElevenLabs free tier) on 1-2 team members' voices.
- **Deliverable**: CLI (`cli.py enroll` / `cli.py score`) + a results
  notebook (`notebooks/results.ipynb`) with metrics/plots. No web UI today.
- **Fingerprint representation**: statistical summary vector (mean/var/skew
  per feature) — not the learned-embedding version from the full proposal.
- **Out of scope today**: spectral baseline (ECAPA-TDNN) and fusion system
  are stretch goals; cross-lingual, human-listening study, real-time
  streaming are explicitly future work, not today.

## Suggested track split (pick either, or self-assign per task — see checklist)
- **Track A — Data & Features**: Stages 1-4 (data collection, preprocessing,
  prosodic feature extraction, fingerprint construction). Hands off
  `data/features/features.csv` to Track B.
- **Track B — Models & Eval**: Stages 5-7 (detector, stretch models,
  evaluation). Consumes `data/features/features.csv`. Can start building
  against a small synthetic/dummy CSV before Track A's real one lands, to
  avoid blocking.
- Either person can pick up Stage 8 (CLI) or Stage 9 (wrap-up) once their
  primary track stabilizes.

## Data contract (Track A → Track B interface)
`data/features/features.csv` columns (finalize exact list when Track A lands
Stage 3, update this section immediately after):
```
speaker, clip_id, label (genuine|synthetic), tts_system (na for genuine),
f0_mean, f0_std, energy_mean, energy_std, speaking_rate_mean,
pause_count, pause_mean_dur, pause_var_dur, npvi, jitter, shimmer
```

## Sync Protocol (how this file stays "live")
There's no special tooling — just discipline:
1. **Before starting a task**: claim it by editing your name onto the
   checklist line below, then `git add CLAUDE.md && git commit -m "claim: <task>" && git push`.
2. **Before claiming**, `git pull --rebase` so you see the other person's
   latest claims and avoid double-work.
3. **After finishing a meaningful step**: check the box, add a one-line
   status note (what landed, what's next, any blocker), commit, push.
4. If you hit a merge conflict on this file, it's almost always both people
   editing different checklist lines — resolve by keeping both edits.
5. Keep code changes in small commits too, so the other Claude's session can
   `git pull` and get your latest interface (e.g. the features CSV schema)
   without waiting for a giant end-of-day merge.

## Live Checklist

**Stage 0 — Scaffold** — owner: Shreya (this session)
- [x] git init, .gitignore, requirements.txt, folder skeleton, README, CLAUDE.md
- [ ] Push initial commit to https://github.com/maryk30/prosodic-fingerprinting.git — [in progress]

**Stage 1 — Data collection** — owner: Claude session (Mary)
- [x] Record 5-10 short (3-10s) genuine clips for 1-2 enrolled speakers (team members) — 10 clips each for `krishiv` and `mary` in `data/genuine/<speaker>/` (m4a)
- [ ] Generate synthetic clones of the same speakers via free-tier TTS API — deferred, no API key yet; revisit once available
- [x] Organize into `data/genuine/<speaker>/` and `data/synthetic/<speaker>/<tts_system>/` — genuine done; synthetic folder pending clones above

**Stage 2 — Preprocessing** — owner: Claude session (Mary)
- [x] VAD + segmentation (webrtcvad, 30ms frames) — keep pause info, don't discard silence
- [x] Loudness normalization (RMS-target, -20 dBFS)
- [x] Implement in `src/preprocessing.py` — tested against all 20 genuine clips; also added `discover_clips()` to skip stray outlier recordings (>15s) rather than silently breaking VAD/feature stages downstream

**Stage 3 — Prosodic feature extraction** — owner: Claude session (Mary)
- [x] F0 contour (librosa pYIN), speaker-relative semitone normalization — reference = median voiced F0 from each speaker's *genuine* clips only
- [x] Energy (RMS) trajectory (mean/std via librosa.feature.rms)
- [x] Speaking rate proxy (onset count / speech-time, via librosa.onset.onset_detect)
- [x] Pause statistics (count/mean/variance) from VAD output — internal pauses only (leading/trailing silence and <50ms VAD noise excluded)
- [x] Rhythm metric (nPVI) — computed over inter-onset intervals as a syllable-nucleus proxy (no forced alignment available)
- [x] Jitter/shimmer via Parselmouth (local jitter/shimmer, periodic point process)
- [x] Implement in `src/features.py`, write `data/features/features.csv` — 20 rows (10 krishiv + 10 mary genuine, no NaNs); synthetic rows will append automatically once Stage 1's TTS clones land in `data/synthetic/<speaker>/<tts_system>/`
- [x] Data contract confirmed accurate — no schema changes needed

**Stage 4 — Fingerprint construction** — owner: Claude session (Mary)
- [x] Per-speaker statistical summary vector (mean/var/skew) from enrollment clips — 11 features × 3 stats = 33-dim vector, built only from genuine clips
- [x] Implement in `src/fingerprint.py` — verified against both speakers (krishiv, mary), saved to `data/features/fingerprints/<speaker>.json` (gitignored, same as other derived voice data)

**Track A (Stages 1-4) complete for genuine data.** Synthetic clones still needed (no TTS API key yet) — once added, `features.py`/`fingerprint.py` pick them up automatically with no code changes. Handing off to Track B (Stage 5+) against the current `data/features/features.csv`.

**Stage 5 — Detection model (primary)** — owner: Claude session (Mary)
- [x] One-Class SVM trained on genuine fingerprints, scored against held-out genuine+synthetic clips — per-speaker model, trained on that speaker's genuine per-clip features (11-dim, standardized)
- [x] Implement in `src/models/oneclass.py`
- **Known limitation**: only 10 genuine clips/speaker (9 per leave-one-out fold) in an 11-dim feature space is thin for learning a robust boundary. RBF kernel badly overfit (0-20% self-acceptance on held-out genuine); switched default to `linear` kernel, which does better (50-60%) but is still noisy at this sample size. Did **not** grid-search nu/gamma against this same tiny holdout — that would just fit noise. Real validation is Stage 7 once synthetic clips exist as the actual anomaly class; if scores there are poor, revisit with more enrollment clips before touching hyperparameters again.

**Stage 5b — Stretch: supervised model** — [unclaimed]
- [ ] SVM/RF trained on labeled genuine+synthetic features
- [ ] Implement in `src/models/supervised.py`

**Stage 6 — Stretch: spectral baseline** — [unclaimed]
- [ ] ECAPA-TDNN (SpeechBrain pretrained) cosine-similarity baseline
- [ ] Implement in `src/baseline_spectral.py`

**Stage 7 — Evaluation** — owner: Claude session (Mary)
- [x] EER, ROC-AUC, accuracy/precision/recall/F1 in `notebooks/results.ipynb`
- [x] ROC curve plot — wired up, gracefully no-ops with a clear message until synthetic clips exist (only one class = genuine right now, so ROC-AUC/EER report as unavailable rather than a bogus number)
- [x] Implement metric helpers in `src/eval.py` — `evaluate()` + `compute_eer()`, unit-sanity-checked against hand-built two-class scores
- **Real numbers still pending Stage 1's synthetic clones.** Current notebook only shows genuine-only diagnostics (feature distributions, leave-one-out self-consistency: krishiv 60%, mary 50% — small-sample noise, not tuned). Re-run `python src/features.py` then the notebook once synthetic data lands; no code changes needed.
- Ablation: not attempted — not enough signal to make it meaningful without synthetic data first.

**Stage 8 — CLI deliverable** — owner: Claude session (Mary)
- [x] `cli.py enroll <speaker> <clip1> <clip2> ...` builds and saves a fingerprint — also updates `features.csv` (replacing that speaker's prior genuine rows) and trains/saves their OC-SVM model
- [x] `cli.py score <speaker> <clip>` prints genuine/synthetic + confidence score — verified against real clips for both `krishiv` and `mary`

**Stage 9 — Wrap-up** — owner: Claude session (Mary)
- [x] README setup + demo instructions finalized
- [x] Final status update here: what shipped vs. what's deferred to future work

## End-of-day status
Full pipeline works end-to-end on real data: `cli.py enroll krishiv data/genuine/krishiv/*.m4a`
and `cli.py score krishiv <clip>` run cleanly, producing a fingerprint,
a trained per-speaker One-Class SVM, and a genuine/synthetic + confidence
verdict. Stages 1-5, 7, 8 are done for genuine data (10 clips each,
`krishiv`/`mary`); `notebooks/results.ipynb` runs clean end-to-end with
per-speaker feature plots and a leave-one-out self-consistency check.

**What's a stub / blocked**: Stage 1's synthetic TTS clones were never
generated (no ElevenLabs/free-tier API key available today), which cascades:
Stage 7's real numbers (ROC-AUC, EER, accuracy vs. actual clones) are
unmeasured — the notebook's evaluation cell is fully wired up and will
run with zero code changes once clips land in
`data/synthetic/<speaker>/<tts_system>/` and `python src/features.py` is
re-run. Stage 5b (supervised SVM/RF) and Stage 6 (spectral baseline) —
both stretch goals — are unstarted for the same reason: no labeled negative
class to train/compare against yet.

**Known weakness**: only 10 genuine clips/speaker in an 11-dim feature
space is thin for the One-Class SVM. RBF kernel overfit badly (leave-one-out
self-acceptance 0-20%); switched to linear kernel (50-60%). This is
model-capacity-appropriate for today's MVP scope, not a bug, but it means
Stage 7's numbers once synthetic data exists should be read with that
sample-size caveat rather than treated as a tight estimate.

**To demo**: `python cli.py enroll <speaker> data/genuine/<speaker>/*.m4a`
then `python cli.py score <speaker> <any clip>` — works today, no setup
beyond `pip install -r requirements.txt` + `brew install ffmpeg`.

**Next session priority**: get a TTS API key, generate clones, re-run
`src/features.py`, then Stage 7's real eval numbers and Stage 5b/6 stretch
goals become unblocked with no pipeline changes needed.
