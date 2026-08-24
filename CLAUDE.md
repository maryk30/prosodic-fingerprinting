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

**Stage 1 — Data collection** — [unclaimed]
- [ ] Record 5-10 short (3-10s) genuine clips for 1-2 enrolled speakers (team members)
- [ ] Generate synthetic clones of the same speakers via free-tier TTS API
- [ ] Organize into `data/genuine/<speaker>/` and `data/synthetic/<speaker>/<tts_system>/`

**Stage 2 — Preprocessing** — [unclaimed]
- [ ] VAD + segmentation (webrtcvad or pyannote VAD) — keep pause info, don't discard silence
- [ ] Loudness normalization
- [ ] Implement in `src/preprocessing.py`

**Stage 3 — Prosodic feature extraction** — [unclaimed]
- [ ] F0 contour (librosa pYIN), speaker-relative semitone normalization
- [ ] Energy (RMS) trajectory
- [ ] Speaking rate proxy
- [ ] Pause statistics (count/mean/variance) from VAD output
- [ ] Rhythm metric (nPVI)
- [ ] Jitter/shimmer via Parselmouth
- [ ] Implement in `src/features.py`, write `data/features/features.csv`
- [ ] Finalize the data contract section above once schema is real

**Stage 4 — Fingerprint construction** — [unclaimed]
- [ ] Per-speaker statistical summary vector (mean/var/skew) from enrollment clips
- [ ] Implement in `src/fingerprint.py`

**Stage 5 — Detection model (primary)** — [unclaimed]
- [ ] One-Class SVM trained on genuine fingerprints, scored against held-out genuine+synthetic clips
- [ ] Implement in `src/models/oneclass.py`

**Stage 5b — Stretch: supervised model** — [unclaimed]
- [ ] SVM/RF trained on labeled genuine+synthetic features
- [ ] Implement in `src/models/supervised.py`

**Stage 6 — Stretch: spectral baseline** — [unclaimed]
- [ ] ECAPA-TDNN (SpeechBrain pretrained) cosine-similarity baseline
- [ ] Implement in `src/baseline_spectral.py`

**Stage 7 — Evaluation** — [unclaimed]
- [ ] EER, ROC-AUC, accuracy/precision/recall/F1 in `notebooks/results.ipynb`
- [ ] ROC curve plot, ablation if time permits
- [ ] Implement metric helpers in `src/eval.py`

**Stage 8 — CLI deliverable** — [unclaimed]
- [ ] `cli.py enroll <speaker> <clip1> <clip2> ...` builds and saves a fingerprint
- [ ] `cli.py score <speaker> <clip>` prints genuine/synthetic + confidence score

**Stage 9 — Wrap-up** — [unclaimed]
- [ ] README setup + demo instructions finalized
- [ ] Final status update here: what shipped vs. what's deferred to future work

## End-of-day status
_(fill in before wrapping up — one paragraph: what works, what's a stub, what
to demo, what broke)_
