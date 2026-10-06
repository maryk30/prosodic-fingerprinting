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
pause_count, pause_mean_dur, pause_var_dur, npvi, jitter, shimmer,
f0_range, f0_slope, f0_velocity, f0_final_move, energy_slope,
voiced_fraction, varco_v, varco_uv          # added Stage 3b (2026-10-05)
```
Stage 3b columns are register-independent (semitones/dB relative to the
clip's own median). Feature groups for the detector live in
`src/fingerprint.py` (`PROSODIC_FEATURES`, `VOICE_QUALITY_FEATURES`,
`REGISTER_FEATURES`); `energy_mean` is in no group (loudness-normalized →
mostly mic/room).

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
- [x] Push initial commit to https://github.com/maryk30/prosodic-fingerprinting.git

**Stage 1 — Data collection** — owner: Claude session (Mary)
- [x] Record 5-10 short (3-10s) genuine clips for 1-2 enrolled speakers (team members) — 10 clips each for `krishiv` and `mary` in `data/genuine/<speaker>/` (m4a)
- [ ] Generate synthetic clones of the same speakers via free-tier TTS API — deferred, no API key yet; when done, clone sentences s21–s30 only, using s01–s20 as reference audio (see `docs/RECORDING_SCRIPT.md`)
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
- [x] **Added (2026-08-25): spectral fingerprint counterpart**, per explicit ask — a "fingerprints (spectral + prosodic)" deliverable for both speakers. 13 MFCCs → mean/std per clip → mean/var/skew per speaker (78-dim), in `src/spectral_features.py` + `src/spectral_fingerprint.py`, reusing `fingerprint.py`'s `build_fingerprint()` generically (it now takes a `numeric_features` param instead of being prosodic-only). Saved as `data/features/fingerprints/<speaker>_spectral.json`. `cli.py enroll` now builds both fingerprint types. This is a lightweight MFCC summary, not the full pretrained ECAPA-TDNN embedding scoped under Stage 6 below — same "no heavy dependency" spirit as the rest of the MVP.
- [x] Panel-facing visual deliverable: `src/visualize_fingerprints.py` renders each speaker's mel-spectrogram + MFCC profile + a cross-speaker prosodic comparison (small multiples in real units — min-max normalizing only 2 speakers to [0,1] was tried first and found to trivially hide magnitude, so real units were used instead). Published as a separate panel artifact alongside the main handover page.

**Track A (Stages 1-4) complete for genuine data.** Synthetic clones still needed (no TTS API key yet) — once added, `features.py`/`fingerprint.py` pick them up automatically with no code changes. Handing off to Track B (Stage 5+) against the current `data/features/features.csv`.

**Stage 5 — Detection model (primary)** — owner: Claude session (Mary)
- [x] One-Class SVM trained on genuine fingerprints, scored against held-out genuine+synthetic clips — per-speaker model, trained on that speaker's genuine per-clip features (11-dim, standardized)
- [x] Implement in `src/models/oneclass.py`
- **Known limitation**: only 10 genuine clips/speaker (9 per leave-one-out fold) in an 11-dim feature space is thin for learning a robust boundary. RBF kernel badly overfit (0-20% self-acceptance on held-out genuine); switched default to `linear` kernel, which does better (50-60%) but is still noisy at this sample size. Did **not** grid-search nu/gamma against this same tiny holdout — that would just fit noise. Real validation is Stage 7 once synthetic clips exist as the actual anomaly class; if scores there are poor, revisit with more enrollment clips before touching hyperparameters again.

**Stage 5c — Detector fix + cross-speaker impostor eval (no clones needed)** — owner: Claude session (Mary) — [in progress 2026-10-05]
- [x] Diagnosed linear OC-SVM: on standardized data its boundary passes through the data centre (rho ~1e-8, |w| ~0, decisions within ±0.001 on its own training clips) — that, not sample size, caused ~50% self-acceptance and ≈0 confidences. Replaced as default by `src/models/distance.py` (RMS z-score from the enrollment fingerprint, threshold = 90th pct of leave-one-out genuine distances, std floors at measurement resolution). `cli.py` now uses it; `oneclass.py` kept for comparison only.
- [x] `src/impostor_eval.py`: cross-speaker impostor eval (LOO genuine vs other speaker re-referenced to the claimed speaker's F0) for prosodic / prosodic-minus-f0_mean (clone proxy) / MFCC × detector, plus feature-group ablation.
- [x] Data fix: 4 round-1 clips were misfiled between krishiv/mary (pitch + MFCC + jitter/shimmer all agreed; confirmed by listening) and moved.
- [x] **Round 2 recordings (2026-10-05)**: 30 matched-text clips per speaker per `docs/RECORDING_SCRIPT.md` (round-1 Munpalle clips removed from `data/genuine/`). Both speakers enrolled on s01–s20; s21–s30 held out.
- **Results on round 2**: held-out s21–s30 vs both models → 38/40 correct, ROC-AUC 1.00 (2 krishiv genuine false-rejects, s21/s30, high-shimmer — likely session drift). LOO impostor eval: full prosodic AUC 0.97, MFCC 0.96, but **prosodic without pitch level only 0.61** — on matched read text the two speakers separate mainly by pitch register/timbre; the "habit" features individually separate weakly (best f0_std 0.76). This is the clone-relevant number and it's weak; round-1's stronger pause/voice-quality separation looks partly like a room/text artifact (krishiv's recordings ~22 dB SNR vs mary's ~30). Real test is still clones of s21–s30.
- [ ] Update notebook, `docs/results.tex`, `DEMO_SCRIPT.md` with the above — next

**Stage 3b/3c — Prosody-led blended fingerprint + text-dependent check** — owner: Claude session (Mary) — [in progress 2026-10-05]
- [x] Per user: verdict = hand-weighted blend of component distances, prosody-led — prosody 0.60 / voice quality (jitter, shimmer) 0.15 / timbre (MFCC) 0.15 / pitch register (f0_mean) 0.10. `src/models/distance.py` (`DEFAULT_COMPONENTS`); `cli.py score` prints each component's distance and share. Per-feature |z| capped at 5; NaN features skipped per clip.
- [x] 8 new register-independent prosodic features in `src/features.py` (pitch range/slope/velocity/final movement, energy slope, voiced fraction, VarcoV, VarcoUV). Best single prosodic separators now f0_velocity (0.86) and voiced_fraction (0.85). Earlier worry that krishiv's noisier recordings inflate pYIN voicing doesn't hold for round 2: same phone + room, SNR krishiv 21.6 dB vs mary 23.5 dB (ranges overlap) — the ~30 dB figure was mary's round-1 clips.
- [x] `impostor_eval.py`: held-out protocol (enroll s01–s20 → test s21–s30, same sentences for genuine and impostor) + LOO; component-share and prosody-group ablation tables.
- **Held-out results**: blend AUC 0.97 (90% acc), prosody-only 0.72 (was 0.61 with old features), register+timbre 1.00. **But pitch register still drives ~69% of the genuine-vs-impostor gap in the blend despite its 0.10 weight** (its distance jumps 0.8 → 5 sd for an impostor; prosody only 1.03 → 1.25). Weight caps influence, not actual influence. Open decision for the user on how to make prosody primary in effect, not just in weight.
- [x] **Per user: cap influence** — non-prosody components capped at 2 sd (`NON_PROSODY_CAP`). Held-out: blend AUC 0.97 → **0.895**, accuracy 0.775 (impostor reject 60%); share of genuine-vs-impostor gap now prosody 41% / register 37% / timbre 16% / voice quality 6% (was register 69%). The honest cost of letting prosody lead: e.g. mary_s25 is now *accepted* by krishiv's model (1.259 vs threshold 1.282).
- [x] **Per user: text-dependent mode** — `src/contour.py`: DTW-aligned pitch (st vs clip median) + loudness (dB vs clip median) contours; `contour_dist` (shape) + `timing_dist` (local tempo variation, global speed factored out). Sanity: self 0, +3 st shift / 10% stretch ≈0.25 (processing artifacts), mary vs krishiv same sentence ≈0.96 each (≈4×). `cli.py score <spk> <clip> --sentence N`. **Uncalibrated until 2nd takes of s21–s30 exist** (`<speaker>_sNN_t2.m4a`, instructions appended to `docs/RECORDING_SCRIPT.md`); then `python src/contour.py` gives ROC-AUC/EER and the CLI gives a verdict.
- [ ] Next: once 2nd takes land — text-dependent eval; decide whether to fold `td_distance` into the blend as a prosody sub-component.
- [ ] Update notebook / `docs/results.tex` / `DEMO_SCRIPT.md` / `demo.sh` (demo.sh enrolls `*.m4a`, which now includes held-out s21–s30)

**Stage 1c — Local voice clones (XTTS-v2), no API key** — owner: Claude session (Mary) — [done 2026-10-06]
- [x] `.venv-tts` pinned in `requirements-tts.txt` (torch 2.8 + torchaudio 2.8 — torch ≥2.9 needs torchcodec; transformers <5 — 5.x breaks XTTS). Coqui's downloader stalled at 794 MB on a flaky connection; weights now come from a resumable HF download (`coqui/XTTS-v2` → `~/Library/Application Support/tts/xtts_v2_hf`, used automatically by `src/generate_clones.py`). ~10 s/sentence on M3 CPU.
- [x] 30 clones (s21–s30 × krishiv/mary/raghav, reference audio = s01–s20 only) in `data/synthetic/<speaker>/xtts/`. `raghav_s26.wav` is a 21 s XTTS babble failure → auto-skipped (>15 s). XTTS clones sit ~3 st *below* each speaker's genuine pitch (krishiv 110 vs 134 Hz, mary 193 vs 225, raghav 109 vs 127).
- [x] `cli.py enroll` fix: upserts enrolled clips only; speaker's other rows kept and f0_mean re-referenced to the new reference.
- **Clone results (held-out genuine s21–s30 vs clones of the same sentences)**: blend (capped) AUC **0.962**, EER 8.4%, 97% clones rejected / 83% genuine accepted; prosody-only 0.874 (0.99 krishiv, 0.99 mary, **0.51 raghav**); register+timbre 0.977.
- **What gives XTTS away** (per-feature, speaker-normalized): shimmer (clones too smooth, AUC 1.00), energy_slope (clones fade out −3.3 dB/s vs +0.56, 0.97), energy_std (0.92), f0_mean (−2.7 st, 0.97), pause durations (clones pause longer, 0.71 — why raghav, a heavy pauser, isn't caught). Pitch movement and rhythm (f0_velocity, f0_range, nPVI, VarcoV…) are reproduced ~perfectly (AUC ≈ 0.5). So the prosodic signal vs XTTS is loudness dynamics + pauses, not intonation/rhythm habits — and may be XTTS-specific.
- **Text-dependent (contour.py)**: clones are *farther* from the real speaker's reading of the same sentence than other humans are (td_distance 2.02 vs 1.74; contour 1.15 vs 0.86); krishiv/mary clearly, raghav level. Needs 2nd takes to know the genuine same-sentence distance.
- 3 speakers, genuine impostor eval (held-out): blend capped 0.932, prosody-only 0.816; prosody now drives 63% of the impostor gap; pauses are the most useful prosody group (raghav).

**Stage 1d — Second cloner (F5-TTS) + accent check + naturalness check** — owner: Claude session (Mary) — [done 2026-10-06]
- [x] `.venv-f5` (`requirements-f5.txt`; own venv — f5-tts pulls transformers 5.x, which breaks XTTS). `src/generate_clones_f5.py`: reference = speaker's s01(+s02) joined up to 11.8 s + exact transcript; 30 clones in `data/synthetic/<speaker>/f5tts/`, ~2 min/clip on M3 (MPS). Shared protocol in `src/clone_common.py`.
- [x] Accent check `src/accent_check.py` (SpeechBrain CommonAccent ECAPA, in `.venv-tts`; outputs are cosine scores, not probabilities): genuine krishiv/raghav Indian 30/30, mary England 20/30. **XTTS Americanizes everyone** (US 8/10, 8/10, 7/9). **F5 keeps accents**: krishiv Indian 10/10, raghav Indian 9/10, mary England 10/10 — per user requirement.
- **Clip-level clone results (blend, capped)**: F5 AUC 0.954, 93% rejected / 83% genuine accepted; XTTS 0.962, 97%. Prosody-only: F5 0.846 (krishiv 0.85, mary 0.96, raghav 0.84 — raghav's F5 clones are caught, unlike XTTS 0.51).
- **The cloners fail differently.** F5 tells are *intonation*: pitch moves too slowly (f0_velocity 14.2 vs 20.9 st/s, AUC 0.99), no declination (f0_slope +0.91 vs −1.30, 0.97), wider range (10.0 vs 7.1), almost no pauses (0.14 vs 1.6/clip); too smooth like XTTS (shimmer 0.98). XTTS copied intonation (≈0.51) but failed loudness (fade-out) and pauses. Shimmer is the one tell common to both.
- [x] **No 2nd takes (user decision)** → text-dependent check recast as a **naturalness check** (`src/contour.py`): candidate claiming X vs the *other* enrolled speakers' readings of the same sentence; per-speaker threshold = 90th pct on enrollment sentences s01–s20 (`data/features/naturalness_thresholds.json`); `cli.py score ... --sentence N`. Held-out: **F5 AUC 0.964, EER 3.3%, 93% rejected / 87% genuine accepted**; XTTS only 0.657 (50% rejected). Mean distance genuine ~1.7, XTTS ~2.05, F5 3.0–3.4. Checks "read like a person", not "is speaker X"; only 2 cohort readers/sentence → noisy. Complements the blend (XTTS caught by the blend, F5 by both).
- Report + deck drafts (claude.ai Docs / Slides artifacts) updated with these numbers.

**Stage 10 — Paper + VoiceGuard Lab front end** — owner: Claude session (Mary) — [in progress 2026-10-06]
- [ ] `docs/paper/` — LaTeX paper in the team's Woxsen report format (title page, contents, abstract, sections, figures, references), compiled to PDF; figures from `src/make_figures.py`
- [ ] `web/` — "VoiceGuard Lab": interactive pipeline visualizer over the held-out clips (VAD, pitch/loudness contours, features vs fingerprint, component distances, verdict, naturalness overlay). Data from `src/export_viz.py` (derived contours/features only — no audio). Also published as a private claude.ai artifact.

**Stage 1b — Third speaker** — owner: team — [recorded 2026-10-06]
- [x] `raghav` recorded 2026-10-06: 30 sentences (files arrived as "New Recording N"; renamed to `raghav_sNN` after checking the order — timestamps strictly sequential, and per-sentence durations correlate 0.64 with mary's vs ~0 when shifted by one). Valid audio, no clipping, SNR ~21 dB (same as others), quieter recording level.
- Notable: raghav's pitch register (median ~125 Hz) ≈ krishiv's (~132 Hz) — register can't separate them, a natural prosody-only test. He pauses far more: 4.7 internal pauses/clip vs ~0.5, incl. 47 pauses ≥200 ms (krishiv 3, mary 5) — real phrasing pauses, not a VAD artifact.
- [ ] 2nd takes of s21–s30 (all three speakers) still not recorded.

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

## Presentation materials (added 2026-08-25)
- `docs/results.tex` — hand-authored LaTeX version of `notebooks/results.ipynb`'s actual output (real tables/figures/printed numbers pulled from the executed notebook, not fabricated). Source only, not compiled — no LaTeX distribution installed in this environment; `pandoc`-based `nbconvert` LaTeX export was tried first and abandoned after Homebrew/network failures, hand-authoring was cleaner anyway. Depends on `docs/assets/feature-boxplots.png` (extracted from the notebook's own boxplot cell output — filename avoids underscores deliberately, they need escaping in raw LaTeX arguments like `\includegraphics`).
- `demo.sh` + `docs/DEMO_SCRIPT.md` — live presentation demo. Runs the real `cli.py enroll` pipeline + a live self-consistency check against actual recordings (not a mockup). **Deliberately does not run `cli.py score` live**: a full sweep of every same-speaker/cross-speaker clip combination showed confidence scores all within ±0.005 of zero — the current model doesn't reliably discriminate at this sample size, so a live single-clip verdict would just be showing noise, and picking a clip that happens to "work" would misrepresent the system to the panel. The demo instead honestly shows the self-consistency numbers already documented (krishiv 60%, mary 50%) and pivots the "does this work" evidence to the fingerprint visuals, which show real measurable spectral/prosodic differences independent of the SVM's current thin margin.

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
