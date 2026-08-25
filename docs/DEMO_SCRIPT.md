# VoiceGuard — Live Demo Script

Run `./demo.sh` from the repo root (with `.venv` activated) during the
presentation. It runs the real pipeline against the actual `krishiv`/`mary`
recordings — nothing here is a mockup or pre-recorded output.

Each step below pauses for you to press Enter, so you control the pacing.
Talking points are matched to each step.

## Before you go up

- `cd` into the repo, activate `.venv`, have a terminal font size the back
  row can read.
- Have the [Voice Fingerprints](https://claude.ai/code/artifact/579bf1cb-8907-4efd-9934-aabc7158826f)
  page open in a browser tab, ready to switch to at Step 5.
- Test `./demo.sh` once beforehand so the first run's cache/font-load
  hiccups don't happen live.

## Step 1–2 — Enroll both speakers

**What happens:** `cli.py enroll <speaker> <clips>` runs the full
pipeline — preprocessing, prosodic + spectral feature extraction,
fingerprint construction, model training — on real audio, live.

**Say:**
> "This isn't a pre-recorded result — I'm enrolling both speakers right
> now, from their actual recordings. In a couple seconds each, the system
> builds two fingerprints per person: one prosodic, one spectral."

## Step 3 — What just got built

**What happens:** lists the fingerprint JSON files and trained model
files that just got created.

**Say:**
> "Two fingerprints per speaker: a 33-dimensional prosodic one — rhythm,
> pitch, pauses, jitter, shimmer — and a 78-dimensional spectral one, from
> MFCCs, the same representation most speaker-ID systems use. Plus a
> trained anomaly-detection model per speaker."

## Step 4 — Self-consistency check

**What happens:** runs a leave-one-out test — for each speaker, hold out
one genuine clip, train on the other 9, check if the model still accepts
it as genuine. Prints real, reproducible numbers (currently ~50-60%).

**Say — be upfront about this, don't undersell it:**
> "This is a self-consistency check, not a detection-accuracy number —
> we don't have cloned voice samples yet, so we can't measure the actual
> genuine-vs-clone task live. What this shows is that with only 10
> enrollment clips per speaker, the model's confidence margin is still
> thin — around half the time it doesn't even recognize the speaker's own
> held-out voice confidently. That's an honest, expected consequence of a
> small sample size in an 11-dimensional feature space, not a bug — we
> found the same pattern when comparing kernel choices, documented in our
> handover doc. More enrollment data is the fix, and the evaluation code
> for the real detection task is fully built and tested — it's wired up
> and ready, just waiting on synthetic clone data to score against."

**Anticipated question: "So does it actually detect clones?"**
> "Not proven yet — that's the honest answer. The detection *model* and
> the *evaluation pipeline* are both built and tested end-to-end. What's
> missing is the negative class: synthetic clones to score against. Right
> now every number you could show is genuine-vs-genuine. The moment we
> add cloned samples, real ROC-AUC and EER numbers come out with zero
> code changes — that cell is already written and tested against
> hand-built data."

## Step 5 — Voice fingerprint visuals

**What happens:** switch to the browser tab.

**Say:**
> "This is where the fingerprints become visible. These are real
> spectrograms and MFCC profiles from their actual recordings — you can
> see the two speakers occupy genuinely different regions of both the
> spectral and prosodic space. Jitter and shimmer — voice-quality
> measures — show the clearest separation of anything we've measured so
> far."

Point at:
- The two mel-spectrograms — visibly different energy/formant patterns.
- The MFCC coefficient profiles — same general shape (expected, MFCCs
  are dominated by the first couple of coefficients for any voice) but
  different magnitudes.
- The jitter/shimmer bars in the prosodic comparison — the largest gap
  between the two speakers of any feature measured.

## Closing line

> "The full pipeline — preprocessing, feature extraction, fingerprinting,
> detection, and evaluation — works end to end on real data today. What's
> honestly still missing is the negative class to prove the detection
> claim, which is a data-collection problem, not an engineering one."

## If something breaks live

- `demo.sh` re-running is safe — `enroll` overwrites that speaker's prior
  data cleanly, nothing accumulates or corrupts.
- If the terminal font is unreadable from the back, have
  `docs/results.tex` or the notebook open as a backup visual.
- If the browser/artifact tab fails to load, the same images are at
  `data/features/fingerprint_visuals/*.png` locally — open them directly.
