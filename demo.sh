#!/usr/bin/env bash
# VoiceGuard — live presentation demo.
#
# Runs the real pipeline (not a mockup) against the actual krishiv/mary
# recordings: enrollment, fingerprint construction, and a live
# self-consistency check. Deliberately does NOT run `cli.py score` live —
# see docs/DEMO_SCRIPT.md for why (confidence margin is currently near
# zero with only 10 clips/speaker; a single live verdict could look
# stronger or weaker than reality depending on which clip is picked).
#
# Usage: ./demo.sh   (run from the repo root, with .venv activated)

set -e

pause() {
    echo
    read -rp "   [press Enter to continue] " _
    echo
}

banner() {
    echo
    echo "════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════"
}

banner "VoiceGuard — Live Demo"
echo "Prosodic + spectral voice fingerprinting, running live on real"
echo "recordings of two enrolled speakers: krishiv and mary."
pause

banner "Step 1 — Enroll krishiv"
echo "\$ python cli.py enroll krishiv data/genuine/krishiv/*.m4a"
echo
python cli.py enroll krishiv data/genuine/krishiv/*.m4a
pause

banner "Step 2 — Enroll mary"
echo "\$ python cli.py enroll mary data/genuine/mary/*.m4a"
echo
python cli.py enroll mary data/genuine/mary/*.m4a
pause

banner "Step 3 — What just got built"
echo "Each enroll call just produced, from real audio, in real time:"
echo "  - a prosodic fingerprint  (33-dim: mean/var/skew of 11 features)"
echo "  - a spectral fingerprint  (78-dim: mean/var/skew of 13 MFCCs)"
echo "  - a trained One-Class SVM model for that speaker"
echo
ls -la data/features/fingerprints/ data/features/models/
pause

banner "Step 4 — Self-consistency check"
echo "How well does each speaker's model recognize their OWN held-out"
echo "voice? (leave-one-out: train on 9 clips, test the 10th)"
echo
echo "\$ python src/models/oneclass.py"
echo
python src/models/oneclass.py
pause

banner "Step 5 — Voice fingerprint visuals"
echo "Switch to the browser tab / published artifact now:"
echo "  -> spectrograms + MFCC profiles for both speakers"
echo "  -> prosodic comparison (jitter/shimmer clearly separate the two)"
echo
echo "(see docs/DEMO_SCRIPT.md for the narration cue card)"
pause

banner "Demo complete."
