# VoiceGuard — Prosodic Fingerprinting MVP

Detects AI voice clones by checking whether a candidate clip matches an
enrolled speaker's *prosodic habits* (rhythm, pause structure, speaking rate,
pitch/energy dynamics) rather than relying on spectral/timbre artifacts.
Full proposal context lives in the project report; this repo is the one-day
MVP build.

See [CLAUDE.md](CLAUDE.md) for the live task board and current status.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (target CLI, once Stage 8 is done)
```bash
python cli.py enroll <speaker_name> data/genuine/<speaker_name>/*.wav
python cli.py score <speaker_name> <candidate_clip.wav>
```
