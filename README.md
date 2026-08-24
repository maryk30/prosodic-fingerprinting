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

## Usage
```bash
python cli.py enroll <speaker_name> data/genuine/<speaker_name>/*.m4a
python cli.py score <speaker_name> <candidate_clip.m4a>
```
Any format librosa/ffmpeg can decode works (m4a, wav, mp3, ...) — make sure
`ffmpeg` is installed (`brew install ffmpeg`).
