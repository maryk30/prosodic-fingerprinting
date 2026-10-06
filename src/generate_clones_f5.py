"""Stage 1d: second cloner — F5-TTS (flow-matching zero-shot TTS), local.

Runs in its own environment (F5's dependencies conflict with the XTTS pins):

    python3.11 -m venv .venv-f5 && .venv-f5/bin/pip install -r requirements-f5.txt
    .venv-f5/bin/python src/generate_clones_f5.py krishiv mary raghav

Same protocol as generate_clones.py (XTTS): text = sentences 21–30, voice
reference = the speaker's own enrollment audio only (s01–s20). F5 takes one
reference clip (≤ ~12 s) *with its transcript*; we know the exact text, so
the reference is the speaker's first enrollment sentences joined (with a
short gap) up to REF_MAX_S, and ref_text is those sentences — no ASR guess.

Accent: F5 has no accent control; it copies the reference audio. Using the
speaker's own recordings is what keeps krishiv/raghav Indian-accented and
mary English-accented; src/accent_check.py verifies the result.

Output: data/synthetic/<speaker>/f5tts/<speaker>_sNN.wav
"""

import argparse
import os
import re
import sys
import tempfile

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from clone_common import CLONE_SENTENCES, ENROLL_SENTENCES, enrollment_clips, sentences  # noqa: E402

TTS_SYSTEM = "f5tts"
REF_SR = 24000
REF_MAX_S = 11.8  # F5 clips references at 12 s (would cut ref_text mid-sentence); stay under it
REF_GAP_S = 0.3
SEED = 0


def build_reference(speaker: str, text: dict[int, str], out_path: str) -> str:
    """Join the speaker's s01, s02, ... until REF_MAX_S; return ref_text."""
    clips = enrollment_clips(speaker)
    by_n = {int(re.search(r"_s0*(\d+)$", os.path.splitext(os.path.basename(p))[0]).group(1)): p for p in clips}
    gap = np.zeros(int(REF_GAP_S * REF_SR), dtype=np.float32)
    audio, words = [], []
    for n in ENROLL_SENTENCES:
        y, _ = librosa.load(by_n[n], sr=REF_SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=35)
        total = sum(len(a) for a in audio) / REF_SR
        if audio and total + REF_GAP_S + len(y) / REF_SR > REF_MAX_S:
            break
        audio += ([gap] if audio else []) + [y]
        words.append(text[n])
    sf.write(out_path, np.concatenate(audio), REF_SR)
    return " ".join(words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("speakers", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None, help="cpu / mps (default: F5 picks)")
    args = parser.parse_args()

    from f5_tts.api import F5TTS

    text = sentences()
    tts = F5TTS(device=args.device)

    for speaker in args.speakers:
        out_dir = f"data/synthetic/{speaker}/{TTS_SYSTEM}"
        os.makedirs(out_dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            ref_wav = os.path.join(tmp, f"{speaker}_ref.wav")
            ref_text = build_reference(speaker, text, ref_wav)
            print(f"{speaker}: reference {sf.info(ref_wav).duration:.1f}s — {ref_text!r}")
            for n in CLONE_SENTENCES:
                out = os.path.join(out_dir, f"{speaker}_s{n:02d}.wav")
                if os.path.exists(out) and not args.overwrite:
                    print(f"exists, skipping: {out}")
                    continue
                tts.infer(
                    ref_file=ref_wav, ref_text=ref_text, gen_text=text[n],
                    file_wave=out, seed=SEED, show_info=lambda *a, **k: None,
                )
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
