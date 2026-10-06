"""Stage 1c: generate voice clones locally with XTTS-v2 (Coqui), no API key.

Runs in the separate TTS environment (torch + coqui-tts are kept out of the
main .venv):

    python3.11 -m venv .venv-tts && .venv-tts/bin/pip install -r requirements-tts.txt
    .venv-tts/bin/python src/generate_clones.py krishiv mary

Protocol (docs/RECORDING_SCRIPT.md):
  - voice reference = the speaker's enrollment sentences s01–s20 only;
    the held-out sentences s21–s30 are never shown to the cloner
  - text = sentences 21–30, so every clone has a genuine held-out
    recording of the same sentence to be compared against
Output: data/synthetic/<speaker>/xtts/<speaker>_sNN.wav

XTTS-v2 weights are under the Coqui Public Model License (non-commercial
use only); setting COQUI_TOS_AGREED below accepts it for this coursework.
"""

import argparse
import os
import subprocess
import tempfile

os.environ.setdefault("COQUI_TOS_AGREED", "1")

import sys  # noqa: E402

import torch  # noqa: E402
from TTS.api import TTS  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from clone_common import CLONE_SENTENCES, ENROLL_SENTENCES, enrollment_clips, sentences  # noqa: E402

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
# Coqui's own downloader stalled mid-file on a flaky connection; a resumable
# Hugging Face download into this dir is preferred when present:
#   huggingface_hub.snapshot_download("coqui/XTTS-v2", local_dir=XTTS_DIR,
#       allow_patterns=["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"])
XTTS_DIR = os.path.expanduser(
    os.environ.get("XTTS_DIR", "~/Library/Application Support/tts/xtts_v2_hf")
)
TTS_SYSTEM = "xtts"


def to_wav(paths: list[str], out_dir: str) -> list[str]:
    """XTTS loads reference audio with torchaudio, which can't read m4a."""
    wavs = []
    for p in paths:
        w = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".wav")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p, "-ac", "1", "-ar", "22050", w], check=True)
        wavs.append(w)
    return wavs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("speakers", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    text = sentences()
    torch.manual_seed(0)
    if os.path.exists(os.path.join(XTTS_DIR, "model.pth")):
        tts = TTS(model_path=XTTS_DIR, config_path=os.path.join(XTTS_DIR, "config.json"))
    else:
        tts = TTS(MODEL)
    tts = tts.to("cpu")  # XTTS has ops unsupported on Apple MPS

    for speaker in args.speakers:
        refs = enrollment_clips(speaker)
        if len(refs) != len(ENROLL_SENTENCES):
            raise SystemExit(f"{speaker}: expected {len(ENROLL_SENTENCES)} enrollment clips, found {len(refs)}")
        out_dir = f"data/synthetic/{speaker}/{TTS_SYSTEM}"
        os.makedirs(out_dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            ref_wavs = to_wav(refs, tmp)
            for n in CLONE_SENTENCES:
                out = os.path.join(out_dir, f"{speaker}_s{n:02d}.wav")
                if os.path.exists(out) and not args.overwrite:
                    print(f"exists, skipping: {out}")
                    continue
                tts.tts_to_file(text=text[n], speaker_wav=ref_wavs, language="en", file_path=out)
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
