"""Stage 1d: accent check for genuine recordings vs voice clones.

Neither XTTS-v2 nor F5-TTS has an accent control — a clone's accent comes
from the speaker's reference audio. This verifies, per speaker and per
source (genuine / each TTS system), which English accent a pretrained
classifier hears: SpeechBrain's CommonAccent ECAPA model (16 accents incl.
indian, england, us), trained on Common Voice.

Runs in .venv-tts (needs torch + speechbrain):

    .venv-tts/bin/python src/accent_check.py

Reads data/genuine/<speaker>/ and data/synthetic/<speaker>/<tts_system>/,
writes per-clip results to data/features/accent.csv.
"""

import glob
import os

import librosa
import pandas as pd
import torch
from speechbrain.inference.classifiers import EncoderClassifier

MODEL = "Jzuluaga/accent-id-commonaccent_ecapa"
SAVEDIR = os.path.expanduser("~/Library/Application Support/speechbrain/accent-id-commonaccent_ecapa")
OUT_CSV = "data/features/accent.csv"
MAX_CLIP_DURATION_S = 15.0  # same skip rule as preprocessing.discover_clips


def clips() -> list[tuple[str, str, str]]:
    """(speaker, source, path); source = 'genuine' or the tts_system dir name."""
    out = []
    for p in sorted(glob.glob("data/genuine/*/*")):
        out.append((p.split(os.sep)[-2], "genuine", p))
    for p in sorted(glob.glob("data/synthetic/*/*/*")):
        parts = p.split(os.sep)
        out.append((parts[-3], parts[-2], p))
    return [c for c in out if c[2].endswith((".m4a", ".wav", ".mp3"))]


def main() -> None:
    clf = EncoderClassifier.from_hparams(source=MODEL, savedir=SAVEDIR, run_opts={"device": "cpu"})
    labels = clf.hparams.label_encoder.decode_ndim(list(range(len(clf.hparams.label_encoder))))

    rows = []
    for speaker, source, path in clips():
        audio, _ = librosa.load(path, sr=16000, mono=True)
        if len(audio) / 16000 > MAX_CLIP_DURATION_S:
            continue
        with torch.no_grad():
            # this model's head is a cosine (AAM-softmax) classifier: outputs
            # are per-accent similarity scores in [-1, 1], not log-probabilities
            scores = clf.classify_batch(torch.tensor(audio).unsqueeze(0))[0].squeeze(0)
        ranked = scores.argsort(descending=True)
        top, second = int(ranked[0]), int(ranked[1])
        rows.append(
            {
                "speaker": speaker,
                "source": source,
                "clip_id": os.path.splitext(os.path.basename(path))[0],
                "accent": labels[top],
                "margin": float(scores[top] - scores[second]),  # over the runner-up accent
                **{f"s_{lab}": float(scores[i]) for i, lab in enumerate(labels)},
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 160)
    print("Top accent per clip, counts by speaker / source:")
    print(df.groupby(["speaker", "source"])["accent"].value_counts().unstack(fill_value=0).to_string())
    print("\nMean similarity score for the main candidate accents, and mean top-accent margin:")
    cols = [c for c in ["s_indian", "s_england", "s_us"] if c in df] + ["margin"]
    print(df.groupby(["speaker", "source"])[cols].mean().round(3).to_string())
    print(f"\nper-clip results -> {OUT_CSV}")


if __name__ == "__main__":
    main()
