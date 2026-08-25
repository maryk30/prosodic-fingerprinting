"""Spectral fingerprint features — MFCC statistical summary per clip.

Companion to the prosodic pipeline (features.py / fingerprint.py): where
prosody captures rhythm/pitch/pause *habits*, this captures the raw
spectral/timbral texture of a speaker's voice via MFCCs — the standard
representation behind most speaker-ID and spectral anti-spoofing systems
(the "spectral baseline" CLAUDE.md calls out as Stage 6, scoped here as an
MFCC summary rather than a full pretrained ECAPA-TDNN embedding, to match
the same lightweight/no-heavy-dependency approach as the prosodic side).

Schema mirrors features.py's features.csv:
    speaker, clip_id, label, tts_system, mfcc1_mean..mfcc13_mean, mfcc1_std..mfcc13_std
"""

import glob
import os

import librosa
import numpy as np
import pandas as pd

from preprocessing import discover_clips, preprocess

N_MFCC = 13
MFCC_FEATURE_COLUMNS = [f"mfcc{i+1}_mean" for i in range(N_MFCC)] + [
    f"mfcc{i+1}_std" for i in range(N_MFCC)
]
SPECTRAL_COLUMNS = ["speaker", "clip_id", "label", "tts_system"] + MFCC_FEATURE_COLUMNS


def extract_mfcc_row(path: str, speaker: str, label: str, tts_system: str) -> dict:
    """MFCCs computed on the same preprocessed (loudness-normalized, 16kHz
    mono) audio as the prosodic pipeline, so both fingerprints describe the
    same signal."""
    clip = preprocess(path)
    mfcc = librosa.feature.mfcc(y=clip.audio, sr=clip.sr, n_mfcc=N_MFCC)  # (n_mfcc, frames)
    means = mfcc.mean(axis=1)
    stds = mfcc.std(axis=1)

    row = {
        "speaker": speaker,
        "clip_id": os.path.splitext(os.path.basename(path))[0],
        "label": label,
        "tts_system": tts_system,
    }
    for i in range(N_MFCC):
        row[f"mfcc{i+1}_mean"] = float(means[i])
        row[f"mfcc{i+1}_std"] = float(stds[i])
    return row


def build_spectral_table(genuine_root: str, synthetic_root: str | None = None) -> list[dict]:
    rows = []

    for speaker_dir in sorted(glob.glob(f"{genuine_root}/*/")):
        speaker = os.path.basename(speaker_dir.rstrip("/"))
        for clip_path in discover_clips(speaker_dir):
            rows.append(extract_mfcc_row(clip_path, speaker, label="genuine", tts_system="na"))

    if synthetic_root and os.path.isdir(synthetic_root):
        for speaker_dir in sorted(glob.glob(f"{synthetic_root}/*/")):
            speaker = os.path.basename(speaker_dir.rstrip("/"))
            for tts_dir in sorted(glob.glob(f"{speaker_dir}*/")):
                tts_system = os.path.basename(tts_dir.rstrip("/"))
                for clip_path in discover_clips(tts_dir):
                    rows.append(
                        extract_mfcc_row(clip_path, speaker, label="synthetic", tts_system=tts_system)
                    )

    return rows


if __name__ == "__main__":
    rows = build_spectral_table("data/genuine", "data/synthetic")
    df = pd.DataFrame(rows, columns=SPECTRAL_COLUMNS)
    out_path = "data/features/spectral_features.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")
    print(df)
