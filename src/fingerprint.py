"""Stage 4: fingerprint construction.

Collapses a speaker's enrollment clips (their rows in features.csv) into a
single statistical summary vector — mean/var/skew per prosodic feature —
per CLAUDE.md's MVP scope decision (summary vector, not a learned embedding).
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import skew

from features import FEATURE_COLUMNS

NUMERIC_FEATURES = [
    c for c in FEATURE_COLUMNS if c not in ("speaker", "clip_id", "label", "tts_system")
]
FINGERPRINT_DIR = "data/features/fingerprints"

# Stage 3b feature groups, used as the components of the blended detector
# (src/models/distance.py). PROSODIC = speaking *habit*: how pitch and
# loudness move, rhythm, rate, pauses — all independent of where the voice
# sits. The other groups describe the voice itself and get small weights.
# energy_mean is in no group: after RMS loudness normalization it mostly
# reflects the mic/room (speech-vs-noise fraction), not the speaker.
PROSODIC_FEATURES = [
    "f0_std", "f0_range", "f0_slope", "f0_velocity", "f0_final_move",  # pitch movement
    "energy_std", "energy_slope",  # loudness movement
    "speaking_rate_mean", "npvi", "voiced_fraction", "varco_v", "varco_uv",  # rate/rhythm
    "pause_count", "pause_mean_dur", "pause_var_dur",  # pauses
]
VOICE_QUALITY_FEATURES = ["jitter", "shimmer"]
REGISTER_FEATURES = ["f0_mean"]  # pitch level, semitones vs the claimed speaker's median
# Pre-Stage-3b 11-feature set, kept for before/after comparison.
LEGACY_FEATURES = NUMERIC_FEATURES[:11]


def build_fingerprint(clip_rows: pd.DataFrame, numeric_features: list[str] = NUMERIC_FEATURES) -> dict:
    """clip_rows: one speaker's enrollment clips, with `numeric_features` columns
    present. Reused by src/spectral_fingerprint.py with a different column set,
    since the mean/var/skew-summary approach is identical for both fingerprint
    types — only the underlying features (prosodic vs. spectral/MFCC) differ."""
    vec = {}
    for col in numeric_features:
        values = clip_rows[col].to_numpy(dtype=float)
        vec[f"{col}_mean"] = float(np.mean(values))
        vec[f"{col}_var"] = float(np.var(values))
        vec[f"{col}_skew"] = float(skew(values)) if len(values) >= 3 else float("nan")
    return vec


def build_all_fingerprints(features_csv: str) -> pd.DataFrame:
    """One fingerprint row per speaker, built from their genuine clips only."""
    df = pd.read_csv(features_csv)
    genuine = df[df["label"] == "genuine"]
    rows = []
    for speaker, group in genuine.groupby("speaker"):
        fp = build_fingerprint(group)
        fp["speaker"] = speaker
        fp["n_clips"] = len(group)
        rows.append(fp)
    return pd.DataFrame(rows).set_index("speaker")


def save_fingerprint(
    speaker: str, fingerprint: dict, out_dir: str = FINGERPRINT_DIR, suffix: str = ""
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{speaker}{suffix}.json")
    with open(path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    return path


def load_fingerprint(speaker: str, out_dir: str = FINGERPRINT_DIR, suffix: str = "") -> dict:
    path = os.path.join(out_dir, f"{speaker}{suffix}.json")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    table = build_all_fingerprints("data/features/features.csv")
    for speaker in table.index:
        fp = table.loc[speaker].drop("n_clips").to_dict()
        n_clips = int(table.loc[speaker, "n_clips"])
        path = save_fingerprint(speaker, fp)
        print(f"saved {speaker} fingerprint ({n_clips} clips) -> {path}")
