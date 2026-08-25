"""Spectral fingerprint construction — the MFCC-side counterpart to
fingerprint.py's prosodic fingerprint. Same mean/var/skew-summary approach,
applied to each speaker's MFCC statistics instead of prosodic features.
"""

import pandas as pd

from fingerprint import FINGERPRINT_DIR, build_fingerprint, save_fingerprint
from spectral_features import MFCC_FEATURE_COLUMNS

SUFFIX = "_spectral"


def build_all_spectral_fingerprints(spectral_csv: str) -> pd.DataFrame:
    """One spectral fingerprint row per speaker, built from their genuine clips only."""
    df = pd.read_csv(spectral_csv)
    genuine = df[df["label"] == "genuine"]
    rows = []
    for speaker, group in genuine.groupby("speaker"):
        fp = build_fingerprint(group, MFCC_FEATURE_COLUMNS)
        fp["speaker"] = speaker
        fp["n_clips"] = len(group)
        rows.append(fp)
    return pd.DataFrame(rows).set_index("speaker")


if __name__ == "__main__":
    table = build_all_spectral_fingerprints("data/features/spectral_features.csv")
    for speaker in table.index:
        fp = table.loc[speaker].drop("n_clips").to_dict()
        n_clips = int(table.loc[speaker, "n_clips"])
        path = save_fingerprint(speaker, fp, out_dir=FINGERPRINT_DIR, suffix=SUFFIX)
        print(f"saved {speaker} spectral fingerprint ({n_clips} clips) -> {path}")
