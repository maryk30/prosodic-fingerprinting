#!/usr/bin/env python3
"""VoiceGuard CLI — Stage 8 deliverable.

    python cli.py enroll <speaker> <clip1> [<clip2> ...]
    python cli.py score <speaker> <candidate_clip> [--sentence N]

--sentence N: the candidate is reading prompt sentence N from
docs/RECORDING_SCRIPT.md, so also run the naturalness check: compare its
pitch/loudness contour and timing with the other enrolled speakers'
readings of that sentence (src/contour.py; run `python src/contour.py` once
to calibrate the thresholds).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "models"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from features import FEATURE_COLUMNS, extract_raw_features, speaker_f0_reference_hz, to_row  # noqa: E402
from fingerprint import FINGERPRINT_DIR, build_fingerprint, save_fingerprint  # noqa: E402
from contour import ContourCache, enrolled_speakers, load_thresholds, naturalness  # noqa: E402
from distance import FingerprintDistanceDetector  # noqa: E402
from spectral_features import MFCC_FEATURE_COLUMNS, SPECTRAL_COLUMNS, extract_mfcc_row  # noqa: E402

FEATURES_CSV = "data/features/features.csv"
SPECTRAL_CSV = "data/features/spectral_features.csv"
F0_REF_JSON = "data/features/f0_reference.json"


def _upsert(csv_path: str, df_new: pd.DataFrame) -> pd.DataFrame:
    """Write df_new into csv_path, replacing only rows with the same
    (speaker, clip_id, label). The speaker's other clips — e.g. held-out
    test sentences, or synthetic clones — are kept."""
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        key = ["speaker", "clip_id", "label"]
        new_keys = set(map(tuple, df_new[key].to_numpy()))
        keep = [tuple(k) not in new_keys for k in df_existing[key].to_numpy()]
        df_all = pd.concat([df_existing[keep], df_new], ignore_index=True)
    else:
        df_all = df_new
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_all.to_csv(csv_path, index=False)
    return df_all


def _rereference_speaker(speaker: str, new_ref_hz: float) -> None:
    """Enrollment recomputes the speaker's median-F0 reference from the
    enrolled clips. Shift f0_mean on their *other* rows in features.csv from
    the old reference to the new one, so every row of that speaker is in the
    same semitone frame, and record the new reference."""
    refs = {}
    if os.path.exists(F0_REF_JSON):
        with open(F0_REF_JSON) as f:
            refs = json.load(f)
    old_ref_hz = refs.get(speaker)
    if old_ref_hz and os.path.exists(FEATURES_CSV) and old_ref_hz != new_ref_hz:
        df = pd.read_csv(FEATURES_CSV)
        rows = df["speaker"] == speaker
        df.loc[rows, "f0_mean"] += 12 * np.log2(old_ref_hz / new_ref_hz)
        df.to_csv(FEATURES_CSV, index=False)
    refs[speaker] = new_ref_hz
    with open(F0_REF_JSON, "w") as f:
        json.dump(refs, f, indent=2)


def cmd_enroll(args: argparse.Namespace) -> None:
    raw_clips = [
        extract_raw_features(path, speaker=args.speaker, label="genuine", tts_system="na")
        for path in args.clips
    ]
    f0_ref_hz = speaker_f0_reference_hz(raw_clips)
    rows = [to_row(r, f0_ref_hz) for r in raw_clips]
    df_new = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    _rereference_speaker(args.speaker, f0_ref_hz)  # before upsert: shifts only pre-existing rows
    _upsert(FEATURES_CSV, df_new)

    fingerprint = build_fingerprint(df_new)
    fp_path = save_fingerprint(args.speaker, fingerprint)

    spectral_rows = [
        extract_mfcc_row(path, speaker=args.speaker, label="genuine", tts_system="na")
        for path in args.clips
    ]
    df_spectral_new = pd.DataFrame(spectral_rows, columns=SPECTRAL_COLUMNS)
    _upsert(SPECTRAL_CSV, df_spectral_new)

    spectral_fingerprint = build_fingerprint(df_spectral_new, MFCC_FEATURE_COLUMNS)
    spectral_fp_path = save_fingerprint(
        args.speaker, spectral_fingerprint, out_dir=FINGERPRINT_DIR, suffix="_spectral"
    )

    enroll_rows = df_new.merge(
        df_spectral_new[["clip_id"] + MFCC_FEATURE_COLUMNS], on="clip_id", how="inner"
    )
    detector = FingerprintDistanceDetector.train(args.speaker, enroll_rows, f0_ref_hz)
    model_path = detector.save()

    print(f"Enrolled '{args.speaker}' from {len(args.clips)} clip(s).")
    print(f"  prosodic fingerprint -> {fp_path}")
    print(f"  spectral fingerprint -> {spectral_fp_path}")
    print(f"  model                -> {model_path} (accept threshold {detector.threshold:.3f})")


def cmd_score(args: argparse.Namespace) -> None:
    detector = FingerprintDistanceDetector.load(args.speaker)
    result = detector.score_clip(args.clip)
    print(
        f"{args.clip}: {result['prediction']} (confidence={result['confidence']:+.3f}; "
        f"distance {result['distance']:.3f} vs threshold {result['threshold']:.3f})"
    )
    for name, part in result["components"].items():
        weight = detector.components[name][1]
        print(
            f"  {name:15s} weight {weight:.2f}  distance {part['distance']:5.2f} sd  "
            f"-> {part['share']:4.0%} of score" + ("  (capped)" if part["capped"] else "")
        )

    if args.sentence is not None:
        try:
            threshold = load_thresholds()[args.speaker]
        except (FileNotFoundError, KeyError):
            print("naturalness: not calibrated for this speaker — run `python src/contour.py` first")
            return
        try:
            cache = ContourCache()
            d = naturalness(cache(args.clip), args.sentence, args.speaker, enrolled_speakers(), cache)
        except ValueError as e:
            print(f"naturalness: {e}")
            return
        verdict = "genuine" if d <= threshold else "synthetic"
        print(f"naturalness (sentence {args.sentence}): distance {d:.3f} vs threshold {threshold:.3f} -> {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="cli.py", description="VoiceGuard prosodic fingerprinting")
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll", help="Build and save a speaker's prosodic fingerprint")
    p_enroll.add_argument("speaker")
    p_enroll.add_argument("clips", nargs="+")
    p_enroll.set_defaults(func=cmd_enroll)

    p_score = sub.add_parser("score", help="Score a candidate clip against an enrolled speaker")
    p_score.add_argument("speaker")
    p_score.add_argument("clip")
    p_score.add_argument(
        "--sentence", type=int, help="prompt sentence number the clip is reading (text-dependent check)"
    )
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
