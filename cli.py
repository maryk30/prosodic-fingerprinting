#!/usr/bin/env python3
"""VoiceGuard CLI — Stage 8 deliverable.

    python cli.py enroll <speaker> <clip1> [<clip2> ...]
    python cli.py score <speaker> <candidate_clip>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "models"))

import pandas as pd  # noqa: E402

from features import FEATURE_COLUMNS, extract_raw_features, speaker_f0_reference_hz, to_row  # noqa: E402
from fingerprint import FINGERPRINT_DIR, build_fingerprint, save_fingerprint  # noqa: E402
from oneclass import SpeakerAnomalyDetector  # noqa: E402
from spectral_features import MFCC_FEATURE_COLUMNS, SPECTRAL_COLUMNS, extract_mfcc_row  # noqa: E402

FEATURES_CSV = "data/features/features.csv"
SPECTRAL_CSV = "data/features/spectral_features.csv"


def cmd_enroll(args: argparse.Namespace) -> None:
    raw_clips = [
        extract_raw_features(path, speaker=args.speaker, label="genuine", tts_system="na")
        for path in args.clips
    ]
    f0_ref_hz = speaker_f0_reference_hz(raw_clips)
    rows = [to_row(r, f0_ref_hz) for r in raw_clips]
    df_new = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    if os.path.exists(FEATURES_CSV):
        df_existing = pd.read_csv(FEATURES_CSV)
        keep = ~((df_existing["speaker"] == args.speaker) & (df_existing["label"] == "genuine"))
        df_all = pd.concat([df_existing[keep], df_new], ignore_index=True)
    else:
        df_all = df_new
    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
    df_all.to_csv(FEATURES_CSV, index=False)

    fingerprint = build_fingerprint(df_new)
    fp_path = save_fingerprint(args.speaker, fingerprint)

    spectral_rows = [
        extract_mfcc_row(path, speaker=args.speaker, label="genuine", tts_system="na")
        for path in args.clips
    ]
    df_spectral_new = pd.DataFrame(spectral_rows, columns=SPECTRAL_COLUMNS)
    if os.path.exists(SPECTRAL_CSV):
        df_spectral_existing = pd.read_csv(SPECTRAL_CSV)
        keep = ~(
            (df_spectral_existing["speaker"] == args.speaker)
            & (df_spectral_existing["label"] == "genuine")
        )
        df_spectral_all = pd.concat([df_spectral_existing[keep], df_spectral_new], ignore_index=True)
    else:
        df_spectral_all = df_spectral_new
    df_spectral_all.to_csv(SPECTRAL_CSV, index=False)

    spectral_fingerprint = build_fingerprint(df_spectral_new, MFCC_FEATURE_COLUMNS)
    spectral_fp_path = save_fingerprint(
        args.speaker, spectral_fingerprint, out_dir=FINGERPRINT_DIR, suffix="_spectral"
    )

    detector = SpeakerAnomalyDetector.train(args.speaker, df_new, f0_ref_hz)
    model_path = detector.save()

    print(f"Enrolled '{args.speaker}' from {len(args.clips)} clip(s).")
    print(f"  prosodic fingerprint -> {fp_path}")
    print(f"  spectral fingerprint -> {spectral_fp_path}")
    print(f"  model                -> {model_path}")


def cmd_score(args: argparse.Namespace) -> None:
    detector = SpeakerAnomalyDetector.load(args.speaker)
    result = detector.score_clip(args.clip)
    print(f"{args.clip}: {result['prediction']} (confidence={result['confidence']:+.3f})")


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
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
