#!/usr/bin/env python3
"""VoiceGuard CLI — Stage 8 deliverable.

    python cli.py enroll <speaker> <clip1> [<clip2> ...]
    python cli.py score <speaker> <candidate_clip> [--sentence N]

--sentence N: the candidate is reading prompt sentence N from
docs/RECORDING_SCRIPT.md, so also compare its pitch/loudness contour and
timing against the speaker's own reading of that sentence (text-dependent
check, src/contour.py).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "models"))

import pandas as pd  # noqa: E402

from features import FEATURE_COLUMNS, extract_raw_features, speaker_f0_reference_hz, to_row  # noqa: E402
from fingerprint import FINGERPRINT_DIR, build_fingerprint, save_fingerprint  # noqa: E402
from contour import compare, contour, find_take, td_distance, td_threshold  # noqa: E402
from distance import FingerprintDistanceDetector  # noqa: E402
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
        ref = find_take(args.speaker, args.sentence)
        if ref is None:
            print(f"text-dependent: no enrolled reading of sentence {args.sentence} by {args.speaker}")
            return
        if os.path.abspath(ref) == os.path.abspath(args.clip):
            print("text-dependent: candidate is the reference take itself — skipped")
            return
        m = compare(contour(ref), contour(args.clip))
        d = td_distance(m)
        # threshold from the speaker's genuine take pairs of *other* sentences
        threshold = td_threshold(args.speaker, exclude_sentence=args.sentence)
        verdict = (
            "uncalibrated (needs 2nd takes, see docs/RECORDING_SCRIPT.md)"
            if threshold is None
            else ("genuine" if d <= threshold else "synthetic") + f" (threshold {threshold:.3f})"
        )
        print(
            f"text-dependent vs {os.path.basename(ref)}: distance {d:.3f} -> {verdict}\n"
            f"  contour {m['contour_dist']:.3f}  timing {m['timing_dist']:.3f}  "
            f"tempo {m['tempo_ratio']:.3f}"
        )


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
