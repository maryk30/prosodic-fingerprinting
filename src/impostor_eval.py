"""Stage 5c/3b: cross-speaker impostor evaluation — real two-class numbers
without any synthetic clones.

Two protocols, per target speaker T:

  heldout (primary) — matches docs/RECORDING_SCRIPT.md:
    enroll on T's s01–s20; genuine trials = T's s21–s30; impostor trials =
    every other speaker's s21–s30 (the *same sentences*, so text can't
    explain a difference).
  loo — every clip of T scored by a model trained on T's other clips;
    impostor trials = every other speaker's clips vs T's full model.

Scores are pooled over targets, then ROC-AUC / EER via eval.py.

This is an *impostor* task (a different human claiming to be T), not the
clone task. A clone copies T's pitch register and timbre, so the
`prosody only` row is the closest proxy for how the detector would fare
against a good clone; `register + timbre` shows how much of the separation
comes from the things a clone copies.

F0 note: features.csv stores f0_mean in semitones relative to the clip
owner's own median pitch. A real impostor clip is measured against the
*claimed* speaker's reference (that's what score_clip() does), so impostor
rows are re-referenced to the target before scoring.

2 speakers x 10 held-out clips = 20 genuine + 20 impostor trials: read the
numbers as indicative, not tight estimates.
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from eval import evaluate  # noqa: E402
from fingerprint import LEGACY_FEATURES, PROSODIC_FEATURES, REGISTER_FEATURES  # noqa: E402
from spectral_features import MFCC_FEATURE_COLUMNS  # noqa: E402
from distance import DEFAULT_COMPONENTS, Components, FingerprintDistanceDetector  # noqa: E402

FEATURES_CSV = "data/features/features.csv"
SPECTRAL_CSV = "data/features/spectral_features.csv"
F0_REF_JSON = "data/features/f0_reference.json"
HELDOUT_FIRST_SENTENCE = 21

SYSTEMS: dict[str, Components] = {
    "blend (default)": DEFAULT_COMPONENTS,
    "prosody only": {"prosody": (PROSODIC_FEATURES, 1.0)},
    "legacy 11 features": {"legacy": (LEGACY_FEATURES, 1.0)},
    "register + timbre": {
        "pitch_register": (REGISTER_FEATURES, 0.5),
        "timbre": (MFCC_FEATURE_COLUMNS, 0.5),
    },
}

# Prosody sub-groups, dropped one at a time from `prosody only` for ablation.
PROSODY_GROUPS = {
    "pitch_movement": ["f0_std", "f0_range", "f0_slope", "f0_velocity", "f0_final_move"],
    "loudness_movement": ["energy_std", "energy_slope"],
    "rate_rhythm": ["speaking_rate_mean", "npvi", "voiced_fraction", "varco_v", "varco_uv"],
    "pauses": ["pause_count", "pause_mean_dur", "pause_var_dur"],
}


def load_table() -> tuple[pd.DataFrame, dict]:
    """Genuine prosodic + MFCC features joined per clip, with `sentence`
    parsed from `<speaker>_sNN` clip ids (NaN for other names)."""
    pros = pd.read_csv(FEATURES_CSV)
    spec = pd.read_csv(SPECTRAL_CSV)
    df = pros.merge(
        spec[["speaker", "clip_id", "label"] + MFCC_FEATURE_COLUMNS],
        on=["speaker", "clip_id", "label"],
        how="inner",
    )
    df = df[df["label"] == "genuine"].reset_index(drop=True)
    df["sentence"] = [
        int(m.group(1)) if (m := re.search(r"_s(\d+)$", c)) else np.nan for c in df["clip_id"]
    ]
    with open(F0_REF_JSON) as f:
        f0_refs = json.load(f)
    return df, f0_refs


def rereference_f0(rows: pd.DataFrame, own_ref_hz: float, target_ref_hz: float) -> pd.DataFrame:
    rows = rows.copy()
    rows["f0_mean"] = rows["f0_mean"] + 12 * np.log2(own_ref_hz / target_ref_hz)
    return rows


def _impostor_rows(df: pd.DataFrame, f0_refs: dict, target: str) -> pd.DataFrame:
    return pd.concat(
        [
            rereference_f0(df[df["speaker"] == other], f0_refs[other], f0_refs[target])
            for other in sorted(df["speaker"].unique())
            if other != target
        ]
    )


def _trials(target: str, genuine: pd.DataFrame, gen_scores, impostors: pd.DataFrame, imp_scores):
    return [(target, target, c, 1, float(s)) for c, s in zip(genuine["clip_id"], gen_scores)] + [
        (target, sp, c, 0, float(s))
        for sp, c, s in zip(impostors["speaker"], impostors["clip_id"], imp_scores)
    ]


def trial_scores(
    df: pd.DataFrame, f0_refs: dict, components: Components, protocol: str = "heldout"
) -> pd.DataFrame:
    out = []
    for target in sorted(df["speaker"].unique()):
        own = df[df["speaker"] == target].reset_index(drop=True)
        impostors = _impostor_rows(df, f0_refs, target)

        if protocol == "heldout":
            enroll = own[own["sentence"] < HELDOUT_FIRST_SENTENCE]
            test = own[own["sentence"] >= HELDOUT_FIRST_SENTENCE]
            impostors = impostors[impostors["sentence"] >= HELDOUT_FIRST_SENTENCE]
            det = FingerprintDistanceDetector.train(target, enroll, f0_refs[target], components=components)
            out += _trials(target, test, det.decision(test), impostors, det.decision(impostors))
        elif protocol == "loo":
            gen_scores = [
                FingerprintDistanceDetector.train(
                    target, own.drop(index=i), f0_refs[target], components=components
                ).decision(own.iloc[[i]])[0]
                for i in range(len(own))
            ]
            det = FingerprintDistanceDetector.train(target, own, f0_refs[target], components=components)
            out += _trials(target, own, gen_scores, impostors, det.decision(impostors))
        else:
            raise ValueError(protocol)

    return pd.DataFrame(out, columns=["target", "clip_speaker", "clip_id", "is_genuine", "score"])


def summarize(trials: pd.DataFrame) -> dict:
    m = evaluate(trials["is_genuine"].to_numpy(), trials["score"].to_numpy())
    m["genuine_accept"] = float((trials.query("is_genuine == 1")["score"] >= 0).mean())
    m["impostor_reject"] = float((trials.query("is_genuine == 0")["score"] < 0).mean())
    return m


def comparison_table(df: pd.DataFrame, f0_refs: dict, protocol: str = "heldout") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"system": name, **summarize(trial_scores(df, f0_refs, comps, protocol))}
            for name, comps in SYSTEMS.items()
        ]
    )


def blend_shares(df: pd.DataFrame, f0_refs: dict) -> pd.DataFrame:
    """Held-out protocol, default blend: what share of the blended distance
    each component contributed, on genuine vs impostor trials. Checks that
    prosody is actually what drives the verdict, not just what's weighted."""
    rows = []
    for target in sorted(df["speaker"].unique()):
        own = df[df["speaker"] == target]
        det = FingerprintDistanceDetector.train(
            target, own[own["sentence"] < HELDOUT_FIRST_SENTENCE], f0_refs[target]
        )
        imp = _impostor_rows(df, f0_refs, target)
        for kind, rows_ in [
            ("genuine", own[own["sentence"] >= HELDOUT_FIRST_SENTENCE]),
            ("impostor", imp[imp["sentence"] >= HELDOUT_FIRST_SENTENCE]),
        ]:
            for _, r in rows_.iterrows():
                comps = det.score(r.to_dict())["components"]
                rows.append({"trial": kind, **{n: c["share"] for n, c in comps.items()}})
    return pd.DataFrame(rows).groupby("trial").mean()


def ablation_table(df: pd.DataFrame, f0_refs: dict, protocol: str = "heldout") -> pd.DataFrame:
    rows = []
    for group, cols in {"(none)": [], **PROSODY_GROUPS}.items():
        feats = [c for c in PROSODIC_FEATURES if c not in cols]
        rows.append(
            {"dropped": group, **summarize(trial_scores(df, f0_refs, {"prosody": (feats, 1.0)}, protocol))}
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)
    df, f0_refs = load_table()
    cols = ["roc_auc", "eer", "genuine_accept", "impostor_reject", "accuracy"]
    print(f"{len(df)} genuine clips, speakers={sorted(df['speaker'].unique())}")

    for protocol in ["heldout", "loo"]:
        print(f"\n== {protocol} ==")
        print(comparison_table(df, f0_refs, protocol).set_index("system")[cols].to_string())

    print("\nBlend: share of distance per component (heldout):")
    print(blend_shares(df, f0_refs).to_string())
    print("\nAblation, prosody only, drop one group (heldout):")
    print(ablation_table(df, f0_refs).set_index("dropped")[cols].to_string())
