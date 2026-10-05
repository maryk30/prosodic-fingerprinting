"""Stage 5c: cross-speaker impostor evaluation — real two-class numbers
without any synthetic clones.

Protocol, per target speaker T:
  - genuine trials:  each of T's clips, scored by a model trained on T's
                     *other* clips (leave-one-out — never scored in-sample)
  - impostor trials: every other speaker's clips, scored by T's model
                     trained on all of T's clips
Scores are pooled over targets, then ROC-AUC / EER via eval.py.

This is an *impostor* task (a different human claiming to be T), not the
clone task VoiceGuard is ultimately about. A clone matches T's pitch
register and timbre, so the closest proxy here is the `prosodic_no_f0mean`
feature set — absolute pitch level removed, leaving only the habit-type
features (pitch *variability*, rhythm, pauses, rate, voice quality). Read
that row as the most relevant one; the full-prosodic and MFCC rows are
easier because pitch register / timbre differ between two real speakers.

F0 note: features.csv stores f0_mean in semitones relative to the clip
owner's own median pitch. A real impostor clip would be measured against
the *claimed* speaker's reference (that's what score_clip() does), so
impostor rows are re-referenced to the target before scoring.

With 2 speakers x 10 clips this is 20 genuine + 20 impostor trials — read
the numbers as indicative, not tight estimates.
"""

import json
import os
import sys
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from eval import evaluate  # noqa: E402
from fingerprint import NUMERIC_FEATURES  # noqa: E402
from spectral_features import MFCC_FEATURE_COLUMNS  # noqa: E402
from distance import FingerprintDistanceDetector  # noqa: E402

FEATURES_CSV = "data/features/features.csv"
SPECTRAL_CSV = "data/features/spectral_features.csv"
F0_REF_JSON = "data/features/f0_reference.json"

FEATURE_SETS = {
    "prosodic": NUMERIC_FEATURES,
    "prosodic_no_f0mean": [c for c in NUMERIC_FEATURES if c != "f0_mean"],
    "spectral_mfcc": MFCC_FEATURE_COLUMNS,
}

# For ablation: drop one group at a time from the clone-proxy feature set.
FEATURE_GROUPS = {
    "pitch_variability": ["f0_std"],
    "energy": ["energy_mean", "energy_std"],
    "rate_rhythm": ["speaking_rate_mean", "npvi"],
    "pauses": ["pause_count", "pause_mean_dur", "pause_var_dur"],
    "voice_quality": ["jitter", "shimmer"],
}

# A detector factory: trained on an (n_clips, n_features) array + the feature
# names, returns a scoring function mapping an array to scores (higher = more
# genuine-like).
DetectorFactory = Callable[[np.ndarray, list[str]], Callable[[np.ndarray], np.ndarray]]


def distance_factory(X: np.ndarray, features: list[str]):
    rows = pd.DataFrame(X, columns=features)
    return FingerprintDistanceDetector.train("_", rows, np.nan, features=features).decision


def ocsvm_factory(kernel: str) -> DetectorFactory:
    def factory(X: np.ndarray, features: list[str]):
        scaler = StandardScaler().fit(X)
        model = OneClassSVM(kernel=kernel, nu=0.1, gamma="scale").fit(scaler.transform(X))
        return lambda Z: model.decision_function(scaler.transform(Z))

    return factory


DETECTORS: dict[str, DetectorFactory] = {
    "distance (new default)": distance_factory,
    "ocsvm_linear (old default)": ocsvm_factory("linear"),
    "ocsvm_rbf": ocsvm_factory("rbf"),
}


def load_table() -> tuple[pd.DataFrame, dict]:
    """Genuine prosodic + MFCC features joined per clip."""
    pros = pd.read_csv(FEATURES_CSV)
    spec = pd.read_csv(SPECTRAL_CSV)
    df = pros.merge(
        spec[["speaker", "clip_id", "label"] + MFCC_FEATURE_COLUMNS],
        on=["speaker", "clip_id", "label"],
        how="inner",
    )
    df = df[df["label"] == "genuine"].reset_index(drop=True)
    with open(F0_REF_JSON) as f:
        f0_refs = json.load(f)
    return df, f0_refs


def rereference_f0(rows: pd.DataFrame, own_ref_hz: float, target_ref_hz: float) -> pd.DataFrame:
    rows = rows.copy()
    rows["f0_mean"] = rows["f0_mean"] + 12 * np.log2(own_ref_hz / target_ref_hz)
    return rows


def trial_scores(
    df: pd.DataFrame, f0_refs: dict, features: list[str], factory: DetectorFactory
) -> pd.DataFrame:
    out = []
    for target in sorted(df["speaker"].unique()):
        own = df[df["speaker"] == target].reset_index(drop=True)
        X_own = own[features].to_numpy(dtype=float)

        for i in range(len(own)):
            scorer = factory(np.delete(X_own, i, axis=0), features)
            out.append((target, target, own.loc[i, "clip_id"], 1, float(scorer(X_own[i : i + 1])[0])))

        scorer = factory(X_own, features)
        for other in sorted(df["speaker"].unique()):
            if other == target:
                continue
            imp = rereference_f0(df[df["speaker"] == other], f0_refs[other], f0_refs[target])
            scores = scorer(imp[features].to_numpy(dtype=float))
            out += [(target, other, cid, 0, float(s)) for cid, s in zip(imp["clip_id"], scores)]

    return pd.DataFrame(out, columns=["target", "clip_speaker", "clip_id", "is_genuine", "score"])


def summarize(trials: pd.DataFrame) -> dict:
    m = evaluate(trials["is_genuine"].to_numpy(), trials["score"].to_numpy())
    m["genuine_accept"] = float((trials.query("is_genuine == 1")["score"] >= 0).mean())
    m["impostor_reject"] = float((trials.query("is_genuine == 0")["score"] < 0).mean())
    return m


def comparison_table(df: pd.DataFrame, f0_refs: dict) -> pd.DataFrame:
    rows = []
    for set_name, features in FEATURE_SETS.items():
        for det_name, factory in DETECTORS.items():
            m = summarize(trial_scores(df, f0_refs, features, factory))
            rows.append({"features": set_name, "detector": det_name, **m})
    return pd.DataFrame(rows)


def ablation_table(df: pd.DataFrame, f0_refs: dict, base_set: str = "prosodic_no_f0mean") -> pd.DataFrame:
    base = FEATURE_SETS[base_set]
    rows = [{"dropped": "(none)", **summarize(trial_scores(df, f0_refs, base, distance_factory))}]
    for group, cols in FEATURE_GROUPS.items():
        features = [c for c in base if c not in cols]
        rows.append({"dropped": group, **summarize(trial_scores(df, f0_refs, features, distance_factory))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)
    df, f0_refs = load_table()
    cols = ["roc_auc", "eer", "genuine_accept", "impostor_reject", "accuracy"]

    print(f"Cross-speaker impostor eval: {len(df)} genuine clips, speakers={sorted(df['speaker'].unique())}\n")
    print(comparison_table(df, f0_refs).set_index(["features", "detector"])[cols].to_string(), "\n")
    print("Ablation (distance detector, prosodic_no_f0mean, drop one group):")
    print(ablation_table(df, f0_refs).set_index("dropped")[cols].to_string())
