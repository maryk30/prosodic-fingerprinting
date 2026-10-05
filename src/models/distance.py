"""Stage 5c/3b: blended fingerprint-distance detector (the default detector).

A speaker's model is a set of weighted *components* — prosody, voice
quality, timbre (MFCC), pitch register. For each component the speaker's
enrollment clips give a per-feature mean/std; a candidate clip's distance
on that component is how many standard deviations it sits from that
profile (RMS z-score, i.e. diagonal Mahalanobis / sqrt(n_features)). The
clip's overall distance is the weighted average of component distances:

    distance = sum_c w_c * min(rms_z_c, cap_c) / sum_c w_c

Weights and caps are hand-set (DEFAULT_COMPONENTS), not fitted: prosody
carries 0.60 so speaking habit decides the verdict; pitch register and
timbre — the things a voice clone copies best — can only nudge it. With 2
speakers and ~20 enrollment clips there isn't enough data to fit weights
without overfitting the test clips, so they were fixed before evaluation.

Why caps as well as weights: a weight limits how much a component *can*
count, not how much it *does*. Uncapped, pitch register (weight 0.10, a
single feature) jumped from ~0.8 to 5 sd for a different speaker and drove
~69% of the genuine-vs-impostor gap. Capping the non-prosody components at
2 sd — roughly the edge of the genuine range — means they can flag a
mismatch but how far beyond it doesn't matter: register adds at most
0.10 x 2 = 0.2 to the distance, prosody typically ~0.6.

The accept threshold comes from leave-one-out distances of the enrollment
clips: the `quantile` (default 0.9) of how far held-out genuine clips land,
so ~10% genuine false-rejects by construction.

Why not the linear One-Class SVM (src/models/oneclass.py, kept for
comparison): it separates data from the *origin*, and after standardization
the enrollment data is centred on the origin, so its boundary passed through
the genuine clips (rho ~1e-8, |w| ~0, ~50% self-acceptance).
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import extract_raw_features, to_row  # noqa: E402
from fingerprint import PROSODIC_FEATURES, REGISTER_FEATURES, VOICE_QUALITY_FEATURES  # noqa: E402
from spectral_features import MFCC_FEATURE_COLUMNS, extract_mfcc_row  # noqa: E402

MODEL_DIR = "data/features/models"
DEFAULT_QUANTILE = 0.9

NON_PROSODY_CAP = 2.0  # sd

# name -> (features, weight, cap on the component's distance or None)
Components = dict[str, tuple[list[str], float, float | None]]
DEFAULT_COMPONENTS: Components = {
    "prosody": (PROSODIC_FEATURES, 0.60, None),
    "voice_quality": (VOICE_QUALITY_FEATURES, 0.15, NON_PROSODY_CAP),
    "timbre": (MFCC_FEATURE_COLUMNS, 0.15, NON_PROSODY_CAP),
    "pitch_register": (REGISTER_FEATURES, 0.10, NON_PROSODY_CAP),
}

# Floor on per-feature std, so a feature that happens to be (near-)constant
# across the enrollment clips can't blow up every z-score (krishiv once had
# pause_var_dur = 0 on every clip, and a single 60ms pause landed at
# z ≈ 60,000). Floors are the feature's measurement resolution where one
# exists, else a fraction of |mean|.
STD_FLOOR_REL = 0.05
STD_FLOOR_ABS = 1e-6
MIN_STD = {
    "f0_mean": 0.5,  # semitones
    "f0_std": 0.25,  # semitones
    "f0_range": 0.25,  # semitones
    "f0_slope": 0.25,  # semitones/s
    "f0_final_move": 0.5,  # semitones
    "energy_slope": 0.25,  # dB/s
    "pause_count": 0.5,
    "pause_mean_dur": 0.03,  # s, one VAD frame
    "pause_var_dur": 0.03**2,  # s^2
}
# Cap on any single feature's |z|, so one wild value (e.g. a pYIN octave
# error in f0_final_move) can't outvote every other feature in its component.
Z_CLIP = 5.0


def _fit_stats(X: np.ndarray, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0, ddof=1) if len(X) > 1 else np.zeros(X.shape[1])
    std = np.nan_to_num(std)
    floor = np.array([MIN_STD.get(f, 0.0) for f in features])
    floor = np.maximum(floor, np.maximum(STD_FLOOR_REL * np.abs(mean), STD_FLOOR_ABS))
    return mean, np.maximum(std, floor)


def _rms_z(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-row RMS z-score. A feature that couldn't be measured on a clip
    (NaN, e.g. varco_uv with too few unvoiced stretches) is left out of that
    clip's average rather than counted as typical or atypical."""
    z = np.clip((X - mean) / std, -Z_CLIP, Z_CLIP)
    return np.sqrt(np.nanmean(z**2, axis=1))


def _fit_components(rows: pd.DataFrame, components: Components) -> dict:
    return {
        name: _fit_stats(rows[feats].to_numpy(dtype=float), feats)
        for name, (feats, _, _) in components.items()
    }


def _blend(rows: pd.DataFrame, stats: dict, components: Components) -> tuple[np.ndarray, dict]:
    """Returns (blended distance, {component: raw uncapped distance})."""
    total_w = sum(w for _, w, _ in components.values())
    parts = {
        name: _rms_z(rows[feats].to_numpy(dtype=float), *stats[name])
        for name, (feats, _, _) in components.items()
    }
    blended = sum(
        w * (np.minimum(parts[name], cap) if cap is not None else parts[name])
        for name, (_, w, cap) in components.items()
    ) / total_w
    return blended, parts


class FingerprintDistanceDetector:
    def __init__(
        self,
        speaker: str,
        stats: dict,
        threshold: float,
        f0_ref_hz: float,
        components: Components = DEFAULT_COMPONENTS,
    ):
        self.speaker = speaker
        self.stats = stats
        self.threshold = threshold
        self.f0_ref_hz = f0_ref_hz
        self.components = components

    @property
    def features(self) -> list[str]:
        return [f for feats, _, _ in self.components.values() for f in feats]

    @classmethod
    def train(
        cls,
        speaker: str,
        clip_rows: pd.DataFrame,
        f0_ref_hz: float,
        quantile: float = DEFAULT_QUANTILE,
        components: Components = DEFAULT_COMPONENTS,
    ) -> "FingerprintDistanceDetector":
        """clip_rows: the speaker's genuine enrollment clips, with every
        component's columns (prosodic features.csv columns joined with the
        MFCC spectral_features.csv columns when timbre is a component)."""
        clip_rows = clip_rows.reset_index(drop=True)
        stats = _fit_components(clip_rows, components)
        # Threshold from leave-one-out: how far does a genuine clip land from
        # a fingerprint built without it? In-sample distances would set the
        # threshold too tight (each clip helped define the mean).
        loo = [
            _blend(
                clip_rows.iloc[[i]],
                _fit_components(clip_rows.drop(index=i), components),
                components,
            )[0][0]
            for i in range(len(clip_rows))
        ]
        threshold = float(np.quantile(loo, quantile))
        return cls(speaker, stats, threshold, f0_ref_hz, components)

    def distance(self, rows: pd.DataFrame) -> np.ndarray:
        return _blend(rows, self.stats, self.components)[0]

    def decision(self, rows: pd.DataFrame) -> np.ndarray:
        """>0 = inside the genuine envelope, <0 = outside. Same sign
        convention as eval.evaluate()."""
        return self.threshold - self.distance(rows)

    def score(self, feature_row: dict) -> dict:
        blended, parts = _blend(pd.DataFrame([feature_row]), self.stats, self.components)
        d = float(blended[0])
        decision = self.threshold - d
        total_w = sum(w for _, w, _ in self.components.values())

        def contribution(name: str) -> float:
            _, w, cap = self.components[name]
            dist = float(parts[name][0])
            return w * (min(dist, cap) if cap is not None else dist) / total_w

        return {
            "prediction": "genuine" if decision >= 0 else "synthetic",
            "confidence": decision,
            "distance": d,
            "threshold": self.threshold,
            # per-component raw distance, whether the cap bit, and its share
            # of the blended distance (after capping)
            "components": {
                name: {
                    "distance": float(parts[name][0]),
                    "capped": cap is not None and float(parts[name][0]) > cap,
                    "share": contribution(name) / d if d else 0.0,
                }
                for name, (_, _, cap) in self.components.items()
            },
        }

    def score_clip(self, path: str) -> dict:
        raw = extract_raw_features(path, speaker=self.speaker, label="candidate", tts_system="na")
        row = to_row(raw, self.f0_ref_hz)
        if set(MFCC_FEATURE_COLUMNS) & set(self.features):
            row.update(extract_mfcc_row(path, speaker=self.speaker, label="candidate", tts_system="na"))
        return self.score(row)

    def save(self, out_dir: str = MODEL_DIR) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.speaker}_distance.joblib")
        joblib.dump(
            {
                "stats": self.stats,
                "threshold": self.threshold,
                "f0_ref_hz": self.f0_ref_hz,
                "components": self.components,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, speaker: str, out_dir: str = MODEL_DIR) -> "FingerprintDistanceDetector":
        data = joblib.load(os.path.join(out_dir, f"{speaker}_distance.joblib"))
        return cls(speaker, data["stats"], data["threshold"], data["f0_ref_hz"], data["components"])
