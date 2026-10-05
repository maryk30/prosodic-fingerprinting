"""Stage 5c: fingerprint-distance anomaly detector (replaces the linear OC-SVM
as the default).

Why not the linear One-Class SVM: a linear OC-SVM separates the data from
the *origin*. After StandardScaler the enrollment data is centred on the
origin, so the learned hyperplane passes straight through it — on the
training clips themselves rho ≈ 1e-8, |w| ≈ 0.005, and every decision value
is within ±0.001. That (not just sample size) is why self-acceptance sat at
~50% and `cli.py score` confidences were all ≈ 0.

This detector is the direct use of the Stage 4 fingerprint idea: a speaker
is their per-feature mean/std over genuine clips, and a candidate clip is
scored by how many standard deviations it sits from that (RMS z-score,
i.e. diagonal Mahalanobis distance / sqrt(n_features)). No kernel or gamma
to tune. The accept threshold comes from leave-one-out distances of the
enrollment clips themselves: the `quantile` (default 0.9, mirroring the old
nu=0.1) of how far genuine held-out clips land, so ~10% genuine
false-rejects by construction.

Same interface as SpeakerAnomalyDetector (train/score/score_clip/save/load)
so cli.py can use either.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import extract_raw_features, to_row  # noqa: E402
from fingerprint import NUMERIC_FEATURES  # noqa: E402

MODEL_DIR = "data/features/models"
DEFAULT_QUANTILE = 0.9
# Floor on per-feature std, so a feature that happens to be (near-)constant
# across ~9 enrollment clips can't blow up every z-score. krishiv has
# pause_var_dur = 0 on every clip; with a ~0 floor a single 60ms pause landed
# at z ≈ 60,000 and dominated everything. Floors are set at the feature's
# measurement resolution where one exists (VAD frames are 30ms; one pause;
# half a semitone), else a fraction of |mean|.
STD_FLOOR_REL = 0.05
STD_FLOOR_ABS = 1e-6
MIN_STD = {
    "f0_mean": 0.5,  # semitones
    "f0_std": 0.25,  # semitones
    "pause_count": 0.5,
    "pause_mean_dur": 0.03,  # s, one VAD frame
    "pause_var_dur": 0.03**2,  # s^2
}


def _fit_stats(X: np.ndarray, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1) if len(X) > 1 else np.zeros(X.shape[1])
    floor = np.array([MIN_STD.get(f, 0.0) for f in features])
    floor = np.maximum(floor, np.maximum(STD_FLOOR_REL * np.abs(mean), STD_FLOOR_ABS))
    return mean, np.maximum(std, floor)


def _rms_z(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(((X - mean) / std) ** 2, axis=1))


class FingerprintDistanceDetector:
    def __init__(
        self,
        speaker: str,
        mean: np.ndarray,
        std: np.ndarray,
        threshold: float,
        f0_ref_hz: float,
        features: list[str] = NUMERIC_FEATURES,
    ):
        self.speaker = speaker
        self.mean = mean
        self.std = std
        self.threshold = threshold
        self.f0_ref_hz = f0_ref_hz
        self.features = list(features)

    @classmethod
    def train(
        cls,
        speaker: str,
        clip_rows: pd.DataFrame,
        f0_ref_hz: float,
        quantile: float = DEFAULT_QUANTILE,
        features: list[str] = NUMERIC_FEATURES,
    ) -> "FingerprintDistanceDetector":
        X = clip_rows[features].to_numpy(dtype=float)
        mean, std = _fit_stats(X, features)
        # Threshold from leave-one-out: how far does a genuine clip land from
        # a fingerprint built without it? Using in-sample distances instead
        # would set the threshold too tight (each clip helped define the mean).
        loo = [
            _rms_z(X[i : i + 1], *_fit_stats(np.delete(X, i, axis=0), features))[0] for i in range(len(X))
        ]
        threshold = float(np.quantile(loo, quantile))
        return cls(speaker, mean, std, threshold, f0_ref_hz, features)

    def distance(self, X: np.ndarray) -> np.ndarray:
        return _rms_z(np.atleast_2d(X), self.mean, self.std)

    def decision(self, X: np.ndarray) -> np.ndarray:
        """>0 = inside the genuine envelope, <0 = outside. Same sign
        convention as OneClassSVM.decision_function / eval.evaluate()."""
        return self.threshold - self.distance(X)

    def score(self, feature_row: dict) -> dict:
        x = np.array([[feature_row[c] for c in self.features]], dtype=float)
        d = float(self.distance(x)[0])
        decision = self.threshold - d
        return {
            "prediction": "genuine" if decision >= 0 else "synthetic",
            "confidence": decision,
            "distance": d,
            "threshold": self.threshold,
        }

    def score_clip(self, path: str) -> dict:
        raw = extract_raw_features(path, speaker=self.speaker, label="candidate", tts_system="na")
        row = to_row(raw, self.f0_ref_hz)
        return self.score(row)

    def save(self, out_dir: str = MODEL_DIR) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.speaker}_distance.joblib")
        joblib.dump(
            {
                "mean": self.mean,
                "std": self.std,
                "threshold": self.threshold,
                "f0_ref_hz": self.f0_ref_hz,
                "features": self.features,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, speaker: str, out_dir: str = MODEL_DIR) -> "FingerprintDistanceDetector":
        data = joblib.load(os.path.join(out_dir, f"{speaker}_distance.joblib"))
        return cls(
            speaker, data["mean"], data["std"], data["threshold"], data["f0_ref_hz"], data["features"]
        )


if __name__ == "__main__":
    import json

    df = pd.read_csv("data/features/features.csv")
    genuine = df[df["label"] == "genuine"]
    with open("data/features/f0_reference.json") as f:
        f0_refs = json.load(f)

    for speaker, group in genuine.groupby("speaker"):
        detector = FingerprintDistanceDetector.train(speaker, group, f0_refs[speaker])
        path = detector.save()
        print(f"{speaker}: threshold (LOO q{DEFAULT_QUANTILE:.0%}) = {detector.threshold:.3f}  -> {path}")
