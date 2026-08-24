"""Stage 5: One-Class SVM prosodic anomaly detector.

Trained per-speaker on their genuine enrollment clips only — matches the
real deployment (you enroll a known speaker, then score candidate clips
against their habitual prosody; no synthetic data needed at training time).
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fingerprint import NUMERIC_FEATURES  # noqa: E402

MODEL_DIR = "data/features/models"
DEFAULT_NU = 0.1
DEFAULT_KERNEL = "linear"  # rbf badly overfits with ~9 training clips in 11-dim feature space
DEFAULT_GAMMA = "scale"


class SpeakerAnomalyDetector:
    def __init__(self, speaker: str, scaler: StandardScaler, model: OneClassSVM):
        self.speaker = speaker
        self.scaler = scaler
        self.model = model

    @classmethod
    def train(
        cls,
        speaker: str,
        clip_rows: pd.DataFrame,
        nu: float = DEFAULT_NU,
        kernel: str = DEFAULT_KERNEL,
        gamma: str = DEFAULT_GAMMA,
    ) -> "SpeakerAnomalyDetector":
        X = clip_rows[NUMERIC_FEATURES].to_numpy(dtype=float)
        scaler = StandardScaler().fit(X)
        model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma).fit(scaler.transform(X))
        return cls(speaker, scaler, model)

    def score(self, feature_row: dict) -> dict:
        x = np.array([[feature_row[c] for c in NUMERIC_FEATURES]], dtype=float)
        xs = self.scaler.transform(x)
        decision = float(self.model.decision_function(xs)[0])  # >0 inlier, <0 outlier
        pred = int(self.model.predict(xs)[0])  # 1 inlier, -1 outlier
        return {"prediction": "genuine" if pred == 1 else "synthetic", "confidence": decision}

    def save(self, out_dir: str = MODEL_DIR) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.speaker}_oneclass.joblib")
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)
        return path

    @classmethod
    def load(cls, speaker: str, out_dir: str = MODEL_DIR) -> "SpeakerAnomalyDetector":
        path = os.path.join(out_dir, f"{speaker}_oneclass.joblib")
        data = joblib.load(path)
        return cls(speaker, data["scaler"], data["model"])


def leave_one_out_eval(
    speaker: str,
    clip_rows: pd.DataFrame,
    nu: float = DEFAULT_NU,
    kernel: str = DEFAULT_KERNEL,
    gamma: str = DEFAULT_GAMMA,
) -> tuple[float, list[str]]:
    """No synthetic data yet, so this only checks self-consistency: does the
    model trained on N-1 genuine clips still accept the held-out genuine
    clip? With ~9 training clips per fold this is a noisy signal, not a
    hyperparameter-tuning target — don't chase a higher number here by
    grid-searching against this same tiny holdout, that's just overfitting
    to noise. Real validation happens in Stage 7 once synthetic clips exist."""
    clip_rows = clip_rows.reset_index(drop=True)
    predictions = []
    for i in range(len(clip_rows)):
        train_rows = clip_rows.drop(index=i)
        test_row = clip_rows.iloc[i]
        detector = SpeakerAnomalyDetector.train(speaker, train_rows, nu=nu, kernel=kernel, gamma=gamma)
        predictions.append(detector.score(test_row.to_dict())["prediction"])
    acceptance_rate = predictions.count("genuine") / len(predictions)
    return acceptance_rate, predictions


if __name__ == "__main__":
    df = pd.read_csv("data/features/features.csv")
    genuine = df[df["label"] == "genuine"]

    for speaker, group in genuine.groupby("speaker"):
        acc, preds = leave_one_out_eval(speaker, group)
        print(f"{speaker}: leave-one-out genuine-acceptance rate = {acc:.0%} {preds}")

        detector = SpeakerAnomalyDetector.train(speaker, group)
        path = detector.save()
        print(f"  saved model -> {path}")
