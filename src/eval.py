"""Stage 7: evaluation metric helpers.

EER, ROC-AUC, and accuracy/precision/recall/F1 for a detector's scores on a
set of clips labeled genuine (positive/inlier) vs synthetic (negative/outlier).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate: the point on the ROC curve where false-positive
    rate equals false-negative rate. Returns (eer, threshold)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thresholds[idx])


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """y_true: 1 = genuine, 0 = synthetic. y_score: higher = more genuine-like
    (e.g. OC-SVM decision_function output). Threshold at 0 for predicted labels."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= 0).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if len(set(y_true.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["eer"], _ = compute_eer(y_true, y_score)
    else:
        # Can't compute ranking metrics with only one class present (e.g. no
        # synthetic clips yet) — report as unavailable rather than a bogus number.
        metrics["roc_auc"] = float("nan")
        metrics["eer"] = float("nan")

    return metrics
