from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy) * 100.0,
        "precision": float(precision) * 100.0,
        "recall": float(recall) * 100.0,
        "f1": float(f1) * 100.0,
    }
