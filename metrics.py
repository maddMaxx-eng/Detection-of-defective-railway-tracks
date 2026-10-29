"""Classification metrics shared by training (validation) and evaluation (test)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from . import config


def compute_metrics(y_true, y_pred, class_names: list[str] | None = None) -> dict:
    """Accuracy, per-class precision/recall/F1, macro/weighted F1, confusion matrix.

    Macro averages only include classes that have at least one true sample, so a
    class with no test images cannot silently drag the score down; such classes
    are listed under ``classes_without_support``.
    """
    names = class_names or config.CLASS_NAMES
    labels = list(range(len(names)))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    supported = [i for i in labels if s[i] > 0]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    def macro(arr):
        return float(np.mean([arr[i] for i in supported])) if supported else 0.0

    total = int(s.sum())
    weighted_f1 = float(np.sum(f * s) / total) if total else 0.0
    return {
        "num_samples": total,
        "accuracy": float(accuracy_score(y_true, y_pred)) if total else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if total else 0.0,
        "macro_precision": macro(p),
        "macro_recall": macro(r),
        "macro_f1": macro(f),
        "weighted_f1": weighted_f1,
        "per_class": {
            names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in labels
        },
        "confusion_matrix": cm.tolist(),
        "class_names": list(names),
        "classes_without_support": [names[i] for i in labels if s[i] == 0],
    }
