from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    n_classes: int,
) -> Dict[str, float]:
    """Return a compact metrics dictionary for downstream reporting."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            if n_classes > 2:
                metrics["auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
            else:
                metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            pass
    return metrics

