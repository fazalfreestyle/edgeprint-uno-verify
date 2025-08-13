import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve
)

def identification(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def _eer_from_scores(y_true, scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)

def tpr_at_fpr(y_true, scores, fpr_target=0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    order = np.argsort(fpr)
    fpr = fpr[order]; tpr = tpr[order]
    if fpr_target <= fpr.min(): return float(tpr[0])
    if fpr_target >= fpr.max(): return float(tpr[-1])
    return float(np.interp(fpr_target, fpr, tpr))

def verification(y_true, scores) -> Dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "eer": _eer_from_scores(y_true, scores),
        "tpr_at_1pct_fpr": tpr_at_fpr(y_true, scores, 0.01),
        "tpr_at_0p1pct_fpr": tpr_at_fpr(y_true, scores, 0.001),
    }
