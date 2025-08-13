import numpy as np
from sklearn.metrics import brier_score_loss

def ece_score(y_true, prob_pos, n_bins: int = 15) -> float:
    """Expected Calibration Error for binary probs (positive class)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (prob_pos >= lo) & (prob_pos < (hi if i < n_bins-1 else hi + 1e-9))
        if not np.any(mask):
            continue
        conf = prob_pos[mask].mean()
        acc = (y_true[mask] == (prob_pos[mask] >= 0.5)).mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)

def brier(y_true, prob_pos) -> float:
    return float(brier_score_loss(y_true, prob_pos))
