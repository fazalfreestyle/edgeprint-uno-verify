import numpy as np
from typing import List

def weights_from_logloss(logloss_list: List[float]) -> np.ndarray:
    inv = 1 / np.clip(np.array(logloss_list, dtype=np.float64), 1e-9, None)
    return inv / inv.sum()

def soft_vote(prob_pairs_list: List[np.ndarray], weights: np.ndarray = None) -> np.ndarray:
    """
    prob_pairs_list: list of (N,2) arrays [p0, p1] from calibrated classifiers.
    Returns fused positive probs (N,).
    """
    P = np.stack(prob_pairs_list)  # (M, N, 2)
    if weights is None:
        weights = np.ones(P.shape[0]) / P.shape[0]
    fused = (P * weights.reshape(-1, 1, 1)).sum(axis=0)
    return fused[:, 1]

def cascade_probs(p_fast: np.ndarray, p_slow: np.ndarray, tau1: float):
    """
    Early-exit cascade: take fast model if p_fast >= tau1; else average fast+slow.
    Returns (final_probs, took_slow_mask).
    """
    take_fast = p_fast >= tau1
    out = p_fast.copy()
    out[~take_fast] = 0.5 * p_fast[~take_fast] + 0.5 * p_slow[~take_fast]
    return out, ~take_fast
