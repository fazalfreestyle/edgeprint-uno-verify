import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class BlockSplit:
    train: List[int]
    val: List[int]
    test: List[int]

def _drop_extra_columns(X: np.ndarray, expected: int) -> np.ndarray:
    """Heuristics to trim down to `expected` columns when there's 1 extra."""
    n_cols = X.shape[1]
    if n_cols == expected:
        return X
    if n_cols < expected:
        raise ValueError(f"Expected at least {expected} features; got {X.shape}")

    # Allow user override: drop the first column if EDGEPRINT_DROP_FIRST_COL=1
    if os.getenv("EDGEPRINT_DROP_FIRST_COL", "0") == "1":
        # Keep columns 1..expected (drop col 0)
        if n_cols >= expected + 1:
            return X[:, 1:1 + expected]
        # fallback
        return X[:, -expected:]

    # If exactly one extra column, try to detect an index-like or near-constant column
    if n_cols == expected + 1:
        idx_candidates = []
        near_const = []
        N = X.shape[0]

        for j in range(n_cols):
            col = X[:, j]
            # near-integer, monotonic index-like detection (e.g., 0..N-1 or 1..N)
            if np.all(np.isfinite(col)):
                if np.all(np.abs(col - np.round(col)) < 1e-6):
                    diffs = np.diff(col)
                    if (col.min() in (0, 1)) and np.all((diffs == 1) | (diffs == 0)):
                        idx_candidates.append(j)
                # near-constant column
                if np.nanstd(col) < 1e-8:
                    near_const.append(j)

        # Prefer dropping index-like; else near-constant
        to_drop = None
        if idx_candidates:
            to_drop = idx_candidates[0]
        elif near_const:
            to_drop = near_const[0]

        if to_drop is not None:
            keep = [k for k in range(n_cols) if k != to_drop]
            X = X[:, keep]
            if X.shape[1] == expected:
                return X

        # Fallback: keep the last `expected` columns (common when first col is an index)
        return X[:, -expected:]

    # If many extra columns, just keep the rightmost `expected`
    return X[:, -expected:]

def load_board_csv(path: str, expected_features: int = 96) -> np.ndarray:
    """
    Load a CSV shaped [96,N] or [N,96] and return float32 array (N, 96).
    Drops rows with NaN/Inf. Automatically trims one extra metadata column when present.
    Set EDGEPRINT_DROP_FIRST_COL=1 to force dropping the first column.
    """
    df = pd.read_csv(path, header=None)
    X = df.values

    # Try to coerce to float; if coercion fails, raise a helpful message
    try:
        X = X.astype(np.float32, copy=False)
    except Exception as e:
        raise ValueError(f"Non-numeric values found in {path}. Ensure the file has only numeric features.") from e

    # Transpose if features are rows
    if X.shape[0] == expected_features and X.shape[1] != expected_features:
        X = X.T

    # Trim extra columns if needed
    if X.shape[1] != expected_features:
        X = _drop_extra_columns(X, expected_features)

    # Final sanity checks
    if X.shape[1] != expected_features:
        raise ValueError(f"After trimming, still not {expected_features} features: got {X.shape} from {path}")

    # Drop any rows with non-finite values
    X = X[np.isfinite(X).all(axis=1)]
    return X

def to_blocks(X: np.ndarray, block_size: int = 1000) -> np.ndarray:
    """
    Group contiguous rows into non-overlapping blocks:
    returns (n_blocks, block_size, 96).
    """
    n = (len(X) // block_size) * block_size
    X = X[:n]
    return X.reshape(-1, block_size, X.shape[1])

def materialize_blocks(blocks: np.ndarray, indices: List[int]) -> np.ndarray:
    """Gather selected blocks and flatten back to (samples, 96)."""
    if not indices:
        return np.empty((0, blocks.shape[-1]), dtype=blocks.dtype)
    return blocks[indices].reshape(-1, blocks.shape[-1])
