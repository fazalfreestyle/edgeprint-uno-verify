import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class BlockSplit:
    train: List[int]
    val: List[int]
    test: List[int]

def load_board_csv(path: str) -> np.ndarray:
    """
    Load a CSV shaped [96,N] or [N,96] and return float32 array (N, 96).
    Drops rows with NaN/Inf.
    """
    X = pd.read_csv(path, header=None).values.astype(np.float32)
    if X.shape[0] == 96 and X.shape[1] != 96:
        X = X.T
    elif X.shape[1] != 96:
        raise ValueError(f"Expected 96 features; got {X.shape} from {path}")
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
