import numpy as np
from .data_io import BlockSplit

def make_block_splits(n_blocks: int, train_frac=0.6, val_frac=0.2, seed: int = 42) -> BlockSplit:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_blocks)
    rng.shuffle(idx)
    n_train = int(train_frac * n_blocks)
    n_val = int(val_frac * n_blocks)
    return BlockSplit(
        train=idx[:n_train].tolist(),
        val=idx[n_train:n_train + n_val].tolist(),
        test=idx[n_train + n_val:].tolist(),
    )
