# Fast sweep: LR only, calibrated, per-target append. Optional target filter via CLI.
# Usage: python scripts/verify_all_targets_fast.py         (all B1..B10)
#        python scripts/verify_all_targets_fast.py B1 B2   (subset)

import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from edgeprint import data_io, splits, models_tabular, metrics as M, utils

BOARDS = [f"B{i}" for i in range(1, 11)]
only = set(sys.argv[1:])
if only:
    BOARDS = [b for b in BOARDS if b in only]

BLOCK_SIZE = 1000
# caps per class to keep runs snappy; lower if still slow, raise if you want more data
MAX_POS_TR, MAX_NEG_TR = 60000, 60000
MAX_POS_VA, MAX_NEG_VA = 30000, 30000

def cyclic_groups(ti, k_seen=6, k_unseen=3):
    seen = [BOARDS[(ti + i + 1) % len(BOARDS)] for i in range(k_seen)]
    unseen = [BOARDS[(ti + k_seen + i + 1) % len(BOARDS)] for i in range(k_unseen)]
    return seen, unseen

def choose_tau(p, y, tpr_min=0.99, fpr_max=0.01):
    th = np.linspace(0, 1, 1001)
    best = None
    for t in th:
        tp = ((p >= t) & (y == 1)).sum(); fn = ((p <  t) & (y == 1)).sum()
        fp = ((p >= t) & (y == 0)).sum(); tn = ((p <  t) & (y == 0)).sum()
        tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
        if tpr >= tpr_min and fpr <= fpr_max: best = (t, tpr, fpr); break
    if best: return best[0]
    # fallback: maximize TPR under fpr_max, else Youden J
    th_ok, best_tpr = None, -1
    for t in th:
        tp = ((p >= t) & (y == 1)).sum(); fn = ((p <  t) & (y == 1)).sum()
        fp = ((p >= t) & (y == 0)).sum(); tn = ((p <  t) & (y == 0)).sum()
        tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
        if fpr <= fpr_max and tpr > best_tpr: th_ok, best_tpr = t, tpr
    if th_ok is not None: return th_ok
    # last resort
    youden, tau = -1, 0.5
    for t in th:
        tp = ((p >= t) & (y == 1)).sum(); fn = ((p <  t) & (y == 1)).sum()
        fp = ((p >= t) & (y == 0)).sum(); tn = ((p <  t) & (y == 0)).sum()
        tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
        if (tpr - fpr) > youden: youden, tau = (tpr - fpr), t
    return tau

def cap_class(X, y, label, cap, rng):
    idx = np.where(y == label)[0]
    if len(idx) > cap: idx = rng.choice(idx, size=cap, replace=False)
    return X[idx], np.full(len(idx), label, int)

def main():
    utils.set_seed(42)
    # Load all boards once
    per_blocks = {}
    for b in BOARDS:
        X = data_io.load_board_csv(f"data/boards/{b}.csv")
        per_blocks[b] = data_io.to_blocks(X, BLOCK_SIZE)
    # equalize #blocks to prevent leakage
    n_blocks = min(b.shape[0] for b in per_blocks.values())
    for k in per_blocks: per_blocks[k] = per_blocks[k][:n_blocks]
    sp = splits.make_block_splits(n_blocks, seed=123)

    OUT = os.path.abspath(os.path.join("runs", f"fast_bs{BLOCK_SIZE}_{time.strftime('%Y%m%d-%H%M%S')}"))
    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, "verify_all_targets_fast.csv")
    print("Saving rows to:", csv_path)

    t0 = time.time()
    for ti, tgt in enumerate(BOARDS):
        seen, unseen = cyclic_groups(ti, 6, 3)

        Xp_tr = data_io.materialize_blocks(per_blocks[tgt], sp.train)
        Xp_va = data_io.materialize_blocks(per_blocks[tgt], sp.val)
        Xp_te = data_io.materialize_blocks(per_blocks[tgt], sp.test)
        stack = lambda names, which: np.row_stack([data_io.materialize_blocks(per_blocks[n], getattr(sp, which)) for n in names])
        Xn_tr, Xn_va, Xn_te = stack(seen, "train"), stack(seen, "val"), stack(unseen, "test")

        Xtr = np.row_stack([Xp_tr, Xn_tr]); ytr = np.concatenate([np.ones(len(Xp_tr), int), np.zeros(len(Xn_tr), int)])
        Xva = np.row_stack([Xp_va, Xn_va]); yva = np.concatenate([np.ones(len(Xp_va), int), np.zeros(len(Xn_va), int)])
        Xte = np.row_stack([Xp_te, Xn_te]); yte = np.concatenate([np.ones(len(Xp_te), int), np.zeros(len(Xn_te), int)])

        # cap size for speed
        rng = np.random.default_rng(42)
        Xp_tr_c, yp_tr_c = cap_class(Xtr, ytr, 1, MAX_POS_TR, rng)
        Xn_tr_c, yn_tr_c = cap_class(Xtr, ytr, 0, MAX_NEG_TR, rng)
        Xtr = np.row_stack([Xp_tr_c, Xn_tr_c]); ytr = np.concatenate([yp_tr_c, yn_tr_c])

        Xp_va_c, yp_va_c = cap_class(Xva, yva, 1, MAX_POS_VA, rng)
        Xn_va_c, yn_va_c = cap_class(Xva, yva, 0, MAX_NEG_VA, rng)
        Xva = np.row_stack([Xp_va_c, Xn_va_c]); yva = np.concatenate([yp_va_c, yn_va_c])

        # LR + calibration (fast & strong)
        lr = models_tabular.pipe_lr().fit(Xtr, ytr)
        clr = CalibratedClassifierCV(lr, cv="prefit", method="sigmoid").fit(Xva, yva)

        pva = clr.predict_proba(Xva)[:,1]; pte = clr.predict_proba(Xte)[:,1]
        val = M.verification(yva, pva); tst = M.verification(yte, pte)

        tau = choose_tau(pva, yva, 0.99, 0.01)
        tpr_te = ((pte >= tau) & (yte == 1)).sum() / max(1, (yte == 1).sum())
        fpr_te = ((pte >= tau) & (yte == 0)).sum() / max(1, (yte == 0).sum())

        row = {
            "target": tgt, "seen": ",".join(seen), "unseen": ",".join(unseen),
            "val_lr_auroc": val["auroc"], "val_lr_auprc": val["auprc"],
            "val_lr_eer": val["eer"], "val_lr_tpr_at_1pct_fpr": val["tpr_at_1pct_fpr"], "val_lr_tpr_at_0p1pct_fpr": val["tpr_at_0p1pct_fpr"],
            "test_lr_auroc": tst["auroc"], "test_lr_auprc": tst["auprc"],
            "test_lr_eer": tst["eer"], "test_lr_tpr_at_1pct_fpr": tst["tpr_at_1pct_fpr"], "test_lr_tpr_at_0p1pct_fpr": tst["tpr_at_0p1pct_fpr"],
            "frozen_tau": float(tau), "test_tpr_at_tau": float(tpr_te), "test_fpr_at_tau": float(fpr_te),
            "block_size": BLOCK_SIZE, "n_blocks": int(n_blocks),
        }

        # append per target
        pd.DataFrame([row]).to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))
        print(f"Done {tgt}: unseen={unseen} -> wrote row to {csv_path}")

    print("All done. Results:", csv_path, "\nElapsed (s):", time.time() - t0)

if __name__ == "__main__":
    main()
