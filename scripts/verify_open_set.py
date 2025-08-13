from edgeprint import data_io, splits, models_tabular, metrics, utils
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

def main():
    utils.set_seed(42)
    boards = {f"B{i}": f"data/boards/B{i}.csv" for i in range(1, 10 + 1)}
    target = "B1"
    seen = ["B2","B3","B4","B5","B6","B7"]
    unseen = ["B8","B9","B10"]

    per_blocks = {n: data_io.to_blocks(data_io.load_board_csv(p), 1000) for n, p in boards.items()}
    n_blocks = next(iter(per_blocks.values())).shape[0]
    sp = splits.make_block_splits(n_blocks, seed=123)

    Xp_tr = data_io.materialize_blocks(per_blocks[target], sp.train)
    Xp_va = data_io.materialize_blocks(per_blocks[target], sp.val)
    Xp_te = data_io.materialize_blocks(per_blocks[target], sp.test)

    def stack(names, which):
        return np.row_stack([data_io.materialize_blocks(per_blocks[n], getattr(sp, which)) for n in names])

    Xn_tr, Xn_va, Xn_te = stack(seen, "train"), stack(seen, "val"), stack(unseen, "test")

    Xtr = np.row_stack([Xp_tr, Xn_tr]); ytr = np.concatenate([np.ones(len(Xp_tr), int), np.zeros(len(Xn_tr), int)])
    Xva = np.row_stack([Xp_va, Xn_va]); yva = np.concatenate([np.ones(len(Xp_va), int), np.zeros(len(Xn_va), int)])
    Xte = np.row_stack([Xp_te, Xn_te]); yte = np.concatenate([np.ones(len(Xp_te), int), np.zeros(len(Xn_te), int)])

    base = models_tabular.pipe_lr().fit(Xtr, ytr)
    cal = CalibratedClassifierCV(base, cv="prefit", method="sigmoid").fit(Xva, yva)

    pva = cal.predict_proba(Xva)[:, 1]
    pte = cal.predict_proba(Xte)[:, 1]

    # Choose frozen threshold: TPR>=99% & FPR<=1% on validation
    thrs = np.linspace(0, 1, 1001); tau = 0.5
    for t in thrs:
        tp = ((pva >= t) & (yva == 1)).sum()
        fn = ((pva <  t) & (yva == 1)).sum()
        fp = ((pva >= t) & (yva == 0)).sum()
        tn = ((pva <  t) & (yva == 0)).sum()
        tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
        if tpr >= 0.99 and fpr <= 0.01:
            tau = t; break

    print("Validation:", metrics.verification(yva, pva))
    print("Unseen Test:", metrics.verification(yte, pte))
    tpr_te = ((pte >= tau) & (yte == 1)).sum() / max(1, (yte == 1).sum())
    fpr_te = ((pte >= tau) & (yte == 0)).sum() / max(1, (yte == 0).sum())
    print(f"Frozen tau={tau:.3f} -> Test TPR={tpr_te:.3f}, FPR={fpr_te:.3f}")

if __name__ == "__main__":
    main()
