from edgeprint import data_io, splits, models_tabular, metrics, ensemble, utils
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import numpy as np

def main():
    utils.set_seed(42)
    boards = {f"B{i}": f"data/boards/B{i}.csv" for i in range(1, 11)}
    target, seen, unseen = "B1", ["B2","B3","B4","B5","B6","B7"], ["B8","B9","B10"]

    per_blocks = {n: data_io.to_blocks(data_io.load_board_csv(p), 1000) for n, p in boards.items()}
    sp = splits.make_block_splits(next(iter(per_blocks.values())).shape[0], seed=123)

    Xp_tr = data_io.materialize_blocks(per_blocks[target], sp.train)
    Xp_va = data_io.materialize_blocks(per_blocks[target], sp.val)
    Xp_te = data_io.materialize_blocks(per_blocks[target], sp.test)
    def stack(names, which): return np.row_stack([data_io.materialize_blocks(per_blocks[n], getattr(sp, which)) for n in names])
    Xn_tr, Xn_va, Xn_te = stack(seen,"train"), stack(seen,"val"), stack(unseen,"test")

    Xtr = np.row_stack([Xp_tr, Xn_tr]); ytr = np.concatenate([np.ones(len(Xp_tr), int), np.zeros(len(Xn_tr), int)])
    Xva = np.row_stack([Xp_va, Xn_va]); yva = np.concatenate([np.ones(len(Xp_va), int), np.zeros(len(Xn_va), int)])
    Xte = np.row_stack([Xp_te, Xn_te]); yte = np.concatenate([np.ones(len(Xp_te), int), np.zeros(len(Xn_te), int)])

    lr = models_tabular.pipe_lr().fit(Xtr, ytr)
    rf = models_tabular.pipe_rf().fit(Xtr, ytr)

    clr = CalibratedClassifierCV(lr, cv="prefit", method="sigmoid").fit(Xva, yva)
    crf = CalibratedClassifierCV(rf, cv="prefit", method="isotonic").fit(Xva, yva)

    pva_lr = clr.predict_proba(Xva)[:,1]; pte_lr = clr.predict_proba(Xte)[:,1]
    pva_rf = crf.predict_proba(Xva)[:,1]; pte_rf = crf.predict_proba(Xte)[:,1]

    # Soft vote
    w = ensemble.weights_from_logloss([
        log_loss(yva, np.c_[1-pva_lr, pva_lr]),
        log_loss(yva, np.c_[1-pva_rf, pva_rf]),
    ])
    fused_val = ensemble.soft_vote([np.c_[1-pva_lr,pva_lr], np.c_[1-pva_rf,pva_rf]], w)
    fused_tst = ensemble.soft_vote([np.c_[1-pte_lr,pte_lr], np.c_[1-pte_rf,pte_rf]], w)
    print("Fusion (val):", metrics.verification(yva, fused_val))
    print("Fusion (test):", metrics.verification(yte, fused_tst))

    # Cascade (LR fast → RF)
    thrs = np.linspace(0,1,1001); tau1 = 0.5
    for t in thrs:
        tp = ((pva_lr >= t) & (yva == 1)).sum(); fn = ((pva_lr < t) & (yva == 1)).sum()
        fp = ((pva_lr >= t) & (yva == 0)).sum(); tn = ((pva_lr < t) & (yva == 0)).sum()
        tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
        if tpr >= 0.99 and fpr <= 0.01: tau1 = t; break
    casc_val, slow_val = ensemble.cascade_probs(pva_lr, pva_rf, tau1)
    casc_tst, slow_tst = ensemble.cascade_probs(pte_lr, pte_rf, tau1)
    print("Cascade (val):", metrics.verification(yva, casc_val))
    print("Cascade (test):", metrics.verification(yte, casc_tst))
    print(f"Slow-path rate (val,test): {slow_val.mean():.3f}, {slow_tst.mean():.3f}")

if __name__ == "__main__":
    main()
