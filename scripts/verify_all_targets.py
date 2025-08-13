from edgeprint import data_io, splits, models_tabular, oneclass, metrics, ensemble, utils
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import numpy as np, os, pandas as pd

BOARDS = [f"B{i}" for i in range(1, 11)]

def cyclic_groups(ti, k_seen=6, k_unseen=3):
    seen = [BOARDS[(ti + i + 1) % len(BOARDS)] for i in range(k_seen)]
    unseen = [BOARDS[(ti + k_seen + i + 1) % len(BOARDS)] for i in range(k_unseen)]
    return seen, unseen

def main():
    utils.set_seed(42)
    per_blocks = {b: data_io.to_blocks(data_io.load_board_csv(f"data/boards/{b}.csv"), 1000) for b in BOARDS}
    sp = splits.make_block_splits(next(iter(per_blocks.values())).shape[0], seed=123)

    rows = []
    os.makedirs("runs", exist_ok=True)

    for ti, tgt in enumerate(BOARDS):
        seen, unseen = cyclic_groups(ti, 6, 3)

        Xp_tr = data_io.materialize_blocks(per_blocks[tgt], sp.train)
        Xp_va = data_io.materialize_blocks(per_blocks[tgt], sp.val)
        Xp_te = data_io.materialize_blocks(per_blocks[tgt], sp.test)

        def stack(names, which):
            return np.row_stack([data_io.materialize_blocks(per_blocks[n], getattr(sp, which)) for n in names])

        Xn_tr, Xn_va, Xn_te = stack(seen, "train"), stack(seen, "val"), stack(unseen, "test")

        Xtr = np.row_stack([Xp_tr, Xn_tr]); ytr = np.concatenate([np.ones(len(Xp_tr), int), np.zeros(len(Xn_tr), int)])
        Xva = np.row_stack([Xp_va, Xn_va]); yva = np.concatenate([np.ones(len(Xp_va), int), np.zeros(len(Xn_va), int)])
        Xte = np.row_stack([Xp_te, Xn_te]); yte = np.concatenate([np.ones(len(Xp_te), int), np.zeros(len(Xn_te), int)])

        # Train base models
        lr  = models_tabular.pipe_lr().fit(Xtr, ytr)
        rf  = models_tabular.pipe_rf().fit(Xtr, ytr)
        svm = models_tabular.pipe_svm_rbf().fit(Xtr, ytr)
        mlp = models_tabular.pipe_mlp().fit(Xtr, ytr)

        # Calibrate
        clr  = CalibratedClassifierCV(lr,  cv="prefit", method="sigmoid").fit(Xva, yva)
        crf  = CalibratedClassifierCV(rf,  cv="prefit", method="isotonic").fit(Xva, yva)
        csvm = CalibratedClassifierCV(svm, cv="prefit", method="sigmoid").fit(Xva, yva)
        cmlp = CalibratedClassifierCV(mlp, cv="prefit", method="sigmoid").fit(Xva, yva)

        def prob(c, X): return c.predict_proba(X)[:, 1]
        pva = {"lr": prob(clr, Xva), "rf": prob(crf, Xva), "svm": prob(csvm, Xva), "mlp": prob(cmlp, Xva)}
        pte = {"lr": prob(clr, Xte), "rf": prob(crf, Xte), "svm": prob(csvm, Xte), "mlp": prob(cmlp, Xte)}

        # One-class baselines (score → min-max to pseudo-prob on val)
        oc = oneclass.oc_svm_rbf().fit(Xp_tr)
        maha = oneclass.MahalanobisModel().fit(Xp_tr)
        def mm(a): lo, hi = a.min(), a.max(); return (a - lo) / (hi - lo + 1e-9)
        pva["ocsvm"] = mm(oc.decision_function(Xva));   pte["ocsvm"] = mm(oc.decision_function(Xte))
        pva["maha"]  = mm(maha.decision_function(Xva)); pte["maha"]  = mm(maha.decision_function(Xte))

        row = {"target": tgt, "seen": ",".join(seen), "unseen": ",".join(unseen)}
        for name in ["lr","rf","svm","mlp","ocsvm","maha"]:
            for k, v in metrics.verification(yva, pva[name]).items(): row[f"val_{name}_{k}"]  = v
            for k, v in metrics.verification(yte, pte[name]).items(): row[f"test_{name}_{k}"] = v

        # Fusion (weights from validation log-loss, binary models only)
        ll = [log_loss(yva, np.c_[1 - pva[n], pva[n]]) for n in ["lr","rf","svm","mlp"]]
        w = ensemble.weights_from_logloss(ll)
        fused_val = ensemble.soft_vote([np.c_[1 - pva[n], pva[n]] for n in ["lr","rf","svm","mlp"]], w)
        fused_tst = ensemble.soft_vote([np.c_[1 - pte[n], pte[n]] for n in ["lr","rf","svm","mlp"]], w)
        for k, v in metrics.verification(yva, fused_val).items(): row[f"val_fusion_{k}"]  = v
        for k, v in metrics.verification(yte, fused_tst).items(): row[f"test_fusion_{k}"] = v

        # Cascade: LR fast → RF
        thrs = np.linspace(0, 1, 1001); tau1 = 0.5
        for t in thrs:
            tp = ((pva["lr"] >= t) & (yva == 1)).sum()
            fn = ((pva["lr"] <  t) & (yva == 1)).sum()
            fp = ((pva["lr"] >= t) & (yva == 0)).sum()
            tn = ((pva["lr"] <  t) & (yva == 0)).sum()
            tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
            if tpr >= 0.99 and fpr <= 0.01: tau1 = t; break
        casc_val, slow_val = ensemble.cascade_probs(pva["lr"], pva["rf"], tau1)
        casc_tst, slow_tst = ensemble.cascade_probs(pte["lr"], pte["rf"], tau1)
        for k, v in metrics.verification(yva, casc_val).items(): row[f"val_cascade_{k}"]  = v
        for k, v in metrics.verification(yte, casc_tst).items(): row[f"test_cascade_{k}"] = v
        row["cascade_slowpath_val"]  = float(slow_val.mean())
        row["cascade_slowpath_test"] = float(slow_tst.mean())

        rows.append(row)
        print(f"Done {tgt}: unseen={unseen}")

    pd.DataFrame(rows).to_csv("runs/verify_all_targets_summary.csv", index=False)
    print("Wrote runs/verify_all_targets_summary.csv")

if __name__ == "__main__":
    main()
