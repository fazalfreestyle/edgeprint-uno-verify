from edgeprint import data_io, splits, models_tabular, metrics as M, utils
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np, os, json, pandas as pd
import matplotlib.pyplot as plt

def choose_tau(p, y, tpr_min=0.99, fpr_max=0.01):
    # Grid search on probabilities for a stable cutoff
    th = np.linspace(0, 1, 1001)
    best = None
    for t in th:
        tp = ((p >= t) & (y == 1)).sum(); fn = ((p < t) & (y == 1)).sum()
        fp = ((p >= t) & (y == 0)).sum(); tn = ((p < t) & (y == 0)).sum()
        tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
        if tpr >= tpr_min and fpr <= fpr_max:
            best = (t, tpr, fpr); break
    if best is not None:
        return best[0]
    # Fallback: best TPR among thresholds with FPR <= fpr_max; else Youden J
    best_idx = None; best_tpr = -1
    for t in th:
        tp = ((p >= t) & (y == 1)).sum(); fn = ((p < t) & (y == 1)).sum()
        fp = ((p >= t) & (y == 0)).sum(); tn = ((p < t) & (y == 0)).sum()
        tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
        if fpr <= fpr_max and tpr > best_tpr:
            best_tpr = tpr; best_idx = t
    if best_idx is not None:
        return best_idx
    # Last resort: maximize (TPR - FPR)
    youden = -1; tau = 0.5
    for t in th:
        tp = ((p >= t) & (y == 1)).sum(); fn = ((p < t) & (y == 1)).sum()
        fp = ((p >= t) & (y == 0)).sum(); tn = ((p < t) & (y == 0)).sum()
        tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
        if (tpr - fpr) > youden: youden = (tpr - fpr); tau = t
    return tau

def main():
    utils.set_seed(42)
    boards = {f"B{i}": f"data/boards/B{i}.csv" for i in range(1, 11)}
    target = "B1"
    seen = ["B2","B3","B4","B5","B6","B7"]
    unseen = ["B8","B9","B10"]

    # Load
    per_blocks = {n: data_io.to_blocks(data_io.load_board_csv(p), 1000) for n, p in boards.items()}

    # Leak-proof: trim to common number of blocks across all boards
    n_blocks = min(b.shape[0] for b in per_blocks.values())
    for k in per_blocks:
        per_blocks[k] = per_blocks[k][:n_blocks]

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

    # Metrics
    val_metrics = M.verification(yva, pva)
    tst_metrics = M.verification(yte, pte)

    # Frozen threshold from validation
    tau = choose_tau(pva, yva, tpr_min=0.99, fpr_max=0.01)
    tpr_te = ((pte >= tau) & (yte == 1)).sum() / max(1, (yte == 1).sum())
    fpr_te = ((pte >= tau) & (yte == 0)).sum() / max(1, (yte == 0).sum())

    print("Validation:", val_metrics)
    print("Unseen Test:", tst_metrics)
    print(f"Frozen tau={tau:.3f} -> Test TPR={tpr_te:.3f}, FPR={fpr_te:.3f}")

    # Save outputs
    os.makedirs("runs", exist_ok=True)
    out = {
        "target": target, "seen": seen, "unseen": unseen,
        "val": val_metrics, "test": tst_metrics,
        "frozen_tau": float(tau), "test_tpr_at_tau": float(tpr_te), "test_fpr_at_tau": float(fpr_te),
        "block_size": 1000, "n_blocks": int(n_blocks)
    }
    with open("runs/verify_B1_lr.json", "w") as f:
        json.dump(out, f, indent=2)

    # --- Curves (test set) ---
    from sklearn.metrics import roc_curve, precision_recall_curve

    # ROC
    fpr, tpr, thr = roc_curve(yte, pte)
    thr = np.asarray(thr, dtype=float)

    # Pad thresholds to match fpr/tpr length
    if thr.size == 0:
        thr_full = np.array([0.5], dtype=float)
    elif thr.size == fpr.size - 1:
        thr_full = np.r_[thr, thr[-1]]
    else:
        # Fallback: clip or pad to ensure equal length
        thr_full = thr[: fpr.size]
        if thr_full.size < fpr.size:
            thr_full = np.r_[thr_full, np.repeat(thr_full[-1], fpr.size - thr_full.size)]

    assert fpr.size == tpr.size == thr_full.size, (fpr.size, tpr.size, thr_full.size)

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thr": thr_full})
    roc_df.to_csv("runs/verify_B1_lr_roc.csv", index=False)

    # PR (precision has same length as recall; thresholds is shorter)
    prec, rec, thr_pr = precision_recall_curve(yte, pte)
    pr_df = pd.DataFrame({"recall": rec, "precision": prec})
    pr_df.to_csv("runs/verify_B1_lr_pr.csv", index=False)

    # --- Quick plots ---
    import matplotlib.pyplot as plt
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (B1 vs unseen)"); plt.grid(True)
    plt.savefig("runs/verify_B1_lr_roc.png", dpi=200); plt.close()

    plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (B1 vs unseen)"); plt.grid(True)
    plt.savefig("runs/verify_B1_lr_pr.png", dpi=200); plt.close()

if __name__ == "__main__":
    main()
