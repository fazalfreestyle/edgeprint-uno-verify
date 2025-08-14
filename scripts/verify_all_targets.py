# Leave-one-device-out verification across all boards.
# Full protocol: LR, RF, SVM, MLP (+ optional OC-SVM, Mahalanobis), calibration, fusion, cascade.
# Adds: per-target append, checkpoints, resume, progress timings, safe toggles.
#
# Usage (all targets):
#   python scripts/verify_all_targets.py
# Resume after a stop: just run the same command again.
# Optional: run subset
#   python scripts/verify_all_targets.py B9 B10
#
import os, sys, time, json, gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from edgeprint import data_io, splits, models_tabular, metrics as M, ensemble, oneclass, utils

# ------------------- CONFIG & TOGGLES (env overrides) -------------------
def _env_bool(name, default):
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1","true","t","yes","y")

BLOCK_SIZE     = int(os.getenv("EDGE_BLOCK_SIZE", "1000"))
TRAIN_FRAC     = float(os.getenv("EDGE_TRAIN_FRAC", "0.60"))
VAL_FRAC       = float(os.getenv("EDGE_VAL_FRAC",   "0.20"))

# Model toggles
USE_LR         = _env_bool("EDGE_USE_LR", True)
USE_RF         = _env_bool("EDGE_USE_RF", True)
USE_SVM        = _env_bool("EDGE_USE_SVM", True)
USE_MLP        = _env_bool("EDGE_USE_MLP", True)
USE_UNSUP      = _env_bool("EDGE_USE_UNSUP", True)   # OC-SVM + Mahalanobis

# Calibration methods (Platt/sigmoid is fast & robust; isotonic is heavier)
LR_CALIB       = os.getenv("EDGE_LR_CALIB", "sigmoid")     # 'sigmoid' | 'isotonic'
RF_CALIB       = os.getenv("EDGE_RF_CALIB", "sigmoid")     # was 'isotonic' originally
SVM_CALIB      = os.getenv("EDGE_SVM_CALIB", "sigmoid")
MLP_CALIB      = os.getenv("EDGE_MLP_CALIB", "sigmoid")

# Caps to keep steps from grinding; set to empty to disable (use all samples)
MAX_POS_TR     = os.getenv("EDGE_MAX_POS_TR", "")   # e.g. "60000"
MAX_NEG_TR     = os.getenv("EDGE_MAX_NEG_TR", "")
MAX_POS_VA     = os.getenv("EDGE_MAX_POS_VA", "")
MAX_NEG_VA     = os.getenv("EDGE_MAX_NEG_VA", "")

def _as_int_or_none(x): 
    return None if x == "" else int(x)

MAX_POS_TR = _as_int_or_none(MAX_POS_TR)
MAX_NEG_TR = _as_int_or_none(MAX_NEG_TR)
MAX_POS_VA = _as_int_or_none(MAX_POS_VA)
MAX_NEG_VA = _as_int_or_none(MAX_NEG_VA)

# OC-SVM can be slow; cap positives it sees
OCSVM_POS_CAP  = int(os.getenv("EDGE_OCSVM_POS_CAP", "30000"))

OUT_DIR        = os.path.abspath("runs")
OUT_CSV        = os.path.join(OUT_DIR, "verify_all_targets_summary.csv")
CKPT_DIR       = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
# ------------------------------------------------------------------------

BOARDS = [f"B{i}" for i in range(1, 11)]
only = set(sys.argv[1:])
if only:
    BOARDS = [b for b in BOARDS if b in only]

def cyclic_groups(ti, k_seen=6, k_unseen=3):
    seen = [BOARDS[(ti + i + 1) % len(BOARDS)] for i in range(k_seen)]
    unseen = [BOARDS[(ti + k_seen + i + 1) % len(BOARDS)] for i in range(k_unseen)]
    return seen, unseen

def cap_class(X, y, label, cap, rng):
    if cap is None:  # no cap
        idx = np.where(y == label)[0]
        return X[idx], np.full(len(idx), label, int)
    idx = np.where(y == label)[0]
    if len(idx) > cap:
        idx = rng.choice(idx, size=cap, replace=False)
    return X[idx], np.full(len(idx), label, int)

def choose_tau(p, y, tpr_min=0.99, fpr_max=0.01):
    th = np.linspace(0, 1, 1001)
    best = None
    for t in th:
        tp = ((p >= t) & (y == 1)).sum(); fn = ((p <  t) & (y == 1)).sum()
        fp = ((p >= t) & (y == 0)).sum(); tn = ((p <  t) & (y == 0)).sum()
        tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
        if tpr >= tpr_min and fpr <= fpr_max:
            best = (t, tpr, fpr); break
    if best: return best[0]
    # fallback: best TPR under fpr_max; else Youden J
    th_ok, best_tpr = None, -1
    for t in th:
        tp = ((p >= t) & (y == 1)).sum(); fn = ((p <  t) & (y == 1)).sum()
        fp = ((p >= t) & (y == 0)).sum(); tn = ((p <  t) & (y == 0)).sum()
        tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
        if fpr <= fpr_max and tpr > best_tpr: th_ok, best_tpr = t, tpr
    if th_ok is not None: return th_ok
    youden, tau = -1, 0.5
    for t in th:
        tp = ((p >= t) & (y == 1)).sum(); fn = ((p <  t) & (y == 1)).sum()
        fp = ((p >= t) & (y == 0)).sum(); tn = ((p <  t) & (y == 0)).sum()
        tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
        if (tpr - fpr) > youden: youden, tau = (tpr - fpr), t
    return tau

def timed(label, fn, *a, **k):
    t0 = time.time()
    res = fn(*a, **k)
    dt = time.time() - t0
    print(f"[{label}] {dt:.1f}s")
    return res, dt

def append_row(row: dict):
    """Append one row and flush to disk; create header if needed."""
    hdr = not os.path.exists(OUT_CSV)
    pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", index=False, header=hdr)
    # also write a small per-target JSON with timings for debugging
    tgt = row.get("target", "UNK")
    with open(os.path.join(CKPT_DIR, f"{tgt}.json"), "w") as f:
        json.dump(row, f, indent=2)
    open(os.path.join(CKPT_DIR, f"{tgt}.done"), "w").write("ok")

def already_done():
    done = set()
    # checkpoint files
    for b in BOARDS:
        if os.path.exists(os.path.join(CKPT_DIR, f"{b}.done")):
            done.add(b)
    # also read targets from OUT_CSV if present
    if os.path.exists(OUT_CSV):
        try:
            df = pd.read_csv(OUT_CSV, usecols=["target"])
            done.update(df["target"].dropna().astype(str).tolist())
        except Exception:
            pass
    return done

def main():
    utils.set_seed(42)

    # Load → blocks for each board
    per_blocks = {}
    for b in BOARDS:
        (X, _) = timed(f"load {b}", data_io.load_board_csv, f"data/boards/{b}.csv")
        (blk, _) = timed(f"to_blocks {b}", data_io.to_blocks, X, BLOCK_SIZE)
        per_blocks[b] = blk

    # Equalize number of blocks & make block splits
    n_blocks = int(min(b.shape[0] for b in per_blocks.values()))
    for k in per_blocks: per_blocks[k] = per_blocks[k][:n_blocks]
    sp = splits.make_block_splits(n_blocks, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=123)
    print(f"[split] blocks={n_blocks} train={len(sp.train)} val={len(sp.val)} test={len(sp.test)}")

    done = already_done()
    if done:
        print("Resuming; skipping:", sorted(done))

    rng = np.random.default_rng(42)

    for ti, tgt in enumerate(tqdm(BOARDS, desc="Targets")):
        if tgt in done:
            print(f"Skipping {tgt} (already done)")
            continue

        seen, unseen = cyclic_groups(ti, 6, 3)
        print(f"\n=== Target {tgt} | seen={seen} | unseen={unseen} ===")

        Xp_tr = data_io.materialize_blocks(per_blocks[tgt], sp.train)
        Xp_va = data_io.materialize_blocks(per_blocks[tgt], sp.val)
        Xp_te = data_io.materialize_blocks(per_blocks[tgt], sp.test)

        def stack(names, which):
            return np.row_stack([data_io.materialize_blocks(per_blocks[n], getattr(sp, which)) for n in names])

        Xn_tr, Xn_va, Xn_te = stack(seen, "train"), stack(seen, "val"), stack(unseen, "test")

        Xtr = np.row_stack([Xp_tr, Xn_tr]); ytr = np.concatenate([np.ones(len(Xp_tr), int), np.zeros(len(Xn_tr), int)])
        Xva = np.row_stack([Xp_va, Xn_va]); yva = np.concatenate([np.ones(len(Xp_va), int), np.zeros(len(Xn_va), int)])
        Xte = np.row_stack([Xp_te, Xn_te]); yte = np.concatenate([np.ones(len(Xp_te), int), np.zeros(len(Xn_te), int)])

        # Optional caps for speed-safety
        def maybe_cap(X, y, pos_cap, neg_cap):
            Xp_c, yp_c = cap_class(X, y, 1, pos_cap, rng)
            Xn_c, yn_c = cap_class(X, y, 0, neg_cap, rng)
            return np.row_stack([Xp_c, Xn_c]), np.concatenate([yp_c, yn_c])

        if any(v is not None for v in (MAX_POS_TR, MAX_NEG_TR)):
            Xtr, ytr = maybe_cap(Xtr, ytr, MAX_POS_TR, MAX_NEG_TR)
        if any(v is not None for v in (MAX_POS_VA, MAX_NEG_VA)):
            Xva, yva = maybe_cap(Xva, yva, MAX_POS_VA, MAX_NEG_VA)

        # ---------------- Train supervised models ----------------
        models = {}
        times = {}

        if USE_LR:
            (m, dt) = timed(f"{tgt} fit LR", models_tabular.pipe_lr().fit, Xtr, ytr)
            models["lr"] = m; times["fit_lr_s"] = dt
        if USE_RF:
            (m, dt) = timed(f"{tgt} fit RF", models_tabular.pipe_rf().fit, Xtr, ytr)
            models["rf"] = m; times["fit_rf_s"] = dt
        if USE_SVM:
            (m, dt) = timed(f"{tgt} fit SVM", models_tabular.pipe_svm_rbf().fit, Xtr, ytr)
            models["svm"] = m; times["fit_svm_s"] = dt
        if USE_MLP:
            (m, dt) = timed(f"{tgt} fit MLP", models_tabular.pipe_mlp().fit, Xtr, ytr)
            models["mlp"] = m; times["fit_mlp_s"] = dt

        # --------------- Calibrate on validation -----------------
        calibrated = {}
        if "lr" in models:
            (c, dt) = timed(f"{tgt} calibrate LR", CalibratedClassifierCV(models["lr"],  cv="prefit", method=LR_CALIB).fit, Xva, yva)
            calibrated["lr"] = c; times["cal_lr_s"] = dt
        if "rf" in models:
            (c, dt) = timed(f"{tgt} calibrate RF", CalibratedClassifierCV(models["rf"],  cv="prefit", method=RF_CALIB).fit, Xva, yva)
            calibrated["rf"] = c; times["cal_rf_s"] = dt
        if "svm" in models:
            (c, dt) = timed(f"{tgt} calibrate SVM", CalibratedClassifierCV(models["svm"], cv="prefit", method=SVM_CALIB).fit, Xva, yva)
            calibrated["svm"] = c; times["cal_svm_s"] = dt
        if "mlp" in models:
            (c, dt) = timed(f"{tgt} calibrate MLP", CalibratedClassifierCV(models["mlp"], cv="prefit", method=MLP_CALIB).fit, Xva, yva)
            calibrated["mlp"] = c; times["cal_mlp_s"] = dt

        # --------------- Probabilities (val/test) ----------------
        prob = lambda c, X: c.predict_proba(X)[:, 1]
        pva, pte = {}, {}
        for k, c in calibrated.items():
            pva[k] = prob(c, Xva)
            pte[k] = prob(c, Xte)

        # --------------- Unsupervised baselines (optional) -------
        if USE_UNSUP:
            # cap positives for OC-SVM
            idx = np.arange(len(Xp_tr))
            if len(idx) > OCSVM_POS_CAP:
                idx = np.random.default_rng(42).choice(idx, size=OCSVM_POS_CAP, replace=False)
            Xp_tr_cap = Xp_tr[idx]

            (oc, dt)   = timed(f"{tgt} fit OC-SVM", oneclass.oc_svm_rbf().fit, Xp_tr_cap)
            (maha, dt2)= timed(f"{tgt} fit Mahalanobis", oneclass.MahalanobisModel().fit, Xp_tr_cap)
            times["fit_ocsvm_s"] = dt; times["fit_maha_s"] = dt2

            def mm(a): lo, hi = a.min(), a.max(); return (a - lo) / (hi - lo + 1e-9)
            pva["ocsvm"] = mm(oc.decision_function(Xva));   pte["ocsvm"] = mm(oc.decision_function(Xte))
            pva["maha"]  = mm(maha.decision_function(Xva)); pte["maha"]  = mm(maha.decision_function(Xte))

        # --------------- Metrics (val/test) ----------------------
        row = {"target": tgt, "seen": ",".join(seen), "unseen": ",".join(unseen),
               "block_size": BLOCK_SIZE, "n_blocks": int(n_blocks)}
        mdl_list = list(pva.keys())  # whatever we actually computed
        for name in mdl_list:
            v = M.verification(yva, pva[name]); t = M.verification(yte, pte[name])
            for k, val in v.items(): row[f"val_{name}_{k}"]  = val
            for k, val in t.items(): row[f"test_{name}_{k}"] = val

        # --------------- Fusion (supervised only) ----------------
        sup_names = [m for m in ["lr","rf","svm","mlp"] if m in pva]
        if len(sup_names) >= 2:
            L = [log_loss(yva, np.c_[1 - pva[n], pva[n]]) for n in sup_names]
            w = ensemble.weights_from_logloss(L)
            fused_val = ensemble.soft_vote([np.c_[1 - pva[n], pva[n]] for n in sup_names], w)
            fused_tst = ensemble.soft_vote([np.c_[1 - pte[n], pte[n]] for n in sup_names], w)
            for k, val in M.verification(yva, fused_val).items(): row[f"val_fusion_{k}"]  = val
            for k, val in M.verification(yte, fused_tst).items(): row[f"test_fusion_{k}"] = val
            row["fusion_weights"] = ",".join(f"{n}:{w[i]:.3f}" for i, n in enumerate(sup_names))

        # --------------- Cascade: LR fast → RF slow ---------------
        if "lr" in pva and "rf" in pva:
            thrs = np.linspace(0, 1, 1001); tau1 = 0.5
            for t in thrs:
                tp = ((pva["lr"] >= t) & (yva == 1)).sum(); fn = ((pva["lr"] <  t) & (yva == 1)).sum()
                fp = ((pva["lr"] >= t) & (yva == 0)).sum(); tn = ((pva["lr"] <  t) & (yva == 0)).sum()
                tpr = tp / max(1, (tp + fn)); fpr = fp / max(1, (fp + tn))
                if tpr >= 0.99 and fpr <= 0.01: tau1 = t; break
            casc_val = 0.5 * pva["lr"] + 0.5 * pva["rf"]; mask_val = pva["lr"] < tau1
            casc_val[~mask_val] = pva["lr"][~mask_val]
            casc_tst = 0.5 * pte["lr"] + 0.5 * pte["rf"]; mask_tst = pte["lr"] < tau1
            casc_tst[~mask_tst] = pte["lr"][~mask_tst]
            for k, val in M.verification(yva, casc_val).items(): row[f"val_cascade_{k}"]  = val
            for k, val in M.verification(yte, casc_tst).items(): row[f"test_cascade_{k}"] = val
            row["cascade_slowpath_val"]  = float(mask_val.mean())
            row["cascade_slowpath_test"] = float(mask_tst.mean())
            row["cascade_tau1"] = float(tau1)

        # timings
        for k, v in list(locals().items()):
            pass
        row.update(times)

        # Append & checkpoint
        append_row(row)
        print(f"Done {tgt} → wrote row to {OUT_CSV}")

        # free memory between targets
        del Xp_tr, Xp_va, Xp_te, Xn_tr, Xn_va, Xn_te, Xtr, Xva, Xte, ytr, yva, yte
        del models, calibrated, pva, pte
        gc.collect()

    print("\nAll targets processed. Summary:", OUT_CSV)

if __name__ == "__main__":
    main()
