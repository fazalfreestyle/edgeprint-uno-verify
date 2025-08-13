import time
import numpy as np

def time_predict_proba(model, X1, repeats: int = 2000, warmup: int = 100):
    """Measure per-sample latency for predict_proba on a 1-row batch."""
    for _ in range(warmup):
        _ = model.predict_proba(X1)
    t = []
    for _ in range(repeats):
        s = time.perf_counter()
        _ = model.predict_proba(X1)
        t.append(time.perf_counter() - s)
    arr = np.array(t, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean() * 1e3),
        "median_ms": float(np.median(arr) * 1e3),
        "p95_ms": float(np.percentile(arr, 95) * 1e3),
        "throughput_sps": float(1.0 / arr.mean() if arr.mean() > 0 else float("inf")),
    }
