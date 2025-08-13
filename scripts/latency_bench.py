from edgeprint import data_io, splits, models_tabular, latency, utils
import numpy as np, os, joblib

def main():
    utils.set_seed(42)
    boards = {"B1": "data/boards/B1.csv", "B2": "data/boards/B2.csv"}
    per_blocks = {n: data_io.to_blocks(data_io.load_board_csv(p), 1000) for n, p in boards.items()}
    sp = splits.make_block_splits(next(iter(per_blocks.values())).shape[0], seed=42)

    Xtr = np.row_stack([
        data_io.materialize_blocks(per_blocks["B1"], sp.train),
        data_io.materialize_blocks(per_blocks["B2"], sp.train),
    ])
    ytr = np.concatenate([
        np.ones(len(data_io.materialize_blocks(per_blocks["B1"], sp.train)), int),
        np.zeros(len(data_io.materialize_blocks(per_blocks["B2"], sp.train)), int),
    ])

    clf = models_tabular.pipe_lr().fit(Xtr, ytr)
    x1 = Xtr[:1]
    print(latency.time_predict_proba(clf, x1))

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/lr.joblib")
    print("Model size MB:", os.path.getsize("models/lr.joblib") / 1e6)

if __name__ == "__main__":
    main()
