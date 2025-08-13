from edgeprint import data_io, splits, models_tabular, metrics, utils
import numpy as np

def main():
    utils.set_seed(42)
    boards = {f"B{i}": f"data/boards/B{i}.csv" for i in range(1, 11)}  # adjust if fewer
    per_blocks = {n: data_io.to_blocks(data_io.load_board_csv(p), 1000) for n, p in boards.items()}
    n_blocks = next(iter(per_blocks.values())).shape[0]
    sp = splits.make_block_splits(n_blocks, seed=42)

    Xtr, ytr, Xte, yte = [], [], [], []
    names = list(boards.keys())
    for bi, name in enumerate(names):
        Xtr.append(data_io.materialize_blocks(per_blocks[name], sp.train))
        ytr.append(np.full(len(Xtr[-1]), bi))
        Xte.append(data_io.materialize_blocks(per_blocks[name], sp.test))
        yte.append(np.full(len(Xte[-1]), bi))

    Xtr = np.row_stack(Xtr); ytr = np.concatenate(ytr)
    Xte = np.row_stack(Xte); yte = np.concatenate(yte)

    clf = models_tabular.pipe_lr().fit(Xtr, ytr)  # try pipe_rf()/pipe_svm_rbf()/pipe_mlp() too
    y_pred = clf.predict(Xte)
    print(metrics.identification(yte, y_pred))

if __name__ == "__main__":
    main()
