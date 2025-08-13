from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def pipe_lr() -> Pipeline:
    return Pipeline([
        ("sc", RobustScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

def pipe_rf(n_estimators: int = 300, max_depth=None) -> Pipeline:
    return Pipeline([
        ("sc", RobustScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42))
    ])

def pipe_svm_rbf(C: float = 1.0, gamma: str = "scale") -> Pipeline:
    return Pipeline([
        ("sc", RobustScaler()),
        ("clf", SVC(C=C, gamma=gamma, probability=True,
                    class_weight="balanced", random_state=42))
    ])

def pipe_mlp(hidden=(64,), alpha=1e-4) -> Pipeline:
    return Pipeline([
        ("sc", RobustScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                              alpha=alpha, early_stopping=True,
                              n_iter_no_change=10, random_state=42, max_iter=200))
    ])
