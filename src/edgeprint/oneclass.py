import numpy as np
from sklearn.svm import OneClassSVM

def oc_svm_rbf(nu: float = 0.05, gamma: str = "scale") -> OneClassSVM:
    return OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)

class MahalanobisModel:
    """Gaussian model on inliers; decision_function returns higher for inliers."""
    def __init__(self, eps: float = 1e-6):
        self.mu_ = None
        self.icov_ = None
        self.eps = eps

    def fit(self, X: np.ndarray):
        self.mu_ = X.mean(axis=0)
        cov = np.cov(X.T) + self.eps * np.eye(X.shape[1])
        self.icov_ = np.linalg.pinv(cov)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        d = X - self.mu_
        m2 = np.einsum("ij,jk,ik->i", d, self.icov_, d)
        return -np.sqrt(m2 + 1e-12)
