import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, RocCurveDisplay

def plot_roc(y_true, scores, title="ROC"):
    fpr, tpr, _ = roc_curve(y_true, scores)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title(title); plt.grid(True); plt.show()
