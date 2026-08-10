import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score


def compute_metrics(y_true: list, y_pred: list, class_names: list = None):
    """Computes accuracy, precision, recall, F1-score, and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    labels = list(range(len(class_names))) if class_names else None
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    save_path: str = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
):
    """Plots and saves the confusion matrix figure."""
    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    else:
        cm_norm = cm

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = ".2f" if normalize else "d"
    thresh = cm_norm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm_norm[i, j]
        plt.text(
            j,
            i,
            format(val, fmt),
            horizontalalignment="center",
            color="white" if val > thresh else "black",
            fontsize=9,
        )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
