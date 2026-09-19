"""Metrics computation, separated from plotting.

Ported from legacy/Project_codeFile.ipynb Cells 14/25 (evaluate_model /
get_metrics), split into a pure metrics function (importable, testable,
no side effects) and an optional plotting function (matplotlib import
is deferred inside the function so importing this module doesn't
require a display backend — needed for pytest/CI).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class Metrics:
    recall: float
    precision: float
    f1: float
    pr_auc: float
    roc_auc: float
    accuracy: float
    naive_baseline_accuracy: float
    true_positives: int
    false_negatives: int
    false_positives: int
    true_negatives: int


def compute_metrics(y_true, y_pred, y_prob) -> Metrics:
    """PR-AUC is the primary metric for this 449:1-imbalanced problem —
    see legacy/Project_codeFile.ipynb Cell 14's note on why accuracy is
    misleading here (a naive all-non-PHA classifier scores ~99.8%)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    report = classification_report(
        y_true, y_pred, output_dict=True, target_names=["Non-PHA", "PHA"]
    )
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)
    roc_auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    naive_acc = (y_true == 0).sum() / len(y_true)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return Metrics(
        recall=report["PHA"]["recall"],
        precision=report["PHA"]["precision"],
        f1=report["PHA"]["f1-score"],
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        accuracy=acc,
        naive_baseline_accuracy=naive_acc,
        true_positives=int(tp),
        false_negatives=int(fn),
        false_positives=int(fp),
        true_negatives=int(tn),
    )


def plot_curves(y_true, y_prob, title: str):
    """Optional — imports matplotlib lazily so this module stays
    importable in headless/CI environments that never call this."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = auc(recalls, precisions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title(f"ROC Curve — {title}")
    axes[0].legend()
    axes[1].plot(recalls, precisions, color="coral", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].set_title(f"Precision-Recall Curve — {title}")
    axes[1].legend()
    plt.tight_layout()
    return fig
