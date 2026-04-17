"""
evaluation.py — Classification Metrics and Reporting (S11)

Computes and formats evaluation metrics for the trained classifier.
All functions are pure: they take arrays, return dicts or print output.

Public API:
    compute_metrics(y_true, y_pred)  → metrics dict
    print_metrics(metrics)           → formatted console report
    print_feature_importances(model) → ranked feature importance table
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.config import CLASS_NAMES, TRAIN_CLASSES, FEATURE_NAMES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute overall and per-class classification metrics.

    Parameters
    ----------
    y_true : (N,) int array — ground truth labels from validation split
    y_pred : (N,) int array — model predictions

    Returns
    -------
    metrics : dict with keys:
        "accuracy"          float
        "per_class"         dict[int → {"precision", "recall", "f1", "support"}]
        "confusion_matrix"  np.ndarray (n_classes × n_classes)
        "labels"            list of int class labels (row/col order of conf matrix)
    """
    labels = TRAIN_CLASSES   # [0, 1, 2] — excludes -1

    accuracy = float(accuracy_score(y_true, y_pred))

    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall    = recall_score(   y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1        = f1_score(       y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class = {}
    for i, cls in enumerate(labels):
        support = int(np.sum(y_true == cls))
        per_class[cls] = {
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1":        float(f1[i]),
            "support":   support,
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy":         accuracy,
        "per_class":        per_class,
        "confusion_matrix": cm,
        "labels":           labels,
    }


def print_metrics(metrics: dict, tag: str = "Validation") -> None:
    """
    Print a structured evaluation report to stdout.

    Parameters
    ----------
    metrics : dict returned by compute_metrics
    tag     : string label for the report header (e.g. "Validation", "Train")
    """
    print(f"\n{'=' * 60}")
    print(f"  Evaluation report — {tag}")
    print(f"{'=' * 60}")
    print(f"  Overall accuracy : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print()

    # Per-class table
    header = f"  {'class':<14}  {'precision':>10}  {'recall':>8}  {'f1':>8}  {'support':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cls in metrics["labels"]:
        m = metrics["per_class"][cls]
        name = CLASS_NAMES.get(cls, str(cls))
        print(
            f"  {name:<14}  "
            f"{m['precision']:>10.4f}  "
            f"{m['recall']:>8.4f}  "
            f"{m['f1']:>8.4f}  "
            f"{m['support']:>10,}"
        )
    print()

    # Confusion matrix
    cm     = metrics["confusion_matrix"]
    labels = metrics["labels"]
    col_w  = 12

    print("  Confusion matrix (rows = true, cols = predicted):")
    header_row = "  " + " " * 14 + "".join(
        CLASS_NAMES.get(c, str(c)).center(col_w) for c in labels
    )
    print(header_row)
    print("  " + "-" * (len(header_row) - 2))
    for i, cls in enumerate(labels):
        row_label = CLASS_NAMES.get(cls, str(cls)).ljust(14)
        row_vals  = "".join(f"{cm[i, j]:>{col_w},}" for j in range(len(labels)))
        print(f"  {row_label}{row_vals}")
    print()


def print_feature_importances(model, tag: str = "") -> None:
    """
    Print ranked feature importances from a fitted RandomForestClassifier.

    Parameters
    ----------
    model : fitted sklearn RandomForestClassifier
    tag   : optional label for the header
    """
    importances = model.feature_importances_
    ranked      = np.argsort(importances)[::-1]

    header = f"  Feature importances" + (f" — {tag}" if tag else "")
    print(header)
    print("  " + "-" * 40)
    for rank, idx in enumerate(ranked):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
        bar  = "█" * int(importances[idx] * 40)
        print(f"  {rank+1}. {name:<8}  {importances[idx]:.4f}  {bar}")
    print()
