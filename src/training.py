"""
training.py — Stratified Training Dataset Construction (S10)

Converts the full labelled pixel set (from S9) into a balanced,
NaN-free training dataset suitable for scikit-learn classifiers.

Public API:
    build_training_set(X, y_full, seed)  → (X_train, y_train)
    training_summary(X_train, y_train)   → prints dataset statistics

No file I/O. No mutation of inputs. No geospatial logic.
"""

import numpy as np
from src.labels import (
    LABEL_INVALID,
    LABEL_OTHER,
    LABEL_VEGETATION,
    LABEL_BUILT_UP,
    get_valid_mask,
)

# All classes considered during training, in a fixed order.
# This order determines nothing about the model but is used for
# consistent reporting. Modify here if other is excluded.
_TRAIN_CLASSES = [LABEL_OTHER, LABEL_VEGETATION, LABEL_BUILT_UP]

_CLASS_NAMES = {
    LABEL_OTHER:      "other",
    LABEL_VEGETATION: "vegetation",
    LABEL_BUILT_UP:   "built-up",
}


# ---------------------------------------------------------------------------
# S10 — Training dataset construction
# ---------------------------------------------------------------------------

def build_training_set(
    X: np.ndarray,
    y_full: np.ndarray,
    seed: int = 42,
) -> tuple:
    """
    Construct a balanced, NaN-free training dataset via stratified sampling.

    Parameters
    ----------
    X : np.ndarray, shape (N_pixels, 3), float32
        Full feature matrix from stack_features. Columns: [red, nir, swir1].
        May contain NaN rows — these are filtered out here.

    y_full : np.ndarray, shape (N_pixels,), int32
        Full label array from generate_labels.
        Invalid pixels are encoded as -1.

    seed : int
        Random seed for reproducible sampling.
        Must be identical across all training runs for reproducibility.

    Returns
    -------
    X_train : np.ndarray, shape (K, 3), float32
              K = n_classes × N_per_class
    y_train : np.ndarray, shape (K,), int32

    Algorithm
    ---------
    Step 1 — Select labelled pixels
        Filter X and y_full to rows where y_full != -1.
        This removes:
            - cloud/shadow masked pixels  (NaN propagated → -1 in S9)
            - conflicting signal pixels   (-1 from conflict resolution)
        After this step: no NaNs in X_valid, every row has a label.

    Step 2 — Stratified sampling
        Count pixels per class.
        N = min(count_per_class) — the smallest class sets the budget.
        Randomly sample exactly N pixels from each class without replacement.
        This enforces a balanced class distribution and prevents the model
        from learning the natural (highly imbalanced) class prior.

    Step 3 — Shuffle and return
        Concatenate sampled rows.
        Shuffle the combined dataset (prevents class-block ordering from
        affecting gradient-based or batch-sensitive learners).

    Safety guarantees
    -----------------
    - Raises ValueError if any class has zero labelled pixels.
    - Raises ValueError if X contains NaN after the valid-mask filter
      (would indicate a bug in S9 label generation).
    - Raises ValueError if shapes of X and y_full are inconsistent.
    """
    # --- Input validation ---
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"X must be shape (N, 3), got {X.shape}")
    if y_full.ndim != 1:
        raise ValueError(f"y_full must be 1-D, got shape {y_full.shape}")
    if X.shape[0] != y_full.shape[0]:
        raise ValueError(
            f"X and y_full row count mismatch: {X.shape[0]} vs {y_full.shape[0]}"
        )

    # --- Step 1: Select labelled pixels ---
    valid_mask = get_valid_mask(y_full)
    X_valid = X[valid_mask]
    y_valid = y_full[valid_mask]

    # Confirm no NaNs survived the filter — this catches any S9 logic gap
    if np.any(np.isnan(X_valid)):
        n_nan_rows = int(np.any(np.isnan(X_valid), axis=1).sum())
        raise ValueError(
            f"{n_nan_rows} NaN-containing rows survived the valid mask. "
            "This indicates a bug in label generation (S9)."
        )

    # --- Step 2: Stratified sampling ---
    rng = np.random.default_rng(seed)

    class_indices = {}
    for cls in _TRAIN_CLASSES:
        idx = np.where(y_valid == cls)[0]
        if idx.size == 0:
            raise ValueError(
                f"Class '{_CLASS_NAMES[cls]}' (label={cls}) has zero labelled pixels. "
                "Adjust thresholds in labels.py or check preprocessing output."
            )
        class_indices[cls] = idx

    counts = {cls: idx.size for cls, idx in class_indices.items()}
    N_per_class = min(counts.values())

    sampled_indices = []
    for cls in _TRAIN_CLASSES:
        idx = class_indices[cls]
        chosen = rng.choice(idx, size=N_per_class, replace=False)
        sampled_indices.append(chosen)

    # --- Step 3: Concatenate and shuffle ---
    all_indices = np.concatenate(sampled_indices)
    shuffle_order = rng.permutation(len(all_indices))
    all_indices = all_indices[shuffle_order]

    X_train = X_valid[all_indices].astype(np.float32)
    y_train = y_valid[all_indices].astype(np.int32)

    # Final shape assertions
    expected_rows = len(_TRAIN_CLASSES) * N_per_class
    assert X_train.shape == (expected_rows, 3), (
        f"Unexpected X_train shape: {X_train.shape}"
    )
    assert y_train.shape == (expected_rows,), (
        f"Unexpected y_train shape: {y_train.shape}"
    )
    assert not np.any(np.isnan(X_train)), "NaN in X_train — sampling logic error"

    return X_train, y_train


def training_summary(X_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    Print a concise summary of the training dataset.

    Parameters
    ----------
    X_train : (K, 3) float32
    y_train : (K,) int32
    """
    col_names = ["red", "nir", "swir1"]
    n_total = X_train.shape[0]

    print("Training dataset summary")
    print(f"  Total samples : {n_total:,}")
    print(f"  Features      : {X_train.shape[1]}  {col_names}")
    print()

    print("  Class distribution:")
    for cls in _TRAIN_CLASSES:
        count = int(np.sum(y_train == cls))
        print(f"    {cls:>2}  {_CLASS_NAMES[cls]:<12}  {count:>8,}  ({100.0 * count / n_total:5.1f}%)")
    print()

    print("  Feature statistics (valid samples):")
    header = f"  {'feature':<8}  {'min':>8}  {'max':>8}  {'mean':>8}  {'std':>8}"
    print(header)
    for i, col in enumerate(col_names):
        col_data = X_train[:, i]
        print(
            f"  {col:<8}  "
            f"{col_data.min():>8.4f}  "
            f"{col_data.max():>8.4f}  "
            f"{col_data.mean():>8.4f}  "
            f"{col_data.std():>8.4f}"
        )
    print()