"""
model.py — Model Training (S11) and Full-Raster Prediction (S12)

Public API:
    train_model(X_train, y_train, config)        → (model, metrics)
    predict_full(model, X_full, shape)  → (class_map (H,W), proba_map (H,W,3))

No file I/O. No mutation of inputs.
Feature column order must match stack_features exactly: [red, nir, swir1]
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import RF_PARAMS, RANDOM_SEED, VAL_SPLIT, FEATURE_NAMES
from src.evaluation import compute_metrics, print_metrics, print_feature_importances


# ---------------------------------------------------------------------------
# S11 — Model Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rf_params: dict = None,
    val_split: float = None,
    seed: int = None,
) -> tuple:
    """
    Fit a RandomForestClassifier on a balanced, NaN-free training set.

    Parameters
    ----------
    X_train   : (K, 3) float32 — feature matrix from build_training_set
    y_train   : (K,) int32     — class labels {0, 1, 2}
    rf_params : dict, optional — RF hyperparameters; defaults to config.RF_PARAMS
    val_split : float, optional — validation fraction; defaults to config.VAL_SPLIT
    seed      : int, optional   — random seed; defaults to config.RANDOM_SEED

    Returns
    -------
    model   : fitted RandomForestClassifier (trained on training split only)
    metrics : dict from evaluation.compute_metrics (on validation split)

    Training protocol
    -----------------
    1. Stratified train/val split preserves class balance in both halves.
    2. Model is fitted on the training split only.
    3. Metrics are computed on the held-out validation split.
    4. The returned model has seen no validation data — it is safe for S12.

    No NaN check is performed here because build_training_set (S10) guarantees
    NaN-free input. A guard is included defensively.
    """
    # --- Resolve defaults ---
    rf_params = rf_params if rf_params is not None else RF_PARAMS
    val_split = val_split if val_split is not None else VAL_SPLIT
    seed      = seed      if seed      is not None else RANDOM_SEED

    # --- Input validation ---
    if X_train.ndim != 2 or X_train.shape[1] != 3:
        raise ValueError(f"X_train must be shape (N, 3), got {X_train.shape}")
    if y_train.ndim != 1 or X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Shape mismatch: X_train={X_train.shape}, y_train={y_train.shape}"
        )
    if np.any(np.isnan(X_train)):
        raise ValueError(
            "NaN values found in X_train. S10 should have removed them. "
            "Check build_training_set output."
        )

    # Feature order guard — log column mapping so any future reordering is
    # immediately visible in output
    print(f"  Feature column mapping: {list(enumerate(FEATURE_NAMES))}")

    # --- Stratified split ---
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_split,
        stratify=y_train,
        random_state=seed,
    )

    print(f"  Train split : {X_tr.shape[0]:,} samples")
    print(f"  Val split   : {X_val.shape[0]:,} samples")

    # Verify balance is preserved in both splits
    for split_name, y_split in [("train", y_tr), ("val", y_val)]:
        counts = {int(c): int(np.sum(y_split == c)) for c in np.unique(y_split)}
        print(f"  {split_name} class counts: {counts}")

    # --- Fit model on training split only ---
    print(f"\n  Fitting RandomForest (n_estimators={rf_params['n_estimators']}, "
          f"max_depth={rf_params['max_depth']}, "
          f"min_samples_leaf={rf_params['min_samples_leaf']})...")

    model = RandomForestClassifier(**rf_params)
    model.fit(X_tr, y_tr)

    # --- Evaluate on validation split ---
    y_pred_val = model.predict(X_val)
    metrics    = compute_metrics(y_val, y_pred_val)

    return model, metrics


# ---------------------------------------------------------------------------
# S12 — Full-Raster Prediction and Reconstruction
# ---------------------------------------------------------------------------

def predict_full(
    model,
    X_full: np.ndarray,
    shape: tuple,
) -> tuple:
    """
    Run inference on a full feature matrix and reconstruct classified rasters.

    Returns both a hard class map and a per-class probability map, enabling
    confidence-aware change detection in S13.

    Parameters
    ----------
    model  : fitted RandomForestClassifier from train_model
    X_full : (N, 3) float32 — full feature matrix from stack_features
             Columns: [red, nir, swir1]. May contain NaN rows.
    shape  : (H, W) — original raster shape (must satisfy H * W == N)

    Returns
    -------
    class_map : np.ndarray of shape (H, W), dtype int32
        -1 → invalid (NaN pixel or outside AOI)
         0 → other
         1 → vegetation
         2 → built-up

    proba_map : np.ndarray of shape (H, W, 3), dtype float32
        proba_map[:, :, i] = probability of class i ∈ {other, vegetation, built-up}
        Invalid pixels are filled with np.nan across all 3 channels.
        Column order matches model.classes_ which is guaranteed to be [0, 1, 2]
        after training on a balanced set containing all three classes.

    Implementation notes
    --------------------
    NaN handling:
        Rows with any NaN are excluded from both predict and predict_proba.
        class_map invalid positions → -1
        proba_map invalid positions → np.nan (preserves float semantics for S13)

    Pixel ordering:
        X_full was produced by stack_features using row-major (C) flattening.
        Reconstruction uses the same order. This invariant must never change.

    Inference independence:
        Called independently per year. No shared state between calls.
    """
    H, W = shape
    N    = H * W

    if X_full.shape != (N, 3):
        raise ValueError(
            f"X_full shape {X_full.shape} is inconsistent with raster shape {shape}. "
            f"Expected ({N}, 3). Check that stack_features uses [red, nir, swir1]."
        )

    # --- Identify valid (non-NaN) rows ---
    nan_rows   = np.any(np.isnan(X_full), axis=1)
    valid_mask = ~nan_rows

    n_valid   = int(valid_mask.sum())
    n_invalid = int(nan_rows.sum())
    print(f"  Pixels: {N:,} total | {n_valid:,} valid | {n_invalid:,} invalid (NaN/masked)")

    if n_valid == 0:
        raise RuntimeError("No valid pixels in X_full — check preprocessing output.")

    X_valid = X_full[valid_mask]   # (n_valid, 3) — NaN-free slice for sklearn

    # --- Hard class predictions ---
    y_pred_valid = model.predict(X_valid).astype(np.int32)

    y_full = np.full(N, fill_value=-1, dtype=np.int32)
    y_full[valid_mask] = y_pred_valid
    class_map = y_full.reshape(H, W)

    # --- Class probability predictions ---
    # predict_proba returns (n_valid, n_classes) in the order of model.classes_.
    # After training on a balanced set with all three classes present, this is
    # guaranteed to be [0, 1, 2]. We assert this to catch any unexpected ordering.
    n_classes = len(model.classes_)
    assert list(model.classes_) == [0, 1, 2], (
        f"Unexpected model class order: {model.classes_}. "
        "Expected [0, 1, 2] = [other, vegetation, built-up]."
    )

    proba_valid = model.predict_proba(X_valid).astype(np.float32)   # (n_valid, 3)

    # Initialise full proba array with NaN so invalid pixels propagate correctly
    proba_full = np.full((N, n_classes), fill_value=np.nan, dtype=np.float32)
    proba_full[valid_mask] = proba_valid
    proba_map = proba_full.reshape(H, W, n_classes)   # (H, W, 3)

    return class_map, proba_map
