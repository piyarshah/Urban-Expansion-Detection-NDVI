"""
change_detection.py — Confidence-Aware Change Detection (S13)

Detects land cover transitions between two classified raster maps using
both hard class labels and per-class probability outputs from the RF model.

Public API:
    compute_change_map(map_2013, map_2024, proba_2013, proba_2024, threshold)
        → change_map (H, W) int8
    change_map_summary(change_map, tag)
        → prints distribution to stdout

Change map encoding:
    -1 → invalid (masked, low-confidence, or outside AOI)
     0 → no change
     1 → vegetation → built-up
     2 → other → built-up

No file I/O. No mutation of inputs. Shape (H, W) preserved throughout.
"""

import numpy as np


def compute_change_map(
    map_2013: np.ndarray,
    map_2024: np.ndarray,
    proba_2013: np.ndarray,
    proba_2024: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Detect confident land cover transitions between 2013 and 2024.

    Parameters
    ----------
    map_2013   : (H, W) int32 — classified raster, values {-1, 0, 1, 2}
    map_2024   : (H, W) int32 — classified raster, values {-1, 0, 1, 2}
    proba_2013 : (H, W, 3) float32 — per-class probabilities for 2013
                 [:, :, 0] = P(other)  [:, :, 1] = P(veg)  [:, :, 2] = P(built-up)
                 NaN at invalid pixels.
    proba_2024 : (H, W, 3) float32 — same structure for 2024
    threshold  : float in [0, 1] — minimum max-class probability required in
                 BOTH years for a pixel to contribute to the change map.
                 Source: config.CONFIDENCE_THRESHOLD (default 0.60).

    Returns
    -------
    change_map : (H, W) int8
        -1 → invalid / low-confidence
         0 → no change
         1 → vegetation (1) → built-up (2)
         2 → other (0)      → built-up (2)

    Algorithm
    ---------
    Step 1 — Validity mask
        Pixels where either year returned -1 (cloud/shadow masked) are
        excluded. They cannot contribute meaningful transition information.

    Step 2 — Confidence mask
        max probability across the 3 classes = model confidence in its
        predicted class. Require >= threshold in BOTH years. Pixels near
        spectral class boundaries have low max probability; including them
        inflates false transitions.

    Step 3 — Transition logic
        No-change is assigned first. Specific urban growth transitions then
        override where applicable. Transitions are strictly to built-up (2)
        — the analysis objective is urban expansion, not general flux.

    Notes
    -----
    - Output dtype is int8: values {-1, 0, 1, 2} fit in int8 and halve
      memory vs int32 at (H, W) raster scale.
    - np.nanmax handles NaN in proba arrays without propagating to the
      confidence comparison. Invalid pixels are already removed by the
      validity mask before the confidence mask is applied.
    """
    H, W = map_2013.shape

    if map_2024.shape != (H, W):
        raise ValueError(
            f"Shape mismatch: map_2013={map_2013.shape}, map_2024={map_2024.shape}"
        )
    if proba_2013.shape != (H, W, 3):
        raise ValueError(f"proba_2013 must be (H, W, 3), got {proba_2013.shape}")
    if proba_2024.shape != (H, W, 3):
        raise ValueError(f"proba_2024 must be (H, W, 3), got {proba_2024.shape}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    # --- Step 1: Validity mask ---
    valid = (map_2013 != -1) & (map_2024 != -1)

    # --- Step 2: Confidence mask ---
    # Compute max-class probability only at valid pixels to avoid the
    # RuntimeWarning that np.nanmax raises when all values in a slice are
    # NaN (which happens at invalid pixels where proba == [nan, nan, nan]).
    # proba_2013[valid] is (n_valid, 3) — np.max over axis=1 is safe
    # because every row has exactly 3 real probability values.
    conf_2013 = np.full((H, W), np.nan, dtype=np.float32)
    conf_2024 = np.full((H, W), np.nan, dtype=np.float32)
    conf_2013[valid] = np.max(proba_2013[valid], axis=1)
    conf_2024[valid] = np.max(proba_2024[valid], axis=1)

    conf_mask = (conf_2013 >= threshold) & (conf_2024 >= threshold)

    # --- Step 3: Combined mask ---
    mask = valid & conf_mask

    # --- Step 4: Transition logic ---
    change_map = np.full((H, W), fill_value=-1, dtype=np.int8)

    # No change: identical class in both years, within valid+confident pixels
    mask_no_change = (map_2013 == map_2024) & mask
    change_map[mask_no_change] = np.int8(0)

    # Vegetation → built-up: urban expansion into green space
    mask_v2b = (map_2013 == 1) & (map_2024 == 2) & mask
    change_map[mask_v2b] = np.int8(1)

    # Other → built-up: urban expansion into bare/mixed land
    mask_o2b = (map_2013 == 0) & (map_2024 == 2) & mask
    change_map[mask_o2b] = np.int8(2)

    # Sanity: transition masks are logically disjoint given distinct class values
    assert not np.any(mask_v2b & mask_o2b),        "v→b and o→b masks overlap"
    assert not np.any(mask_no_change & mask_v2b),  "no-change and v→b overlap"
    assert not np.any(mask_no_change & mask_o2b),  "no-change and o→b overlap"

    return change_map


def change_map_summary(change_map: np.ndarray, tag: str = "") -> None:
    """
    Print a pixel-count and percentage distribution of the change map.

    Parameters
    ----------
    change_map : (H, W) int8, output of compute_change_map
    tag        : optional label for the header
    """
    header = "Change map summary" + (f" — {tag}" if tag else "")
    print(header)

    total = change_map.size
    encoding = {
        -1: "invalid / low-conf",
         0: "no change        ",
         1: "veg → built-up   ",
         2: "other → built-up ",
    }
    for val, name in encoding.items():
        count = int(np.sum(change_map == val))
        pct   = 100.0 * count / total
        print(f"  {val:>2}  {name}  {count:>10,}  ({pct:5.1f}%)")

    # Total new urban pixels (both transition types)
    new_urban = int(np.sum(change_map == 1) + np.sum(change_map == 2))
    print(f"  {'':>2}  {'total new urban  '}  {new_urban:>10,}  "
          f"({100.0 * new_urban / total:5.1f}%)")
    print()