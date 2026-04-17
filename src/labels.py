"""
labels.py — Proxy Label Generation (S9)

Assigns weak supervision labels to pixels using rule-based thresholds
in spectral index space (NDVI, NDBI).

Public API:
    THRESHOLDS                  — module-level dict; single source of truth
    generate_labels(X)          → y_full (N_pixels,) int32
    get_valid_mask(y_full)      → boolean mask (N_pixels,)
    label_summary(y_full)       → prints class distribution

Label encoding:
    -1 → invalid (NaN input, or conflicting signal)
     0 → other   (valid but not confidently vegetation or built-up)
     1 → vegetation
     2 → built-up

No file I/O. No mutation of inputs. Thresholds are identical constants
across all years — this is mandatory for valid change detection.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds — single source of truth for all years
#
# These define confidence regions in (NDVI, NDBI) space, not decision
# boundaries. They are intentionally conservative: a pixel must be clearly
# in one regime to receive a hard label.
#
# Chosen to reflect Surat's semi-arid November conditions:
#   - Vegetation is sparse, so T_v_high is set moderately (not too strict)
#   - T_v_low raised to 0.30 to allow built-up pixels with mixed vegetation
#     signal, which is common in dense urban areas with street trees / parks
#   - T_b_high lowered to 0.05 to capture the full range of impervious
#     surfaces; Surat's NDBI distribution is compressed by atmospheric and
#     seasonal effects, so 0.10 was cutting off large genuine built-up areas
#   - T_b_low kept at 0.05 to maintain a clear gap between the vegetation
#     confidence region (NDBI ≤ 0.05) and built-up (NDBI ≥ 0.05); note this
#     means the boundary is shared — conflict resolution handles the edge
#
# Calibration evidence: with old thresholds (T_b_high=0.10, T_v_low=0.20),
# 2013 produced only 13,237 built-up pixels (0.1%) vs 896,293 in 2000 (4.8%)
# — a physically implausible collapse, not real land cover change.
# Relaxing to T_b_high=0.05, T_v_low=0.30 restores realistic proportions.
#
# Modify here and nowhere else. All functions read from this dict.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "T_v_high": 0.40,   # minimum NDVI to qualify as vegetation
    "T_v_low":  0.30,   # maximum NDVI allowed in a built-up pixel (relaxed from 0.20)
    "T_b_high": 0.05,   # minimum NDBI to qualify as built-up    (relaxed from 0.10)
    "T_b_low":  0.05,   # maximum NDBI allowed in a vegetation pixel
}

# Integer label constants — import these downstream for consistency
LABEL_INVALID    = np.int32(-1)
LABEL_OTHER      = np.int32(0)
LABEL_VEGETATION = np.int32(1)
LABEL_BUILT_UP   = np.int32(2)

# NOTE: labels are derived from NDVI/NDBI arrays in the scene dict,
# NOT from the feature matrix X. This keeps the label space and feature
# space separate, preventing circular supervision.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_thresholds(t: dict) -> None:
    if t["T_v_low"] >= t["T_v_high"]:
        raise ValueError(
            f"T_v_low ({t['T_v_low']}) must be < T_v_high ({t['T_v_high']})"
        )
    if t["T_b_low"] > t["T_b_high"]:
        raise ValueError(
            f"T_b_low ({t['T_b_low']}) must be <= T_b_high ({t['T_b_high']})"
        )


# ---------------------------------------------------------------------------
# S9 — Label generation
# ---------------------------------------------------------------------------

def generate_labels(
    scene: dict,
    thresholds: dict = None,
) -> np.ndarray:
    """
    Assign a proxy label to every pixel in a scene using spectral index rules.

    Parameters
    ----------
    scene : dict with keys 'ndvi' and 'ndbi' — (H, W) float32 arrays.
            This is the output of compute_indices. Labels are derived from
            the spectral index arrays directly, NOT from the feature matrix,
            to keep the label space and feature space fully separate.

    thresholds : dict, optional
        Override the module-level THRESHOLDS. Provide all four keys:
        T_v_high, T_v_low, T_b_high, T_b_low.
        If None, module-level THRESHOLDS is used.
        Pass the same dict for all years — this is a hard requirement.

    Returns
    -------
    y_full : np.ndarray, shape (H*W,), dtype int32
        Flattened in row-major order to match the pixel ordering of
        stack_features. Spatial index correspondence is preserved exactly.
        -1  invalid (NaN or conflicting signal)
         0  other
         1  vegetation
         2  built-up

    Notes
    -----
    Labelling logic (applied in priority order):

    Step 1 — NaN guard
        Any pixel where ndvi or ndbi is NaN → label = -1

    Step 2 — Conflict detection
        Pixels satisfying BOTH vegetation and built-up conditions
        are spectrally ambiguous (mixed pixels, index instability).
        They are removed (label = -1) rather than forced into a class.

        conflict: ndvi >= T_v_high AND ndbi >= T_b_high

    Step 3 — Class assignment (mutually exclusive after conflict removal)
        vegetation : ndvi >= T_v_high AND ndbi <= T_b_low
        built-up   : ndbi >= T_b_high AND ndvi <= T_v_low
        other      : everything else that is not -1
    """
    if "ndvi" not in scene or "ndbi" not in scene:
        raise KeyError(
            "scene must contain 'ndvi' and 'ndbi' arrays. "
            "Run compute_indices before generate_labels."
        )
    t = thresholds if thresholds is not None else THRESHOLDS
    _check_thresholds(t)

    # Flatten to 1-D in row-major order — identical to stack_features flattening
    ndvi = scene["ndvi"].reshape(-1).astype(np.float32)
    ndbi = scene["ndbi"].reshape(-1).astype(np.float32)

    N = ndvi.size
    y_full = np.full(N, LABEL_OTHER, dtype=np.int32)

    # Step 1 — NaN guard
    # A pixel is invalid if either index is NaN (NaNs are consistent across
    # columns from S8, but we check both explicitly for safety).
    nan_mask = np.isnan(ndvi) | np.isnan(ndbi)
    y_full[nan_mask] = LABEL_INVALID

    # Step 2 — Conflict detection
    # Pixels with high signal in BOTH indices are spectrally ambiguous.
    # These are set to -1 BEFORE class assignment so they cannot bleed
    # into any class region.
    conflict_mask = (ndvi >= t["T_v_high"]) & (ndbi >= t["T_b_high"])
    y_full[conflict_mask] = LABEL_INVALID

    # Step 3 — Class assignment
    # Only operates on pixels not already marked invalid.
    # Each condition is evaluated independently on the full array;
    # the invalid mask suppresses affected pixels afterwards.
    # This avoids chained conditions that obscure intent.

    already_invalid = (y_full == LABEL_INVALID)

    veg_mask = (
        (ndvi >= t["T_v_high"]) &
        (ndbi <= t["T_b_low"])  &
        ~already_invalid
    )

    built_mask = (
        (ndbi >= t["T_b_high"]) &
        (ndvi <= t["T_v_low"])  &
        ~already_invalid
    )

    # Mutual exclusivity check: the two masks must not overlap.
    # If thresholds are consistent (T_v_low < T_v_high, T_b_low < T_b_high)
    # this is guaranteed, but we assert it explicitly.
    assert not np.any(veg_mask & built_mask), (
        "vegetation and built-up masks overlap — check threshold consistency"
    )

    y_full[veg_mask]   = LABEL_VEGETATION
    y_full[built_mask] = LABEL_BUILT_UP
    # Remaining non-invalid pixels keep LABEL_OTHER (initialised above)

    return y_full


def get_valid_mask(y_full: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask: True where y_full != -1.

    Parameters
    ----------
    y_full : (N_pixels,) int32, output of generate_labels

    Returns
    -------
    valid_mask : (N_pixels,) bool
    """
    return y_full != LABEL_INVALID


def label_summary(y_full: np.ndarray, tag: str = "") -> None:
    """
    Print a human-readable class distribution for y_full.

    Parameters
    ----------
    y_full : (N_pixels,) int32
    tag    : optional string label printed in the header (e.g. "2013")
    """
    header = f"Label distribution — {tag}" if tag else "Label distribution"
    print(header)

    total = y_full.size
    for label_val, name in [
        (LABEL_INVALID,    "invalid / unlabelled"),
        (LABEL_OTHER,      "other               "),
        (LABEL_VEGETATION, "vegetation          "),
        (LABEL_BUILT_UP,   "built-up            "),
    ]:
        count = int(np.sum(y_full == label_val))
        pct   = 100.0 * count / total
        print(f"  {label_val:>2}  {name}  {count:>10,}  ({pct:5.1f}%)")

    valid = int(np.sum(y_full != LABEL_INVALID))
    print(f"  {'':>2}  {'total labelled      '}  {valid:>10,}  ({100.0 * valid / total:5.1f}%)")
    print()
