"""
features.py — Spectral Index Computation (S7) and Feature Stacking (S8)

Public API:
    compute_ndvi(red, nir)       → (H, W) float32
    compute_ndbi(nir, swir1)     → (H, W) float32
    compute_indices(scene)       → scene dict extended with 'ndvi' and 'ndbi'
    stack_features(scene)        → np.ndarray of shape (H*W, 3)

No file I/O. No mutation of inputs. No geospatial logic. No thresholds.
"""

import numpy as np


# Small epsilon added to all denominators to prevent division by zero.
# Large enough to avoid true zero but negligible relative to any real SR value.
_EPS = np.float32(1e-6)

# Column ordering in the feature matrix — fixed, consistent across all scenes.
_FEATURE_COLUMNS = ("red", "nir", "swir1")


# ---------------------------------------------------------------------------
# S7 — Spectral Indices
# ---------------------------------------------------------------------------

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute the Normalised Difference Vegetation Index.

        NDVI = (NIR - RED) / (NIR + RED + ε)

    Parameters
    ----------
    red : (H, W) float32 array
    nir : (H, W) float32 array

    Returns
    -------
    ndvi : (H, W) float32 array
        NaNs propagated from any NaN in red or nir.
        Clipped to [-1, 1]: values outside this range are atmospheric
        correction artefacts with no physical meaning. NaNs are preserved.
    """
    if red.shape != nir.shape:
        raise ValueError(f"Shape mismatch: red={red.shape}, nir={nir.shape}")

    red  = red.astype(np.float32)
    nir  = nir.astype(np.float32)

    ndvi = (nir - red) / (nir + red + _EPS)
    # Clip to the physically valid range. Values outside [-1, 1] are
    # atmospheric correction artefacts — they carry no spectral meaning
    # and would bias index statistics and visualisations.
    # NaNs are preserved: np.clip passes them through unchanged.
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def compute_ndbi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Compute the Normalised Difference Built-up Index.

        NDBI = (SWIR1 - NIR) / (SWIR1 + NIR + ε)

    Sign convention:
        positive → built-up / bare soil dominant
        negative → vegetation dominant

    Parameters
    ----------
    nir   : (H, W) float32 array
    swir1 : (H, W) float32 array

    Returns
    -------
    ndbi : (H, W) float32 array
        NaNs propagated from any NaN in nir or swir1.
        Values are not clipped to [-1, 1].
    """
    if nir.shape != swir1.shape:
        raise ValueError(f"Shape mismatch: nir={nir.shape}, swir1={swir1.shape}")

    nir   = nir.astype(np.float32)
    swir1 = swir1.astype(np.float32)

    ndbi = (swir1 - nir) / (swir1 + nir + _EPS)
    return ndbi.astype(np.float32)


def compute_indices(scene: dict) -> dict:
    """
    Extend a preprocessed scene dict with NDVI and NDBI arrays.

    Does not mutate the input dict.
    All existing keys are preserved unchanged.

    Parameters
    ----------
    scene : dict with keys 'red', 'nir', 'swir1', 'qa', 'profile'

    Returns
    -------
    dict with two additional keys:
        'ndvi' : (H, W) float32
        'ndbi' : (H, W) float32
    """
    red   = scene["red"]
    nir   = scene["nir"]
    swir1 = scene["swir1"]

    # Shape guard
    shape = red.shape
    if nir.shape != shape or swir1.shape != shape:
        raise ValueError(
            f"Band shape mismatch: red={shape}, nir={nir.shape}, swir1={swir1.shape}"
        )

    ndvi = compute_ndvi(red, nir)
    ndbi = compute_ndbi(nir, swir1)

    # NaN propagation check: if any input band is NaN at a pixel,
    # the corresponding output index must also be NaN.
    # This is guaranteed by NumPy arithmetic but asserted here for safety.
    input_nan = np.isnan(red) | np.isnan(nir) | np.isnan(swir1)
    assert np.all(np.isnan(ndvi[input_nan])), "NDVI NaN propagation failure"
    assert np.all(np.isnan(ndbi[input_nan])), "NDBI NaN propagation failure"

    # Divergence guard: ±inf must not appear (epsilon prevents true zero denom)
    if np.any(np.isinf(ndvi)) or np.any(np.isinf(ndbi)):
        raise RuntimeError("Infinite values in computed indices — check input arrays")

    out = dict(scene)       # shallow copy of dict; arrays not copied
    out["ndvi"] = ndvi
    out["ndbi"] = ndbi
    return out


# ---------------------------------------------------------------------------
# S8 — Feature Stacking (Raster → Tabular)
# ---------------------------------------------------------------------------

def stack_features(scene: dict) -> np.ndarray:
    """
    Flatten and column-stack spectral index bands into a 2-D feature matrix.

    Column order: [red, nir, swir1]  (see _FEATURE_COLUMNS)

    Raw reflectance bands only. Derived indices (ndvi, ndbi) are stored in
    the scene dict by compute_indices but are intentionally excluded here.
    Labels are generated from ndvi/ndbi thresholds in labels.py — including
    those same indices as model features creates circular supervision where
    the RF trivially memorises the threshold rules (100% val accuracy) rather
    than learning a generalisable spectral boundary. Using raw bands forces
    the model to approximate the class regions in reflectance space, producing
    honest validation accuracy and better cross-sensor generalisation.

    All bands are flattened in row-major (C) order — consistent with NumPy
    default — so spatial pixel ordering is identical across all columns and
    all scenes. This invariant makes pixel-wise temporal comparison valid.

    NaN pixels are retained. Filtering happens downstream in S9/S10
    to preserve spatial indexing.

    Parameters
    ----------
    scene : dict with keys 'red', 'nir', 'swir1'
            (i.e. output of compute_indices, which also preserves 'swir1')

    Returns
    -------
    X : np.ndarray of shape (H*W, 3), dtype float32
        Columns: [red, nir, swir1]
    """
    for key in _FEATURE_COLUMNS:
        if key not in scene:
            raise KeyError(
                f"'{key}' not found in scene. "
                f"Ensure compute_indices has been called and 'swir1' is present."
            )

    shape = scene[_FEATURE_COLUMNS[0]].shape
    for key in _FEATURE_COLUMNS:
        if scene[key].shape != shape:
            raise ValueError(
                f"Shape mismatch: {_FEATURE_COLUMNS[0]}={shape}, {key}={scene[key].shape}"
            )

    n_features = len(_FEATURE_COLUMNS)

    # Flatten each band in row-major order, then stack column-wise.
    X = np.stack(
        [scene[key].reshape(-1) for key in _FEATURE_COLUMNS],
        axis=1,
    ).astype(np.float32)

    assert X.shape == (shape[0] * shape[1], n_features), (
        f"Unexpected feature matrix shape: {X.shape}"
    )
    return X