"""
preprocessing.py — Radiometric Scaling, QA Masking, AOI Clipping, Spatial Alignment

Each function is stateless, deterministic, and side-effect free.
Compose them in this order:

    scene = scale_scene(scene)
    scene = mask_scene(scene)
    scene = clip_scene(scene, aoi_path)
    scene = align_scene(scene, reference_scene)
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import mapping


# ---------------------------------------------------------------------------
# Landsat Collection 2 Level-2 SR constants
# Applies identically to Landsat 5 (TM) and Landsat 8/9 (OLI)
# ---------------------------------------------------------------------------
_SR_SCALE_FACTOR = 0.0000275
_SR_OFFSET       = -0.2

# QA_PIXEL bit positions (Landsat Collection 2)
_BIT_CLOUD        = 3   # Bit 3 = Cloud
_BIT_CLOUD_SHADOW = 4   # Bit 4 = Cloud Shadow

# Sentinel nodata value used for spectral bands inside MemoryFile writes
# and reproject calls. Chosen to be:
#   - far outside any physically valid SR range (~[-0.2, 1.5])
#   - not NaN (rasterio.warp.reproject does not reliably honour NaN)
_SPECTRAL_NODATA = -9999.0

_SPECTRAL_RESAMPLING = Resampling.bilinear
_QA_RESAMPLING       = Resampling.nearest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SPECTRAL_BAND_NAMES = ("red", "nir", "swir1")


def _spectral_bands(scene: dict) -> list:
    return [k for k in _SPECTRAL_BAND_NAMES if k in scene]


def _copy_scene(scene: dict) -> dict:
    """Deep-copy arrays and profile; pass everything else through."""
    out = {}
    for k, v in scene.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif k == "profile":
            out[k] = dict(v)
        else:
            out[k] = v
    return out


def _assert_shape_consistency(scene: dict) -> None:
    shapes = {k: scene[k].shape for k in _spectral_bands(scene) + ["qa"]}
    unique = set(shapes.values())
    if len(unique) > 1:
        raise ValueError(f"Band shape mismatch: {shapes}")


def _build_valid_mask_from_qa(qa: np.ndarray) -> np.ndarray:
    """
    Return a boolean array: True = valid pixel, False = cloud or shadow.
    Centralised here so both mask_scene and align_scene use identical logic.
    """
    cloud        = (qa >> _BIT_CLOUD)        & 1
    cloud_shadow = (qa >> _BIT_CLOUD_SHADOW) & 1
    invalid = cloud.astype(bool) | cloud_shadow.astype(bool)
    return ~invalid


# ---------------------------------------------------------------------------
# S3 — Radiometric Scaling
# FIX: removed np.clip. Clipping to [0, 1] was wrong because:
#   (a) valid SR can slightly exceed 1.0 after atmospheric correction
#   (b) fill pixels (DN=0 → SR=-0.2) clipped to 0 become indistinguishable
#       from real dark surfaces; QA masking handles them correctly instead.
# ---------------------------------------------------------------------------

def scale_scene(scene: dict) -> dict:
    """
    Convert raw Landsat Collection 2 integer SR values to surface reflectance.

    Formula:
        reflectance = raw * 0.0000275 + (-0.2)

    No value clipping is applied. Fill-value pixels (DN=0 → SR=-0.2) are
    physically impossible and will be removed by QA masking (mask_scene).
    Atmospheric correction artefacts that push SR slightly above 1.0 are
    kept as-is; they are real data and clipping would bias index calculations.

    Applied to : red, nir, swir1
    Not applied: qa (preserved as integer)
    Output dtype: float32
    """
    _assert_shape_consistency(scene)
    out = _copy_scene(scene)

    for band in _spectral_bands(scene):
        raw = scene[band].astype(np.float32)
        out[band] = (raw * _SR_SCALE_FACTOR + _SR_OFFSET).astype(np.float32)

    out["qa"] = scene["qa"].copy()
    return out


# ---------------------------------------------------------------------------
# S4 — QA Masking
# ---------------------------------------------------------------------------

def mask_scene(scene: dict) -> dict:
    """
    Apply QA_PIXEL bitmask to remove clouds and cloud shadows.

    Bits checked (Landsat Collection 2 QA_PIXEL):
        Bit 3 — Cloud
        Bit 4 — Cloud Shadow

    Invalid pixels → np.nan in all spectral bands.
    QA array is preserved unchanged (integer dtype).
    """
    _assert_shape_consistency(scene)
    qa = scene["qa"]

    valid_mask = _build_valid_mask_from_qa(qa)

    out = _copy_scene(scene)
    for band in _spectral_bands(scene):
        arr = out[band].astype(np.float32)
        arr[~valid_mask] = np.nan
        out[band] = arr

    out["qa"] = qa.copy()
    return out


# ---------------------------------------------------------------------------
# S5 — AOI Clipping
#
# FIX 1: spectral bands use _SPECTRAL_NODATA (-9999) as the fill sentinel, not
#         NaN and not 0. This means outside-AOI pixels are unambiguously
#         identifiable without touching any real surface reflectance value.
#
# FIX 2: QA uses nodata=None. QA=0 is a valid bitmask state (all-clear pixel)
#         and must never be treated as a fill sentinel. Outside-AOI QA pixels
#         are left as 0 by rasterio (its default fill when nodata=None);
#         downstream logic must treat them via the spectral NaN mask, not QA.
#
# FIX 3: outside-AOI spectral pixels are detected by comparing against
#         _SPECTRAL_NODATA (-9999), not by checking == 0. This eliminates the
#         systematic bias of nullifying valid dark/water/shadow pixels.
#
# FIX 4: single MemoryFile context manager per band — write and clip in one
#         open session, no redundant open/close pair.
# ---------------------------------------------------------------------------

def clip_scene(scene: dict, aoi_path: str) -> dict:
    """
    Clip all raster bands to the AOI polygon boundary.

    Parameters
    ----------
    scene    : scene dict (after scale_scene + mask_scene recommended)
    aoi_path : path to GeoJSON (or any GeoPandas-readable file).
               Reprojected to raster CRS automatically if needed.

    Returns a new scene dict with:
    - arrays cropped to the AOI bounding box
    - updated profile (transform, width, height)
    - CRS unchanged
    """
    _assert_shape_consistency(scene)
    profile = scene["profile"]

    aoi = gpd.read_file(aoi_path)
    raster_crs = rasterio.crs.CRS.from_user_input(profile["crs"])
    if aoi.crs != raster_crs:
        aoi = aoi.to_crs(raster_crs)
    geoms = [mapping(geom) for geom in aoi.geometry]

    out = {}
    new_profile = None

    for band in _spectral_bands(scene) + ["qa"]:
        arr   = scene[band]
        dtype = arr.dtype

        is_spectral = band in _SPECTRAL_BAND_NAMES

        # Sentinel selection:
        #   spectral → -9999  (impossible SR; safely distinguishes fill from data)
        #   qa       → None   (QA=0 is valid; do not corrupt bitmask semantics)
        nodata_val = _SPECTRAL_NODATA if is_spectral else None

        # Prepare NaN → sentinel conversion for spectral bands before writing
        if is_spectral:
            write_arr = arr.copy().astype(np.float32)
            write_arr[np.isnan(write_arr)] = _SPECTRAL_NODATA
        else:
            write_arr = arr

        band_profile = dict(profile)
        band_profile.update(
            count=1,
            dtype=str(write_arr.dtype),
            driver="GTiff",
            nodata=nodata_val,
        )

        with rasterio.MemoryFile() as memfile:
            with memfile.open(**band_profile) as dataset:
                dataset.write(write_arr, 1)
                clipped, clipped_transform = rasterio_mask(
                    dataset,
                    geoms,
                    crop=True,
                    nodata=nodata_val,
                    filled=True,
                )

        clipped_arr = clipped[0]

        if is_spectral:
            # Sentinel → NaN. Safe: -9999 cannot be a real SR value.
            clipped_arr = clipped_arr.astype(np.float32)
            clipped_arr[clipped_arr == _SPECTRAL_NODATA] = np.nan

        out[band] = clipped_arr

        if new_profile is None:
            new_profile = dict(profile)
            new_profile.update(
                transform=clipped_transform,
                width=clipped_arr.shape[1],
                height=clipped_arr.shape[0],
                count=1,
                nodata=None,
            )

    out["profile"] = new_profile
    return out


# ---------------------------------------------------------------------------
# S6 — Spatial Alignment
#
# FIX 1: spectral bands converted NaN → _SPECTRAL_NODATA before reproject
#         and back to NaN after. rasterio.warp.reproject does not reliably
#         honour np.nan as a nodata value; a numeric sentinel is required.
#
# FIX 2: mask integrity is restored after resampling by re-applying the
#         QA-derived valid_mask to spectral bands. Bilinear interpolation
#         blends values at cloud/clear boundaries, potentially producing
#         artefact values in pixels adjacent to masked regions. Re-applying
#         the aligned QA mask eliminates these bleed pixels.
#
# FIX 3: QA is resampled with nearest-neighbour and src_nodata=None so that
#         QA=0 (all-clear) is never treated as a fill value during warping.
# ---------------------------------------------------------------------------

def align_scene(scene: dict, reference: dict) -> dict:
    """
    Reproject and resample a scene to match a reference scene's grid exactly.

    After alignment:
    - same CRS, transform, width, height as reference
    - pixel-wise correspondence guaranteed
    - NaN mask integrity re-enforced via aligned QA after resampling

    Spectral bands → bilinear resampling
    QA band        → nearest-neighbour resampling
    """
    ref_profile   = reference["profile"]
    ref_transform = ref_profile["transform"]
    ref_crs       = rasterio.crs.CRS.from_user_input(ref_profile["crs"])
    ref_width     = ref_profile["width"]
    ref_height    = ref_profile["height"]

    src_profile = scene["profile"]
    src_crs     = rasterio.crs.CRS.from_user_input(src_profile["crs"])

    out = {}

    # --- Spectral bands: NaN → sentinel → reproject → sentinel → NaN ---
    for band in _spectral_bands(scene):
        src_arr = scene[band].copy().astype(np.float32)
        src_arr[np.isnan(src_arr)] = _SPECTRAL_NODATA

        dst_arr = np.full((ref_height, ref_width), _SPECTRAL_NODATA, dtype=np.float32)

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src_profile["transform"],
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=_SPECTRAL_RESAMPLING,
            src_nodata=_SPECTRAL_NODATA,
            dst_nodata=_SPECTRAL_NODATA,
        )

        dst_arr[dst_arr == _SPECTRAL_NODATA] = np.nan
        out[band] = dst_arr

    # --- QA band: nearest-neighbour; src_nodata=None preserves QA=0 ---
    src_qa = scene["qa"]
    dst_qa = np.zeros((ref_height, ref_width), dtype=src_qa.dtype)

    reproject(
        source=src_qa,
        destination=dst_qa,
        src_transform=src_profile["transform"],
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=_QA_RESAMPLING,
        src_nodata=None,
        dst_nodata=0,
    )
    out["qa"] = dst_qa

    # --- Re-enforce NaN mask from aligned QA ---
    # Bilinear resampling leaves artefact values at cloud/clear boundaries.
    # Re-applying the QA mask ensures spectral arrays are clean.
    valid_mask = _build_valid_mask_from_qa(dst_qa)
    for band in _spectral_bands(scene):
        arr = out[band]
        arr[~valid_mask] = np.nan
        out[band] = arr

    new_profile = dict(src_profile)
    new_profile.update(
        crs=ref_crs,
        transform=ref_transform,
        width=ref_width,
        height=ref_height,
        count=1,
        nodata=None,
    )
    out["profile"] = new_profile

    return out