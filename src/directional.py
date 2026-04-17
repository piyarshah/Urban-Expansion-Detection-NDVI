"""
directional.py — Directional Urban Expansion Analysis (S15)

Computes the geographic direction of new urban growth relative to the AOI
centroid by binning new built-up pixels into 8 compass directions.

Public API:
    compute_directional_growth(change_map, transform, aoi_path, pixel_area_km2)
        → dict of {direction: {"pixels": int, "area_km2": float, "pct": float}}
    print_directional_growth(result)
        → compass table + polar-style bar to stdout

No file I/O beyond reading the AOI file. No mutation of inputs.
"""

import numpy as np
import geopandas as gpd
import rasterio.transform


# 8-direction compass bins
# Angles are measured from East (0°), increasing counter-clockwise.
# arctan2(dy, dx) returns angles in standard math convention;
# we convert to compass convention below.
_DIRECTION_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]

# Bin edges: each direction spans 45°, centred on its cardinal angle.
# E = [337.5, 22.5), NE = [22.5, 67.5), ... SE = [292.5, 337.5)
# Implemented as a flat [0, 360) digitise with wrap-around.
_BIN_EDGES = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360], dtype=np.float64)


def compute_directional_growth(
    change_map: np.ndarray,
    transform,
    aoi_path: str,
    pixel_area_km2: float,
) -> dict:
    """
    Bin new urban growth pixels into 8 compass directions from the AOI centroid.

    Parameters
    ----------
    change_map     : (H, W) int8, output of compute_change_map
                     Values {-1, 0, 1, 2}. Growth pixels: change_map ∈ {1, 2}.
    transform      : rasterio Affine transform from the aligned scene profile.
                     Must match the grid used to produce change_map.
    aoi_path       : path to the AOI GeoJSON. Used to compute the centroid.
                     Reprojected to the raster CRS (EPSG:32643) automatically.
    pixel_area_km2 : float — area per pixel in km², from quantification stats.

    Returns
    -------
    result : dict
        Keys: direction labels ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        Each value:
            {
                "pixels":   int,
                "area_km2": float,
                "pct":      float,   # percentage of all new urban pixels
            }
        Also contains "_meta" key with centroid and total counts.

    Algorithm
    ---------
    Step 1 — AOI centroid
        Load AOI polygon, reproject to UTM (EPSG:32643), compute centroid.
        The centroid is the reference point for all directional vectors.

    Step 2 — Extract growth pixels
        Identify all pixels where change_map ∈ {1, 2} (both transition types).

    Step 3 — Pixel → map coordinates
        Convert row/col indices to (x, y) UTM coordinates using the affine
        transform. rasterio.transform.xy returns centre coordinates of pixels.

    Step 4 — Direction vectors
        dx = x - centroid_x
        dy = y - centroid_y
        Vectors point FROM the centroid TO each growth pixel.

    Step 5 — Angle computation
        Standard math angles: arctan2(dy, dx) → [-180°, 180°]
        Shift to [0°, 360°): angles = (angles + 360) % 360

        Angle interpretation (standard math / geographic mapping):
            0°   = East
            90°  = North
            180° = West
            270° = South

        Bins are defined in 45° increments starting at 0° (East).

    Step 6 — Bin assignment
        np.digitize assigns each angle to a bin index 1-8.
        Angles at exactly 360° are wrapped to bin 0 (East).

    Notes
    -----
    - rasterio.transform.xy returns lists for large arrays; converting to
      np.array is required before arithmetic.
    - The centroid is computed in the same CRS as the raster (UTM 43N),
      so dx/dy are in metres and angle computation is geometrically correct.
    - Pixels at the exact centroid (dx=dy=0) would produce undefined angles
      but are physically impossible at 30m resolution.
    """
    if change_map.ndim != 2:
        raise ValueError(f"change_map must be 2-D, got shape {change_map.shape}")

    # --- Step 1: AOI centroid ---
    aoi = gpd.read_file(aoi_path)

    # Determine raster CRS from transform — assume EPSG:32643 (UTM 43N)
    # consistent with the pipeline; reproject AOI if needed
    target_crs = "EPSG:32643"
    if str(aoi.crs) != target_crs:
        aoi = aoi.to_crs(target_crs)

    centroid = aoi.geometry.centroid.iloc[0]
    cx, cy   = centroid.x, centroid.y

    # --- Step 2: Extract growth pixel indices ---
    growth_mask = (change_map == 1) | (change_map == 2)
    rows, cols  = np.where(growth_mask)

    if len(rows) == 0:
        # No growth detected — return zero counts
        result = {
            d: {"pixels": 0, "area_km2": 0.0, "pct": 0.0}
            for d in _DIRECTION_LABELS
        }
        result["_meta"] = {
            "centroid_x": cx, "centroid_y": cy,
            "total_growth_pixels": 0,
        }
        return result

    # --- Step 3: Pixel indices → UTM coordinates ---
    # rasterio.transform.xy returns centre of each pixel cell
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)

    # --- Step 4: Direction vectors from centroid ---
    dx = xs - cx
    dy = ys - cy

    # --- Step 5: Angle computation ---
    # arctan2(dy, dx): standard math convention, East = 0°, CCW positive
    angles = np.degrees(np.arctan2(dy, dx))   # [-180, 180]
    angles = (angles + 360.0) % 360.0         # [0, 360)

    # --- Step 6: Bin assignment ---
    # digitize returns 1-indexed bins; subtract 1 for 0-indexed labels
    idx = np.digitize(angles, _BIN_EDGES) - 1
    # Wrap: any angle == 360 falls in bin 8 (out of range) → bin 0 (East)
    idx[idx == 8] = 0

    # --- Step 7: Count per direction ---
    total_growth = len(rows)
    result = {}
    for i, label in enumerate(_DIRECTION_LABELS):
        count = int(np.sum(idx == i))
        result[label] = {
            "pixels":   count,
            "area_km2": count * pixel_area_km2,
            "pct":      100.0 * count / total_growth if total_growth > 0 else 0.0,
        }

    result["_meta"] = {
        "centroid_x":          cx,
        "centroid_y":          cy,
        "total_growth_pixels": total_growth,
        "total_growth_km2":    total_growth * pixel_area_km2,
    }

    return result


def print_directional_growth(result: dict, tag: str = "") -> None:
    """
    Print a compass table and ASCII polar bar for directional growth.

    Parameters
    ----------
    result : dict returned by compute_directional_growth
    tag    : optional label for the header
    """
    header = "Directional urban expansion" + (f" — {tag}" if tag else "")
    print(header)

    meta = result.get("_meta", {})
    if meta:
        print(f"  AOI centroid   : ({meta['centroid_x']:.0f}, {meta['centroid_y']:.0f}) UTM 43N")
        print(f"  Total growth   : {meta['total_growth_pixels']:,} pixels  "
              f"({meta['total_growth_km2']:.4f} km²)")
    print()

    print(f"  {'direction':<6}  {'pixels':>10}  {'area_km2':>10}  {'pct':>7}  bar")
    print("  " + "-" * 60)

    # Print in compass order: N, NE, E, SE, S, SW, W, NW
    compass_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    for d in compass_order:
        row = result[d]
        bar = "█" * int(row["pct"] / 2)   # 2% per character → max 50 chars
        print(
            f"  {d:<6}  "
            f"{row['pixels']:>10,}  "
            f"{row['area_km2']:>10.4f}  "
            f"{row['pct']:>6.1f}%  {bar}"
        )
    print()

    # Dominant direction
    dirs_only = {k: v for k, v in result.items() if k != "_meta"}
    dominant  = max(dirs_only, key=lambda d: dirs_only[d]["pixels"])
    print(f"  Dominant direction: {dominant}  "
          f"({result[dominant]['pct']:.1f}% of new urban pixels)")
    print()
