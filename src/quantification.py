"""
quantification.py — Transition Area Quantification (S14)

Converts pixel counts from the change map into physical areas (km²)
using the raster's affine transform to derive pixel dimensions.

Public API:
    compute_transition_stats(change_map, transform)
        → dict of {transition_name: {"pixels": int, "area_km2": float}}
    print_transition_stats(stats)
        → formatted table to stdout

No file I/O. No mutation of inputs.
"""

import numpy as np
from rasterio.transform import Affine


# Change map encoding — mirrors change_detection.py
_TRANSITIONS = {
    0: "no_change",
    1: "veg_to_built",
    2: "other_to_built",
}


def compute_transition_stats(
    change_map: np.ndarray,
    transform: Affine,
) -> dict:
    """
    Count pixels per transition class and convert to area in km².

    Parameters
    ----------
    change_map : (H, W) int8, output of compute_change_map
                 Values: -1 (invalid), 0 (no change), 1 (veg→built), 2 (other→built)
    transform  : rasterio Affine transform from the aligned scene profile.
                 Used to extract pixel dimensions in map units (metres for UTM).

    Returns
    -------
    stats : dict
        Keys: "no_change", "veg_to_built", "other_to_built"
        Each value is a dict:
            {
                "pixels":   int,
                "area_km2": float,
            }

    Notes
    -----
    Pixel area is derived from the transform, not hardcoded. The transform
    diagonal elements give pixel width (x) and height (y) in map units.
    For EPSG:32643 (UTM Zone 43N), units are metres, so:
        pixel_area_m2  = |transform.a| * |transform.e|
        pixel_area_km2 = pixel_area_m2 / 1e6

    Invalid pixels (change_map == -1) are excluded from all counts and
    are not reported in the output dict.
    """
    if change_map.ndim != 2:
        raise ValueError(f"change_map must be 2-D, got shape {change_map.shape}")

    # Extract pixel dimensions from transform
    # transform.a = pixel width (positive), transform.e = pixel height (negative)
    res_x = abs(transform.a)   # metres per pixel in x
    res_y = abs(transform.e)   # metres per pixel in y
    pixel_area_m2  = res_x * res_y
    pixel_area_km2 = pixel_area_m2 / 1e6

    stats = {}
    for val, name in _TRANSITIONS.items():
        count = int(np.sum(change_map == val))
        stats[name] = {
            "pixels":   count,
            "area_km2": count * pixel_area_km2,
        }

    # Store pixel area metadata for downstream use (e.g. directional.py)
    stats["_meta"] = {
        "pixel_area_km2": pixel_area_km2,
        "res_x_m":        res_x,
        "res_y_m":        res_y,
        "total_pixels":   change_map.size,
        "valid_pixels":   int(np.sum(change_map != -1)),
        "invalid_pixels": int(np.sum(change_map == -1)),
    }

    return stats


def print_transition_stats(stats: dict, tag: str = "") -> None:
    """
    Print a formatted transition area table.

    Parameters
    ----------
    stats : dict returned by compute_transition_stats
    tag   : optional label for the header
    """
    header = "Transition area statistics" + (f" — {tag}" if tag else "")
    print(header)

    meta = stats.get("_meta", {})
    if meta:
        print(f"  Pixel resolution : {meta['res_x_m']:.0f} m × {meta['res_y_m']:.0f} m")
        print(f"  Pixel area       : {meta['pixel_area_km2']:.6f} km²")
        print(f"  Valid pixels     : {meta['valid_pixels']:,} / {meta['total_pixels']:,}")
    print()

    col_w = 16
    print(f"  {'transition':<18}  {'pixels':>12}  {'area_km2':>12}")
    print("  " + "-" * 46)

    display_order = ["veg_to_built", "other_to_built", "no_change"]
    for name in display_order:
        if name not in stats:
            continue
        row = stats[name]
        print(
            f"  {name:<18}  "
            f"{row['pixels']:>12,}  "
            f"{row['area_km2']:>12.4f}"
        )

    # Total new urban
    total_new_urban_px = (
        stats.get("veg_to_built",   {}).get("pixels", 0) +
        stats.get("other_to_built", {}).get("pixels", 0)
    )
    total_new_urban_km2 = (
        stats.get("veg_to_built",   {}).get("area_km2", 0.0) +
        stats.get("other_to_built", {}).get("area_km2", 0.0)
    )
    print("  " + "-" * 46)
    print(
        f"  {'total_new_urban':<18}  "
        f"{total_new_urban_px:>12,}  "
        f"{total_new_urban_km2:>12.4f}"
    )
    print()
