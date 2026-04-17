"""
export.py — Export Outputs (S17)

Saves all pipeline deliverables to a structured output directory:

    outputs/
        rasters/   classified_2013.tif, classified_2024.tif, change_map.tif
        figures/   ndvi_2000.png, map_2013.png, map_2024.png,
                   change_map.png, transition_matrix.png, directional.png
        tables/    transition_matrix.csv, directional.csv
        logs/      run.log   (written by logger.py)

Public API:
    create_output_dirs(base)                           → dict of Paths
    save_rasters(maps, change_map, profile, dirs)      → None
    save_tables(transition_stats, directional, dirs)   → None
    save_figures(figures, dirs)                        → None
    export_all(...)                                    → None
"""

import numpy as np
import rasterio
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def create_output_dirs(base: str = "outputs") -> dict:
    """
    Create the standard output directory tree and return Path objects.

    Parameters
    ----------
    base : root output directory (relative or absolute)

    Returns
    -------
    dirs : dict with keys "rasters", "figures", "tables", "logs"
    """
    root = Path(base)
    dirs = {
        "root":    root,
        "rasters": root / "rasters",
        "figures": root / "figures",
        "tables":  root / "tables",
        "logs":    root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# S17a — Save GeoTIFFs
# ---------------------------------------------------------------------------

def save_rasters(
    map_2013: np.ndarray,
    map_2024: np.ndarray,
    change_map: np.ndarray,
    profile: dict,
    dirs: dict,
) -> None:
    """
    Write classified maps and change map as LZW-compressed GeoTIFFs.

    The profile from the aligned 2013 scene (S6) is used as the spatial
    reference. dtype is cast to int8 to minimise file size — all class
    values (-1, 0, 1, 2) fit within int8.

    Parameters
    ----------
    map_2013   : (H, W) int32 — classified 2013 raster
    map_2024   : (H, W) int32 — classified 2024 raster
    change_map : (H, W) int8  — transition map
    profile    : rasterio profile dict from the aligned scene
    dirs       : dict from create_output_dirs
    """
    out_profile = dict(profile)
    out_profile.update(
        driver="GTiff",
        dtype="int8",
        count=1,
        compress="lzw",
        nodata=-1,
    )
    # Remove keys that rasterio will derive from the array
    out_profile.pop("tiled", None)

    raster_dir = dirs["rasters"]

    to_save = {
        "classified_2013.tif": map_2013.astype(np.int8),
        "classified_2024.tif": map_2024.astype(np.int8),
        "change_map.tif":      change_map.astype(np.int8),
    }

    for filename, array in to_save.items():
        path = raster_dir / filename
        with rasterio.open(path, "w", **out_profile) as dst:
            dst.write(array, 1)
        print(f"  Saved raster → {path}")


# ---------------------------------------------------------------------------
# S17b — Save tables (CSV)
# ---------------------------------------------------------------------------

def save_tables(
    transition_stats: dict,
    directional_result: dict,
    dirs: dict,
) -> None:
    """
    Write transition area statistics and directional growth data as CSVs.

    Parameters
    ----------
    transition_stats   : dict from compute_transition_stats
    directional_result : dict from compute_directional_growth
    dirs               : dict from create_output_dirs
    """
    table_dir = dirs["tables"]

    # --- Transition matrix CSV ---
    # Build a 3×3 matrix of class-to-class pixel counts (mirroring the heatmap)
    classes     = [0, 1, 2]
    class_names = ["other", "vegetation", "built-up"]

    # transition_stats keys: "no_change", "veg_to_built", "other_to_built", "_meta"
    # For CSV we emit the flat per-class rows directly
    rows = []
    for key in ["veg_to_built", "other_to_built", "no_change"]:
        if key not in transition_stats:
            continue
        entry = transition_stats[key]
        rows.append({
            "transition": key,
            "pixels":     entry["pixels"],
            "area_km2":   entry["area_km2"],
        })

    df_transition = pd.DataFrame(rows)
    path_transition = table_dir / "transition_matrix.csv"
    df_transition.to_csv(path_transition, index=False)
    print(f"  Saved table  → {path_transition}")

    # --- Directional growth CSV ---
    direction_order = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    dir_rows = []
    for d in direction_order:
        if d not in directional_result:
            continue
        entry = directional_result[d]
        dir_rows.append({
            "direction": d,
            "pixels":    entry["pixels"],
            "area_km2":  entry["area_km2"],
            "pct":       entry.get("pct", 0.0),
        })

    df_directional = pd.DataFrame(dir_rows)
    path_directional = table_dir / "directional.csv"
    df_directional.to_csv(path_directional, index=False)
    print(f"  Saved table  → {path_directional}")


# ---------------------------------------------------------------------------
# S17c — Save figures (PNG)
# ---------------------------------------------------------------------------

def save_figures(figures: dict, dirs: dict) -> None:
    """
    Save all matplotlib figures to PNG at 300 dpi.

    Parameters
    ----------
    figures : dict[str → matplotlib.Figure], from build_all_figures
    dirs    : dict from create_output_dirs
    """
    fig_dir = dirs["figures"]

    filename_map = {
        "ndvi_2000":         "ndvi_2000.png",
        "map_2013":          "map_2013.png",
        "map_2024":          "map_2024.png",
        "change_map":        "change_map.png",
        "transition_matrix": "transition_matrix.png",
        "directional":       "directional.png",
    }

    for key, fig in figures.items():
        filename = filename_map.get(key, f"{key}.png")
        path = fig_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)   # release memory
        print(f"  Saved figure → {path}")


# ---------------------------------------------------------------------------
# Convenience wrapper — export everything
# ---------------------------------------------------------------------------

def export_all(
    map_2013: np.ndarray,
    map_2024: np.ndarray,
    change_map: np.ndarray,
    profile: dict,
    transition_stats: dict,
    directional_result: dict,
    figures: dict,
    base: str = "outputs",
) -> dict:
    """
    Run the full S17 export: rasters, tables, and figures.

    Parameters
    ----------
    map_2013           : (H, W) int32
    map_2024           : (H, W) int32
    change_map         : (H, W) int8
    profile            : rasterio profile from aligned 2013 scene
    transition_stats   : dict from compute_transition_stats
    directional_result : dict from compute_directional_growth
    figures            : dict from build_all_figures
    base               : root output directory

    Returns
    -------
    dirs : dict of output Paths
    """
    dirs = create_output_dirs(base)

    print("\n--- Saving rasters ---")
    save_rasters(map_2013, map_2024, change_map, profile, dirs)

    print("\n--- Saving tables ---")
    save_tables(transition_stats, directional_result, dirs)

    print("\n--- Saving figures ---")
    save_figures(figures, dirs)

    return dirs
