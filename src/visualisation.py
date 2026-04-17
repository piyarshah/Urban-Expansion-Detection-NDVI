"""
visualisation.py — Visualisation (S16)

Pure plotting functions. No file I/O — all functions return matplotlib Figure
objects. Saving is handled by export.py (S17).

Public API:
    plot_ndvi_map(ndvi_2000, aoi_path)
    plot_classified_map(class_map, year, aoi_path)
    plot_change_map(change_map, transition_stats, directional_result, aoi_path)
    plot_transition_matrix(map_2013, map_2024)
    plot_directional_polar(directional_result)
    build_all_figures(...)                    → dict of Figures

Design invariants
-----------------
- Uniform font sizes via _FONT module-level dict
- Consistent title format "... — 2013 → 2024" across all plots
- Same colour palette for shared concepts (CLASS_COLORS) everywhere
- AOI boundary overlay on all raster plots
- NDVI strictly clipped to [-1, 1] before display
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import geopandas as gpd
from pathlib import Path


# ---------------------------------------------------------------------------
# Uniform typography — change here, applies everywhere
# ---------------------------------------------------------------------------
_FONT = {
    "title":   14,
    "subtitle": 11,
    "axis":    11,
    "legend":  11,
    "annot":   10,
    "small":    8,
}

# ---------------------------------------------------------------------------
# Shared colour definitions — consistent across ALL plots
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    -1: (0.05, 0.05, 0.05),   # near-black → invalid
     0: (0.82, 0.82, 0.82),   # light grey → other
     1: (0.10, 0.68, 0.10),   # vivid green → vegetation (increased saturation)
     2: (0.90, 0.10, 0.10),   # vivid red   → built-up   (increased saturation)
}
CLASS_LABELS = {-1: "invalid", 0: "other", 1: "vegetation", 2: "built-up"}

CHANGE_COLORS = {
    -1: (0.05, 0.05, 0.05),   # near-black    → invalid
     0: (0.92, 0.92, 0.92),   # very light grey (lightened to reduce dominance)
     1: (0.15, 0.35, 0.95),   # blue           → vegetation → built-up
     2: (0.95, 0.55, 0.05),   # amber          → other → built-up
}
CHANGE_LABELS = {
    -1: "invalid",
     0: "no change",
     1: "vegetation → built-up",
     2: "other → built-up",
}

# AOI overlay style
_AOI_STYLE = dict(edgecolor="#222222", facecolor="none", linewidth=1.2, zorder=5)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _class_map_to_rgb(class_map: np.ndarray, color_dict: dict) -> np.ndarray:
    """Convert an integer label map to an (H, W, 3) float32 RGB image."""
    H, W = class_map.shape
    rgb  = np.zeros((H, W, 3), dtype=np.float32)
    for cls, color in color_dict.items():
        rgb[class_map == cls] = color
    return rgb


def _legend_patches(color_dict: dict, label_dict: dict) -> list:
    """Return Patch objects for a custom legend, sorted by key."""
    return [
        mpatches.Patch(facecolor=color_dict[k], label=label_dict[k])
        for k in sorted(color_dict.keys())
        if k in label_dict
    ]


def _overlay_aoi(ax: plt.Axes, aoi_path: str, transform, shape: tuple) -> None:
    """
    Overlay the AOI boundary on a raster axis as a thin outline.

    Converts AOI geometries from projected CRS to pixel coordinates using
    the raster affine transform, so the overlay is pixel-accurate without
    needing cartopy or a proper geo-aware axis.
    """
    try:
        from rasterio.transform import rowcol
        aoi = gpd.read_file(aoi_path)
        H, W = shape

        for geom in aoi.geometry:
            if geom is None:
                continue
            # Handle both Polygon and MultiPolygon
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                xs, ys = poly.exterior.xy
                rows, cols = rowcol(transform, xs, ys)
                rows = np.clip(rows, 0, H - 1)
                cols = np.clip(cols, 0, W - 1)
                ax.plot(cols, rows, color=_AOI_STYLE["edgecolor"],
                        linewidth=_AOI_STYLE["linewidth"],
                        zorder=_AOI_STYLE["zorder"])
    except Exception:
        # AOI overlay is best-effort — never break a figure over it
        pass


# ---------------------------------------------------------------------------
# 1. NDVI map — 2000
# ---------------------------------------------------------------------------

def plot_ndvi_map(
    ndvi_2000: np.ndarray,
    aoi_path: str = None,
    transform=None,
) -> plt.Figure:
    """
    Plot the 2000 NDVI raster using a diverging Red-Yellow-Green colour map.

    NDVI is clipped strictly to [-1, 1] before display. NaNs are masked so
    they render as white. AOI boundary is overlaid if aoi_path is provided.
    """
    ndvi_clipped = np.clip(ndvi_2000, -1.0, 1.0)   # strict clip, idempotent
    ndvi_masked  = np.ma.masked_invalid(ndvi_clipped)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ndvi_masked, cmap="RdYlGn", vmin=-1.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, label="NDVI", fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=_FONT["small"])
    cbar.set_label("NDVI", fontsize=_FONT["axis"])

    ax.set_title("NDVI — 2000 (Landsat 5)", fontsize=_FONT["title"], fontweight="bold")
    ax.axis("off")

    if aoi_path and transform is not None:
        _overlay_aoi(ax, aoi_path, transform, ndvi_2000.shape)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Classified land cover maps
# ---------------------------------------------------------------------------

def plot_classified_map(
    class_map: np.ndarray,
    year: str,
    aoi_path: str = None,
    transform=None,
) -> plt.Figure:
    """
    Render a classified land cover raster with the shared 4-class colour scheme.

    Vivid red/green values ensure built-up is visually distinct from vegetation.
    AOI boundary is overlaid if provided.
    """
    rgb = _class_map_to_rgb(class_map, CLASS_COLORS)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    ax.set_title(
        f"Land Cover Classification — {year}",
        fontsize=_FONT["title"], fontweight="bold",
    )
    ax.axis("off")

    if aoi_path and transform is not None:
        _overlay_aoi(ax, aoi_path, transform, class_map.shape)

    legend = ax.legend(
        handles=_legend_patches(CLASS_COLORS, CLASS_LABELS),
        loc="lower right",
        fontsize=_FONT["legend"],
        framealpha=0.85,
        title="Class",
        title_fontsize=_FONT["legend"],
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Change map
# ---------------------------------------------------------------------------

def plot_change_map(
    change_map: np.ndarray,
    transition_stats: dict = None,
    directional_result: dict = None,
    aoi_path: str = None,
    transform=None,
) -> plt.Figure:
    """
    Render the 2013→2024 transition map.

    Improvements over previous version:
    - "no change" background lightened to (0.92, 0.92, 0.92) — less dominant
    - Legend font size increased to _FONT["legend"]
    - Total new built-up area (km²) annotated in top-left corner
    - Dominant growth direction annotated in bottom-left corner
    - AOI boundary overlaid if provided
    """
    rgb = _class_map_to_rgb(change_map.astype(np.int32), CHANGE_COLORS)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    ax.set_title(
        "Urban Change Detection — 2013 → 2024",
        fontsize=_FONT["title"], fontweight="bold",
    )
    ax.axis("off")

    if aoi_path and transform is not None:
        _overlay_aoi(ax, aoi_path, transform, change_map.shape)

    # Annotation: total growth area
    if transition_stats is not None:
        total_km2 = (
            transition_stats.get("veg_to_built",   {}).get("area_km2", 0.0) +
            transition_stats.get("other_to_built", {}).get("area_km2", 0.0)
        )
        ax.text(
            0.02, 0.98,
            f"Total new built-up: {total_km2:.2f} km²",
            transform=ax.transAxes,
            fontsize=_FONT["annot"], fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
        )

    # Annotation: dominant direction
    if directional_result is not None:
        direction_order = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        dom = max(direction_order, key=lambda d: directional_result[d]["pixels"])
        dom_area = directional_result[dom]["area_km2"]
        ax.text(
            0.02, 0.04,
            f"Dominant growth: {dom}  ({dom_area:.2f} km²)",
            transform=ax.transAxes,
            fontsize=_FONT["annot"], fontweight="bold",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
        )

    ax.legend(
        handles=_legend_patches(CHANGE_COLORS, CHANGE_LABELS),
        loc="lower right",
        fontsize=_FONT["legend"],
        framealpha=0.85,
        title="Transition",
        title_fontsize=_FONT["legend"],
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Transition matrix heatmap
# ---------------------------------------------------------------------------

def plot_transition_matrix(
    map_2013: np.ndarray,
    map_2024: np.ndarray,
) -> plt.Figure:
    """
    Build and plot the 3×3 pixel transition count matrix (2013 rows, 2024 cols).

    Improvements:
    - Log-scale colourmap so built-up transitions (small counts) are visible
      alongside the dominant no-change diagonal (large counts)
    - Colorbar labelled "pixels (log scale)" for clarity
    - Cell annotations show raw pixel counts (millions for large cells,
      exact counts for built-up transitions so they're not suppressed)
    - Separate inset bar chart for built-up transitions (bottom panel)
      gives visual emphasis to the small but meaningful values
    """
    classes     = [0, 1, 2]
    class_names = ["other", "vegetation", "built-up"]

    valid = (map_2013 != -1) & (map_2024 != -1)

    M = np.zeros((3, 3), dtype=np.int64)
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            M[i, j] = int(np.sum((map_2013 == ci) & (map_2024 == cj) & valid))

    # --- Main heatmap with log scale ---
    fig, (ax_main, ax_inset) = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    M_safe = np.where(M > 0, M, 1)   # log(0) guard — 0 cells become 1 for display
    im = ax_main.imshow(M_safe, cmap="viridis", norm=LogNorm(vmin=1, vmax=M.max()))

    cbar = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label("pixels (log scale)", fontsize=_FONT["axis"])
    cbar.ax.tick_params(labelsize=_FONT["small"])

    ax_main.set_xticks([0, 1, 2])
    ax_main.set_yticks([0, 1, 2])
    ax_main.set_xticklabels(class_names, fontsize=_FONT["axis"])
    ax_main.set_yticklabels(class_names, fontsize=_FONT["axis"])
    ax_main.set_xlabel("2024 class", fontsize=_FONT["axis"])
    ax_main.set_ylabel("2013 class", fontsize=_FONT["axis"])
    ax_main.set_title(
        "Transition Matrix — 2013 → 2024\n(log scale)",
        fontsize=_FONT["title"], fontweight="bold",
    )

    # Annotate cells — millions for large values, exact for small
    for i in range(3):
        for j in range(3):
            val = M[i, j]
            if val >= 1_000_000:
                label = f"{val/1e6:.1f}M"
            elif val >= 1_000:
                label = f"{val/1e3:.1f}K"
            else:
                label = f"{val:,}"
            # Dark cells (high values in log space) get white text
            log_val = np.log10(max(val, 1))
            log_max = np.log10(max(M.max(), 1))
            text_color = "white" if log_val > log_max * 0.6 else "black"
            ax_main.text(j, i, label,
                         ha="center", va="center",
                         fontsize=_FONT["annot"], color=text_color,
                         fontweight="bold")

    # --- Inset: built-up transitions only (column 2 of M = 2024 built-up) ---
    built_up_transitions = {
        "other\n→ built": M[0, 2],
        "veg\n→ built":   M[1, 2],
        "built\n→ built": M[2, 2],
    }
    bar_labels = list(built_up_transitions.keys())
    bar_values = list(built_up_transitions.values())
    bar_colors = [CHANGE_COLORS[2], CHANGE_COLORS[1], CLASS_COLORS[2]]

    bars = ax_inset.bar(bar_labels, bar_values, color=bar_colors, edgecolor="white")
    ax_inset.set_title(
        "Built-up transitions\n(linear scale)",
        fontsize=_FONT["subtitle"], fontweight="bold",
    )
    ax_inset.set_ylabel("pixels", fontsize=_FONT["axis"])
    ax_inset.tick_params(axis="x", labelsize=_FONT["annot"])
    ax_inset.tick_params(axis="y", labelsize=_FONT["small"])
    ax_inset.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else str(int(x)))
    )

    for bar, val in zip(bars, bar_values):
        ax_inset.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:,}",
            ha="center", va="bottom",
            fontsize=_FONT["small"], fontweight="bold",
        )

    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Directional polar bar chart
# ---------------------------------------------------------------------------

def plot_directional_polar(directional_result: dict) -> plt.Figure:
    """
    Plot urban growth area (km²) as a polar bar chart by compass direction.

    Improvements:
    - Radial axis shows km² instead of raw pixel counts
    - Continuous viridis colourmap (smooth, no abrupt jumps)
    - Colourbar added to show km² scale
    - Bar labels show area in km² (more interpretable than pixel counts)
    """
    direction_order = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]

    values_km2 = np.array(
        [directional_result[d]["area_km2"] for d in direction_order],
        dtype=np.float64,
    )

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    width  = 2 * np.pi / 8

    dominant_dir = direction_order[int(np.argmax(values_km2))]

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, polar=True)

    # Continuous viridis colourmap normalised across the value range
    norm   = plt.Normalize(vmin=values_km2.min(), vmax=values_km2.max())
    cmap   = plt.cm.viridis
    colors = cmap(norm(values_km2))

    bars = ax.bar(
        angles, values_km2,
        width=width,
        align="center",
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.92,
    )

    # Radial axis in km²
    ax.set_rlabel_position(22.5)   # offset labels from the East bar
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.2f}")
    )
    ax.tick_params(axis="y", labelsize=_FONT["small"])

    ax.set_xticks(angles)
    ax.set_xticklabels(direction_order, fontsize=_FONT["legend"])
    ax.set_title(
        f"Urban Growth Direction — 2013 → 2024\nDominant: {dominant_dir}",
        fontsize=_FONT["title"], fontweight="bold", pad=22,
    )

    # Bar labels in km²
    for angle, val, bar in zip(angles, values_km2, bars):
        if val > 0:
            ax.text(
                angle,
                bar.get_height() * 1.08,
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=_FONT["small"], color="black",
            )

    # Colourbar to show km² scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.12, fraction=0.04, shrink=0.6)
    cbar.set_label("area (km²)", fontsize=_FONT["axis"])
    cbar.ax.tick_params(labelsize=_FONT["small"])

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_all_figures(
    ndvi_2000: np.ndarray,
    map_2013: np.ndarray,
    map_2024: np.ndarray,
    change_map: np.ndarray,
    directional_result: dict,
    transition_stats: dict = None,
    aoi_path: str = None,
    transform=None,
) -> dict:
    """
    Generate all six pipeline visualisation figures.

    Parameters
    ----------
    ndvi_2000          : (H, W) float32
    map_2013           : (H, W) int32
    map_2024           : (H, W) int32
    change_map         : (H, W) int8
    directional_result : dict from compute_directional_growth
    transition_stats   : dict from compute_transition_stats (for annotations)
    aoi_path           : path to AOI GeoJSON for boundary overlay
    transform          : rasterio Affine from aligned scene profile

    Returns
    -------
    dict[str → matplotlib.Figure]
        Keys: "ndvi_2000", "map_2013", "map_2024",
              "change_map", "transition_matrix", "directional"
    """
    return {
        "ndvi_2000":  plot_ndvi_map(
                          ndvi_2000, aoi_path=aoi_path, transform=transform),
        "map_2013":   plot_classified_map(
                          map_2013, year="2013", aoi_path=aoi_path, transform=transform),
        "map_2024":   plot_classified_map(
                          map_2024, year="2024", aoi_path=aoi_path, transform=transform),
        "change_map": plot_change_map(
                          change_map,
                          transition_stats=transition_stats,
                          directional_result=directional_result,
                          aoi_path=aoi_path,
                          transform=transform),
        "transition_matrix": plot_transition_matrix(map_2013, map_2024),
        "directional":       plot_directional_polar(directional_result),
    }