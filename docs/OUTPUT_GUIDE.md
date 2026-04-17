# Output Guide

This document describes every file produced by the pipeline, where to find it, how to interpret it, and any caveats that apply.

After a successful run, the `outputs/` directory has the following structure:

```
outputs/
├── rasters/
│   ├── classified_2013.tif
│   ├── classified_2024.tif
│   └── change_map.tif
├── figures/
│   ├── ndvi_2000.png
│   ├── map_2013.png
│   ├── map_2024.png
│   ├── change_map.png
│   ├── transition_matrix.png
│   └── directional.png
├── tables/
│   ├── transition_matrix.csv
│   └── directional.csv
├── logs/
│   └── run.log
└── rf_model.joblib
```

---

## Rasters

All rasters are saved as LZW-compressed GeoTIFFs in EPSG:32643 (UTM Zone 43N) at 30 m resolution, aligned to the 2013 reference grid.

### `classified_2013.tif` / `classified_2024.tif`

Land cover classification maps for each year.

| Pixel value | Class | Description |
|---|---|---|
| −1 | invalid | Cloud, shadow, or outside AOI |
| 0 | other | Bare soil, mixed land, water, unclassified |
| 1 | vegetation | Vegetated surfaces (NDVI-dominant) |
| 2 | built-up | Urban, impervious surfaces (NDBI-dominant) |

**dtype:** int8  
**nodata:** −1  
**CRS:** EPSG:32643  

To open in Python:
```python
import rasterio
with rasterio.open("outputs/rasters/classified_2013.tif") as src:
    arr = src.read(1)
```

To open in QGIS: File → Open Raster. Use Symbology → Paletted/Unique values and assign: −1 black, 0 grey, 1 green, 2 red.

### `change_map.tif`

Pixel-wise land cover transition map from 2013 to 2024. Only pixels passing both the validity gate (valid in both years) and the confidence gate (max class probability ≥ 0.60 in both years) receive a change code.

| Pixel value | Transition | Description |
|---|---|---|
| −1 | invalid | Masked, low-confidence, or outside AOI |
| 0 | no change | Same class in both years |
| 1 | vegetation → built-up | Direct green space conversion |
| 2 | other → built-up | Bare soil / mixed land converted |

**dtype:** int8  
**nodata:** −1  
**CRS:** EPSG:32643  

---

## Figures

All figures are saved at 300 dpi with tight bounding boxes, suitable for publication or report inclusion.

### `ndvi_2000.png`

NDVI raster for the year 2000 using the RdYlGn diverging colourmap. Red = low/negative NDVI (bare, built-up), green = high NDVI (dense vegetation). NDVI is strictly clipped to [−1, 1]. NaN pixels render as white. The AOI boundary is overlaid as a thin dark outline.

**Use:** Baseline vegetation cover reference. Compare qualitatively against the 2013 and 2024 classification maps.

### `map_2013.png` / `map_2024.png`

Classified land cover maps using the fixed colour scheme: grey = other, green = vegetation, red = built-up, black = invalid. The AOI boundary is overlaid. Colour values are consistent between both years — any apparent colour difference reflects actual land cover change, not visualisation artefact.

**Use:** Direct side-by-side comparison of land cover extent between years.

### `change_map.png`

Transition map with annotated total new built-up area (km²) and dominant growth direction. Light grey = no change, blue = vegetation converted to built-up, orange = other converted to built-up, black = invalid.

**Interpretation note:** The "no change" grey dominates spatially — this is expected. The meaningful signal is in the blue and orange pixels, which represent genuine urban expansion. The annotations in the top-left (total area) and bottom-left (dominant direction) summarise the key findings directly on the figure.

### `transition_matrix.png`

Two-panel figure. Left: 3×3 heatmap (log scale) showing pixel counts for every class-to-class transition pair. Rows = 2013 class, columns = 2024 class. Log scale is used because the no-change diagonal dominates by several orders of magnitude, which would suppress built-up transitions on a linear scale. Cell annotations adapt to value magnitude (M = millions, K = thousands). Right: Linear-scale bar chart showing only the three built-up transitions (other→built, veg→built, built→built) for visual emphasis.

**Key cells:**
- `(1, 2)` — vegetation → built-up: direct green space loss to urbanism
- `(0, 2)` — other → built-up: peripheral land conversion
- `(2, 2)` — built → built: stable built-up (should be the largest built-up cell)

### `directional.png`

Polar bar chart showing urban growth area (km²) distributed across 8 compass directions from the AOI centroid. Bars are coloured by magnitude using the viridis colourmap (dark purple = low, yellow = high). The radial axis is labelled in km². The dominant direction appears in the title. A colourbar on the right provides the km² scale reference.

**Interpretation:** A dominant direction indicates the primary axis of urban expansion. Balanced bars indicate diffuse, multi-directional growth. Compare with the change map to verify the directional result is geographically plausible.

---

## Tables

### `transition_matrix.csv`

| Column | Description |
|---|---|
| `transition` | Transition type: `veg_to_built`, `other_to_built`, `no_change` |
| `pixels` | Pixel count |
| `area_km2` | Area in km² (pixels × 0.0009 km² per 30 m pixel) |

```
transition,pixels,area_km2
veg_to_built,45231,40.708
other_to_built,23891,21.502
no_change,9821456,8839.310
```

### `directional.csv`

| Column | Description |
|---|---|
| `direction` | Compass direction: E, NE, N, NW, W, SW, S, SE |
| `pixels` | New built-up pixel count in this direction |
| `area_km2` | Area in km² |
| `pct` | Percentage of total new built-up pixels |

```
direction,pixels,area_km2,pct
E,12341,11.107,17.8
NE,8923,8.031,12.9
...
```

---

## Logs

### `run.log`

Appended on each run. Contains timestamps, stage completion events, dataset sizes, model parameters, and key quantitative results. Format:

```
2025-01-15 14:32:01 | INFO     | Pipeline run started  2025-01-15 14:32:01
2025-01-15 14:32:01 | INFO     | --- Configuration ---
2025-01-15 14:32:01 | INFO     |   confidence_threshold         0.6
2025-01-15 14:32:01 | INFO     |   n_estimators                 300
...
2025-01-15 14:45:22 | INFO     | [S13] Change detection complete  |  new_built_up_pixels=69122
2025-01-15 14:45:23 | INFO     | [S14] Quantification complete  |  total_growth_km2=62.2098
```

The log file is never overwritten — each run appends a dated block. This allows comparison of results across runs when thresholds or parameters are changed.

---

## Model

### `rf_model.joblib`

Serialised `sklearn.ensemble.RandomForestClassifier` trained on the 2013 stratified sample. Load with:

```python
from src.io_model import load_model
model = load_model("outputs/rf_model.joblib")

# Predict on new data (must be shape (N, 3), columns: [red, nir, swir1])
y_pred = model.predict(X_new)
proba  = model.predict_proba(X_new)
```

**Model class order:** `model.classes_` is `[0, 1, 2]` = `[other, vegetation, built-up]`. `proba[:, 2]` is the built-up probability for each pixel.

The model is trained on 2013 data only. It is applied to 2013, 2024, and 2000 scenes under the assumption that the spectral-to-class mapping is stable across years — reasonable for Collection 2 SR data with consistent atmospheric correction.
