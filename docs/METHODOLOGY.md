# Methodology

This document describes the technical decisions made at each stage of the pipeline. It is intended as a reference for replication, peer review, or extension of this work.

---

## 1. Data

### Sensor and product selection

Three Landsat scenes were selected for the WRS-2 tile Path 148 / Row 045, which covers Surat and its surroundings:

| Year | Satellite | Acquisition date | Product |
|---|---|---|---|
| 2000 | Landsat 5 TM | 17 February 2000 | Collection 2 L2 SR |
| 2013 | Landsat 8 OLI | 03 November 2013 | Collection 2 L2 SR |
| 2024 | Landsat 8 OLI | 17 November 2024 | Collection 2 L2 SR |

November acquisitions were chosen for 2013 and 2024 to minimise seasonal NDVI variation. The 2000 scene is from February due to cloud availability constraints; users should apply caution when interpreting cross-season comparisons.

USGS Collection 2 Level-2 Surface Reflectance products apply sensor-specific atmospheric correction (LaSRC for Landsat 8, LEDAPS for Landsat 5), making the values physically comparable in reflectance units across sensors.

### Bands used

| Band name | Landsat 5 | Landsat 8 | Wavelength |
|---|---|---|---|
| Red | SR_B3 | SR_B4 | ~0.63–0.69 µm |
| NIR | SR_B4 | SR_B5 | ~0.76–0.90 µm |
| SWIR1 | SR_B5 | SR_B6 | ~1.55–1.75 µm |
| QA_PIXEL | QA_PIXEL | QA_PIXEL | — |

---

## 2. Preprocessing

### Radiometric scaling (S3)

Raw DN values are converted to surface reflectance using USGS Collection 2 constants:

```
SR = DN × 0.0000275 + (−0.2)
```

No value clipping is applied. Fill values (DN = 0 → SR = −0.2) are physically impossible and are removed by the subsequent QA mask. Atmospheric correction artefacts that push SR slightly above 1.0 are retained — clipping would bias spectral index calculations. Values are cast to float32 before scaling to prevent integer overflow.

### Cloud and shadow masking (S4)

QA_PIXEL bitmask interpretation follows the USGS Collection 2 specification:

- Bit 3: Cloud (1 = cloud present)
- Bit 4: Cloud Shadow (1 = shadow present)

Masked pixels are set to `np.nan`. The mask is derived from a single boolean array applied identically to all spectral bands, ensuring spatial consistency. The QA band is preserved as integer for potential downstream reuse.

### AOI clipping (S5)

The AOI polygon (`surat_aoi_simple.geojson`) is loaded with GeoPandas and reprojected to the raster CRS (EPSG:32643) if needed. `rasterio.mask.mask` is used with `crop=True` to reduce array size. Spectral bands use a sentinel nodata value of −9999 (physically impossible SR, not NaN) during the MemoryFile write/clip operation to avoid the RuntimeWarning raised by rasterio when NaN is used as nodata. Sentinel pixels are converted back to NaN after clipping. QA band nodata is set to `None` because QA=0 represents a valid bitmask state (all flags clear) and must not be treated as fill.

### Spatial alignment (S6)

The 2013 scene is used as the spatial reference. All other scenes are reprojected to match its affine transform, CRS, width, and height using `rasterio.warp.reproject`. Spectral bands use bilinear resampling; the QA band uses nearest-neighbour to preserve bitmask integer values.

NaN propagation during warping: because `rasterio.warp.reproject` does not reliably honour `np.nan` as nodata, masked pixels are converted to the sentinel value −9999 before warping and restored to NaN after. After resampling, the QA-derived valid mask is re-applied to all spectral bands to eliminate bilinear bleed artefacts at cloud/clear boundaries.

---

## 3. Spectral Indices (S7)

### NDVI

```
NDVI = (NIR − RED) / (NIR + RED + ε)
```

Where ε = 1×10⁻⁶ prevents division by zero without meaningfully affecting the ratio. NDVI is clipped to [−1, 1] post-computation to remove physically impossible atmospheric correction artefacts (values > 1 observed in the 2024 Landsat 8 scene). NaN propagation is guaranteed by NumPy arithmetic.

### NDBI

```
NDBI = (SWIR1 − NIR) / (SWIR1 + NIR + ε)
```

Positive NDBI indicates more SWIR1 reflectance than NIR — the spectral signature of impervious surfaces, bare soil, and built-up material. NDBI is not clipped; its physical range is inherently [−1, 1] by construction.

---

## 4. Feature Engineering (S8)

The feature matrix is `[red, nir, swir1]` — raw reflectance bands only.

Derived indices (NDVI, NDBI) are computed and stored in the scene dictionary for use in label generation but are **excluded** from the model feature matrix. This separation is critical: including the indices as model features alongside the thresholds that were used to generate labels from them produces circular supervision, where the Random Forest trivially learns to reproduce the exact threshold rules rather than a generalisable spectral decision boundary. With raw bands as features, the RF must approximate the label boundaries through learned spectral relationships, yielding honest validation accuracy (typically 85–95%) and better cross-sensor generalisation.

All bands are flattened in row-major (C) order consistently across all scenes to preserve pixel-wise spatial correspondence.

---

## 5. Proxy Label Generation (S9)

Labels are assigned using hard thresholds in (NDVI, NDBI) spectral index space:

| Class | Condition |
|---|---|
| vegetation (1) | NDVI ≥ 0.40 AND NDBI ≤ 0.05 |
| built-up (2) | NDBI ≥ 0.05 AND NDVI ≤ 0.30 |
| other (0) | everything else that is valid and unambiguous |
| invalid (−1) | NaN in either index, or conflict (both conditions satisfied) |

Conflict pixels — those satisfying both vegetation and built-up conditions simultaneously — are discarded rather than assigned to either class. These arise from mixed pixels at class boundaries and index instability near the thresholds. Removing them reduces label noise.

The same thresholds are applied identically to all three years. Cross-year threshold consistency is mandatory for valid change detection; if the labelling rule changes between years, the change map reflects threshold drift rather than real land cover change.

**Threshold calibration.** Initial thresholds (T_b_high=0.10, T_v_low=0.20) produced only 13,237 built-up pixels in 2013 (0.07% of the AOI) versus 896,293 in 2000 — a physically implausible collapse explained by Surat's compressed NDBI distribution under November atmospheric conditions. Thresholds were relaxed to T_b_high=0.05, T_v_low=0.30 to restore physically plausible class proportions.

---

## 6. Model Training (S11)

### Algorithm

Random Forest classifier (`sklearn.ensemble.RandomForestClassifier`) with the following hyperparameters:

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 300 | Sufficient to stabilise probability estimates; marginal gain plateaus beyond ~200 |
| `max_depth` | 20 | Prevents memorisation of proxy threshold boundaries |
| `min_samples_leaf` | 10 | Enforces local averaging; smooths decision boundary in mixed-signal regions |
| `max_features` | sqrt | Standard RF variance reduction; sqrt(3) ≈ 2 features per split |
| `class_weight` | None | Training set is balanced by stratified sampling in S10 |
| `n_jobs` | −1 | Use all available CPU cores |

### Validation

An 80/20 stratified train/validation split is performed before fitting. Class balance is verified in both splits. The model is fitted on the 80% training portion only; metrics are reported on the held-out 20% validation set. Cross-validation is not used — at this scale (430K+ training samples) it is computationally unnecessary and single-split variance is low.

---

## 7. Prediction (S12)

The fitted model is applied independently to the full feature matrix of each year (2000, 2013, 2024). Pixels containing any NaN feature value are excluded from prediction and filled with −1 in the output raster. Two outputs are produced per year:

- **class_map** `(H, W)` int32: argmax class label per pixel
- **proba_map** `(H, W, 3)` float32: predicted class probabilities

The `proba_map` enables confidence-aware change detection in S13.

---

## 8. Change Detection (S13)

A pixel transition from 2013 to 2024 is flagged as change only if it passes both:

1. **Validity gate:** pixel must be valid (class ≠ −1) in both 2013 and 2024
2. **Confidence gate:** max predicted class probability must be ≥ 0.60 in both years

The confidence gate prevents low-certainty predictions near class boundaries from generating spurious change signals. The threshold of 0.60 (versus 0.50 argmax) excludes the most ambiguous predictions while retaining 99%+ of the valid pixel population.

Tracked transitions:

| Code | Transition | Interpretation |
|---|---|---|
| 1 | vegetation → built-up | Direct conversion of green space to urban |
| 2 | other → built-up | Conversion of bare soil / mixed land to urban |
| 0 | no change | Same class in both years |
| −1 | invalid | Failed validity or confidence gate |

Transitions away from built-up are not tracked — the analysis scope is urban expansion only.

---

## 9. Directional Analysis (S15)

New built-up pixels (change codes 1 and 2) are binned into 8 compass directions (N, NE, E, SE, S, SW, W, NW) relative to the AOI centroid. Pixel coordinates are obtained by converting raster row/column indices to projected (x, y) via the rasterio affine transform. Direction angles use standard mathematical convention (East = 0°, counter-clockwise), which aligns correctly with UTM Zone 43N where x increases eastward and y increases northward.

---

## 10. Limitations

- Labels are derived from spectral thresholds, not field validation. Accuracy figures should be interpreted as model-internal consistency, not ground-truth accuracy.
- The 2000 Landsat 5 scene is from February versus November for 2013/2024. Phenological differences (dry season vs. post-monsoon) affect NDVI values and may confound 2000→2013 comparisons.
- The 30 m spatial resolution limits detection of small-scale urban features (narrow roads, individual buildings). Sub-pixel mixing is common at class boundaries.
- Atmospheric correction differences between LEDAPS (Landsat 5) and LaSRC (Landsat 8) introduce residual inter-sensor biases, particularly in NDBI values.
- Cloud masking is based on the QA_PIXEL band only. Thin cirrus or haze may pass the mask and introduce noise in the classification.
