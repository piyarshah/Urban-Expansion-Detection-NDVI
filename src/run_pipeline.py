import numpy as np
from pathlib import Path

from src.data_loading import load_scene
from src.preprocessing import scale_scene, mask_scene, clip_scene, align_scene
from src.features import compute_indices, stack_features
from src.labels import generate_labels, get_valid_mask, label_summary, THRESHOLDS
from src.training import build_training_set, training_summary
from src.model import train_model, predict_full
from src.evaluation import print_metrics, print_feature_importances
from src.io_model import save_model
from src.change_detection import compute_change_map, change_map_summary
from src.quantification import compute_transition_stats, print_transition_stats
from src.directional import compute_directional_growth, print_directional_growth
from src.visualisation import build_all_figures
from src.export import export_all
from src.logger import setup_logger, log_pipeline_start, log_stage
from src.config import CONFIDENCE_THRESHOLD, RF_PARAMS


def main():
    # --- Logger ---
    logger = setup_logger("outputs/logs/run.log")
    log_pipeline_start(logger, {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "n_estimators":         RF_PARAMS["n_estimators"],
        "max_depth":            RF_PARAMS["max_depth"],
        "min_samples_leaf":     RF_PARAMS["min_samples_leaf"],
        "random_seed":          RF_PARAMS["random_state"],
    })

    base_path  = Path("/Users/piyashah/Downloads/ndvi-urban-expansion")
    data_path  = base_path / "data" / "raw"
    aoi_path   = base_path / "data" / "aoi" / "surat_aoi_simple.geojson"
    output_dir = base_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_2013 = data_path / "LC08_L2SP_148045_20131103_20200912_02_T1"
    scene_2024 = data_path / "LC08_L2SP_148045_20241117_20241126_02_T1"
    scene_2000 = data_path / "LT05_L2SP_148045_20000217_20200907_02_T1"

    # -------------------------------------------------------------------------
    # S2 — Data Loading
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("S2 — Loading raw scenes")
    print("=" * 60)

    print("Loading 2013...")
    s2013 = load_scene(scene_2013)
    print("Loading 2024...")
    s2024 = load_scene(scene_2024)
    print("Loading 2000...")
    s2000 = load_scene(scene_2000)

    print("\n--- Raw shape check ---")
    for year, s in [("2013", s2013), ("2024", s2024), ("2000", s2000)]:
        print(f"  {year}: {s['red'].shape}")

    # -------------------------------------------------------------------------
    # S3 — Radiometric Scaling
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("S3 — Radiometric scaling")
    print("=" * 60)

    s2013 = scale_scene(s2013)
    s2024 = scale_scene(s2024)
    s2000 = scale_scene(s2000)

    for year, s in [("2013", s2013), ("2024", s2024), ("2000", s2000)]:
        print(f"  {year} red  min/max: {s['red'].min():.4f} / {s['red'].max():.4f}")
    print(f"  dtype: {s2013['red'].dtype}")

    # -------------------------------------------------------------------------
    # S4 — QA Masking
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("S4 — QA masking (clouds + cloud shadows)")
    print("=" * 60)

    s2013 = mask_scene(s2013)
    s2024 = mask_scene(s2024)
    s2000 = mask_scene(s2000)

    def _valid_pct(arr):
        total = arr.size
        valid = int(np.sum(~np.isnan(arr)))
        return f"{valid:,}/{total:,}  ({100 * valid / total:.1f}% valid)"

    for year, s in [("2013", s2013), ("2024", s2024), ("2000", s2000)]:
        print(f"  {year} valid pixels: {_valid_pct(s['red'])}")

    # -------------------------------------------------------------------------
    # S5 — AOI Clipping
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("S5 — AOI clipping")
    print("=" * 60)

    s2013 = clip_scene(s2013, str(aoi_path))
    s2024 = clip_scene(s2024, str(aoi_path))
    s2000 = clip_scene(s2000, str(aoi_path))

    for year, s in [("2013", s2013), ("2024", s2024), ("2000", s2000)]:
        print(f"  {year} clipped shape: {s['red'].shape}  "
              f"transform origin: ({s['profile']['transform'].c:.0f}, {s['profile']['transform'].f:.0f})")

    # -------------------------------------------------------------------------
    # S6 — Spatial Alignment (reference = 2013)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("S6 — Spatial alignment (reference: 2013)")
    print("=" * 60)

    s2024 = align_scene(s2024, reference=s2013)
    s2000 = align_scene(s2000, reference=s2013)

    shapes = {s2013["red"].shape, s2024["red"].shape, s2000["red"].shape}
    assert len(shapes) == 1, f"Shape mismatch after alignment: {shapes}"
    raster_shape = s2013["red"].shape
    print(f"  Aligned shape (all scenes): {raster_shape}")
    print("  Alignment verified.")

    # -------------------------------------------------------------------------
    # S7 — Spectral Indices
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("S7 — Spectral indices (NDVI, NDBI)")
    print("=" * 60)

    s2013 = compute_indices(s2013)
    s2024 = compute_indices(s2024)
    s2000 = compute_indices(s2000)

    def _band_stats(arr, name):
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            print(f"    {name}: all NaN")
            return
        print(f"    {name}: min={valid.min():.4f}  max={valid.max():.4f}  "
              f"mean={valid.mean():.4f}  NaN%={100 * np.isnan(arr).mean():.1f}%")

    for year, s in [("2013", s2013), ("2024", s2024), ("2000", s2000)]:
        print(f"\n  {year}:")
        _band_stats(s["ndvi"], "NDVI")
        _band_stats(s["ndbi"], "NDBI")

    # -------------------------------------------------------------------------
    # S8 — Feature Stacking
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("S8 — Feature stacking [red, nir, swir1]")
    print("=" * 60)

    X_2013 = stack_features(s2013)
    X_2024 = stack_features(s2024)
    X_2000 = stack_features(s2000)

    for year, X in [("2013", X_2013), ("2024", X_2024), ("2000", X_2000)]:
        print(f"  X_{year}: {X.shape} | dtype: {X.dtype}")

    assert X_2013.shape == X_2024.shape == X_2000.shape, "Feature matrix shape mismatch"
    print("  Feature matrices consistent across scenes.")

    # -------------------------------------------------------------------------
    # S9 — Proxy Label Generation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("S9 — Proxy label generation")
    print("=" * 60)

    print(f"  Thresholds: {THRESHOLDS}\n")

    # Labels derived from scene ndvi/ndbi — NOT the feature matrix.
    y_2013 = generate_labels(s2013, thresholds=THRESHOLDS)
    y_2024 = generate_labels(s2024, thresholds=THRESHOLDS)
    y_2000 = generate_labels(s2000, thresholds=THRESHOLDS)

    label_summary(y_2013, tag="2013")
    label_summary(y_2024, tag="2024")
    label_summary(y_2000, tag="2000")

    # -------------------------------------------------------------------------
    # S10 — Training Dataset Construction
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("S10 — Stratified training dataset (from 2013 labels)")
    print("=" * 60)

    X_train, y_train = build_training_set(X_2013, y_2013, seed=42)
    training_summary(X_train, y_train)

    # -------------------------------------------------------------------------
    # S11 — Model Training
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("S11 — Random Forest training")
    print("=" * 60)

    model, val_metrics = train_model(X_train, y_train)

    # Print validation report
    print_metrics(val_metrics, tag="Validation (held-out 20%)")

    # Print feature importances
    print_feature_importances(model, tag="2013 training set")

    # Serialise model for S12 decoupling and reproducibility
    model_path = output_dir / "rf_model.joblib"
    save_model(model, str(model_path))

    # -------------------------------------------------------------------------
    # S12 — Full-Raster Prediction
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("S12 — Full-raster classification")
    print("=" * 60)

    print("\n  Predicting 2013...")
    map_2013, proba_2013 = predict_full(model, X_2013, raster_shape)

    print("\n  Predicting 2024...")
    map_2024, proba_2024 = predict_full(model, X_2024, raster_shape)

    print("\n  Predicting 2000...")
    map_2000, proba_2000 = predict_full(model, X_2000, raster_shape)

    # Verify output shape and encoding
    label_names = {-1: "invalid", 0: "other", 1: "vegetation", 2: "built-up"}
    for year, m, p in [
        ("2013", map_2013, proba_2013),
        ("2024", map_2024, proba_2024),
        ("2000", map_2000, proba_2000),
    ]:
        unique, counts = np.unique(m, return_counts=True)
        print(f"\n  {year} class map:  shape={m.shape}  dtype={m.dtype}")
        for u, c in zip(unique, counts):
            print(f"    {int(u):>2}  {label_names.get(int(u), '?'):<12}  "
                  f"{int(c):>10,}  ({100 * int(c) / m.size:.1f}%)")

        # Confidence summary: mean predicted-class probability for valid pixels
        valid_px  = m != -1
        pred_cls  = m[valid_px]                        # (n_valid,)
        proba_2d  = p[valid_px]                        # (n_valid, 3)
        # pick the probability of the predicted class for each valid pixel
        conf_vals = proba_2d[np.arange(len(pred_cls)), pred_cls]
        print(f"  {year} confidence: mean={conf_vals.mean():.3f}  "
              f"min={conf_vals.min():.3f}  "
              f"pct>=0.60: {100*(conf_vals>=0.60).mean():.1f}%  "
              f"pct>=0.80: {100*(conf_vals>=0.80).mean():.1f}%")
        print(f"  {year} proba map:  shape={p.shape}  dtype={p.dtype}")

    print("\nPipeline complete through S12. Ready for S13 (change detection).")
    log_stage(logger, "S12", "Full-raster classification complete",
              valid_2013=int(np.sum(map_2013 != -1)),
              valid_2024=int(np.sum(map_2024 != -1)))

    # -------------------------------------------------------------------------
    # S13 — Change Detection
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("S13 — Confidence-aware change detection (2013 → 2024)")
    print("=" * 60)

    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    change_map = compute_change_map(
        map_2013, map_2024,
        proba_2013, proba_2024,
        threshold=CONFIDENCE_THRESHOLD,
    )
    change_map_summary(change_map, tag="2013 → 2024")

    total_new_px = int(np.sum((change_map == 1) | (change_map == 2)))
    log_stage(logger, "S13", "Change detection complete",
              new_built_up_pixels=total_new_px)

    # -------------------------------------------------------------------------
    # S14 — Quantification
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("S14 — Transition area quantification")
    print("=" * 60)

    transition_stats = compute_transition_stats(
        change_map,
        transform=s2013["profile"]["transform"],
    )
    print_transition_stats(transition_stats, tag="2013 → 2024")

    total_growth_km2 = (
        transition_stats["veg_to_built"]["area_km2"] +
        transition_stats["other_to_built"]["area_km2"]
    )
    log_stage(logger, "S14", "Quantification complete",
              total_growth_km2=f"{total_growth_km2:.4f}")

    # -------------------------------------------------------------------------
    # S15 — Directional Expansion
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("S15 — Directional urban growth analysis")
    print("=" * 60)

    directional_result = compute_directional_growth(
        change_map,
        transform=s2013["profile"]["transform"],
        aoi_path=str(aoi_path),
        pixel_area_km2=transition_stats["_meta"]["pixel_area_km2"],
    )
    print_directional_growth(directional_result, tag="2013 → 2024")

    direction_order = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    dominant_dir = max(direction_order,
                       key=lambda d: directional_result[d]["pixels"])
    log_stage(logger, "S15", "Directional analysis complete",
              dominant_direction=dominant_dir,
              total_growth_km2=f"{directional_result['_meta']['total_growth_km2']:.4f}")

    # -------------------------------------------------------------------------
    # S16 — Visualisation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("S16 — Generating visualisations")
    print("=" * 60)

    figures = build_all_figures(
        ndvi_2000=s2000["ndvi"],
        map_2013=map_2013,
        map_2024=map_2024,
        change_map=change_map,
        directional_result=directional_result,
        transition_stats=transition_stats,
        aoi_path=str(aoi_path),
        transform=s2013["profile"]["transform"],
    )
    print(f"  Generated {len(figures)} figures: {list(figures.keys())}")
    log_stage(logger, "S16", "Visualisation complete", figures=len(figures))

    # -------------------------------------------------------------------------
    # S17 — Export Outputs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("S17 — Exporting outputs")
    print("=" * 60)

    output_dirs = export_all(
        map_2013=map_2013,
        map_2024=map_2024,
        change_map=change_map,
        profile=s2013["profile"],
        transition_stats=transition_stats,
        directional_result=directional_result,
        figures=figures,
        base=str(output_dir),
    )
    log_stage(logger, "S17", "All outputs exported",
              rasters=str(output_dirs["rasters"]),
              figures=str(output_dirs["figures"]),
              tables=str(output_dirs["tables"]))

    print("\n" + "=" * 60)
    print("Pipeline complete — S2 through S17.")
    print("=" * 60)
    logger.info("Pipeline finished successfully.")

    return {
        "scenes":    {"2013": s2013, "2024": s2024, "2000": s2000},
        "features":  {"2013": X_2013, "2024": X_2024, "2000": X_2000},
        "labels":    {"2013": y_2013, "2024": y_2024, "2000": y_2000},
        "training":  {"X_train": X_train, "y_train": y_train},
        "model":     model,
        "val_metrics": val_metrics,
        "maps":      {"2013": map_2013, "2024": map_2024, "2000": map_2000},
        "probas":    {"2013": proba_2013, "2024": proba_2024, "2000": proba_2000},
        "raster_shape":      raster_shape,
        "profile":           s2013["profile"],
        "change_map":        change_map,
        "transition_stats":  transition_stats,
        "directional":       directional_result,
        "figures":           figures,
        "output_dirs":       output_dirs,
    }


if __name__ == "__main__":
    main()