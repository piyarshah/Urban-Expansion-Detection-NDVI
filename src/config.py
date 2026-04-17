"""
config.py — Centralised configuration for S11/S12

All hyperparameters, seeds, and split ratios live here.
Import from this module; never hardcode values in model.py or run_pipeline.py.
"""

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Train / validation split (S11)
# ---------------------------------------------------------------------------
VAL_SPLIT = 0.20          # fraction of training data held out for validation

# ---------------------------------------------------------------------------
# Random Forest hyperparameters (S11)
#
# Rationale for each non-default value:
#
#   n_estimators = 300
#       Enough trees to stabilise class probability estimates across the
#       feature space. The marginal gain plateaus around 200-300 for tabular
#       data of this dimensionality; beyond 400 the runtime cost dominates.
#
#   max_depth = 20
#       Prevents the model from memorising the hard spectral thresholds used
#       to generate proxy labels. Labels are noisy by construction; deep trees
#       would overfit the threshold artefacts rather than learning the
#       underlying spectral signal. 20 layers is sufficient to capture
#       meaningful interactions between the 4 features without memorising
#       individual training pixels.
#
#   min_samples_leaf = 10
#       Enforces local averaging — each leaf must represent at least 10 pixels.
#       This smooths the decision boundary in regions where NDVI and NDBI have
#       overlapping distributions (e.g. sparse vegetation vs bare urban soil).
#
#   max_features = "sqrt"
#       With 3 features, sqrt(3) ≈ 1.7, rounded to 2 features considered per
#       split. Fixing this explicitly ensures reproducibility across sklearn
#       versions and prevents the default from changing silently.
#
#   class_weight = None
#       Training set is already balanced by S10 stratified sampling.
#       Applying class weights on a balanced set would reintroduce bias.
#
#   n_jobs = -1
#       Use all available cores. Tree fitting is embarrassingly parallel.
# ---------------------------------------------------------------------------
RF_PARAMS = {
    "n_estimators":    300,
    "max_depth":       20,
    "min_samples_leaf": 10,
    "max_features":    "sqrt",
    "class_weight":    None,
    "n_jobs":          -1,
    "random_state":    RANDOM_SEED,
}

# ---------------------------------------------------------------------------
# Class metadata (mirrors labels.py constants — kept separate to avoid
# importing labels.py into config, which would create a circular dependency
# if config is ever imported by labels.py itself)
# ---------------------------------------------------------------------------
CLASS_LABELS = [-1, 0, 1, 2]
CLASS_NAMES  = {-1: "invalid", 0: "other", 1: "vegetation", 2: "built-up"}
TRAIN_CLASSES = [0, 1, 2]   # excludes -1

# Feature column order — must match stack_features in features.py exactly.
# Raw bands only. ndvi and ndbi are EXCLUDED despite being in the scene dict.
# Labels are derived from ndvi/ndbi thresholds — including them as features
# causes circular supervision and trivial 100% validation accuracy.
FEATURE_NAMES = ["red", "nir", "swir1"]

# Confidence threshold for S13 change detection.
# A pixel transition is only flagged as change if the predicted class
# probability exceeds this value in BOTH years. Below this threshold the
# prediction is treated as uncertain and excluded from the change map.
CONFIDENCE_THRESHOLD = 0.60
