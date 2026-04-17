"""
io_model.py — Model Serialisation (S11/S12)

Encapsulates joblib save/load for the trained RandomForestClassifier.
Keeping this separate from model.py means model.py has zero file I/O.

Public API:
    save_model(model, path)  → None
    load_model(path)         → RandomForestClassifier
"""

from pathlib import Path
import joblib


def save_model(model, path: str) -> None:
    """
    Serialise a fitted model to disk using joblib.

    Parameters
    ----------
    model : fitted sklearn estimator
    path  : destination file path (e.g. "outputs/rf_model.joblib")
             Parent directories are created automatically.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"  Model saved → {out_path.resolve()}")


def load_model(path: str):
    """
    Deserialise a model from disk.

    Parameters
    ----------
    path : path to a .joblib file produced by save_model

    Returns
    -------
    Fitted sklearn estimator

    Raises
    ------
    FileNotFoundError if path does not exist.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Model file not found: {in_path.resolve()}")
    model = joblib.load(in_path)
    print(f"  Model loaded ← {in_path.resolve()}")
    return model
