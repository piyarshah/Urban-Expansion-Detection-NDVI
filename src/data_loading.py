import rasterio
from pathlib import Path


def _read_band(path):
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile
    return array, profile


def _get_band_paths(scene_path: Path):
    files = list(scene_path.glob("*.TIF"))

    # Detect sensor type
    name = scene_path.name

    if name.startswith("LC08") or name.startswith("LC09"):
        # Landsat 8/9
        return {
            "red": next(scene_path.glob("*SR_B4.TIF")),
            "nir": next(scene_path.glob("*SR_B5.TIF")),
            "swir1": next(scene_path.glob("*SR_B6.TIF")),
            "qa": next(scene_path.glob("*QA_PIXEL.TIF")),
        }

    elif name.startswith("LT05"):
        # Landsat 5
        return {
            "red": next(scene_path.glob("*SR_B3.TIF")),
            "nir": next(scene_path.glob("*SR_B4.TIF")),
            "swir1": next(scene_path.glob("*SR_B5.TIF")),
            "qa": next(scene_path.glob("*QA_PIXEL.TIF")),
        }

    else:
        raise ValueError(f"Unknown sensor type for scene: {name}")


def load_scene(scene_path: str):
    scene_path = Path(scene_path)

    band_paths = _get_band_paths(scene_path)

    red, profile = _read_band(band_paths["red"])
    nir, _ = _read_band(band_paths["nir"])
    swir1, _ = _read_band(band_paths["swir1"])
    qa, _ = _read_band(band_paths["qa"])

    # Sanity check: all shapes must match
    shape = red.shape
    if not (nir.shape == shape and swir1.shape == shape and qa.shape == shape):
        raise ValueError("Band shapes are not consistent")

    return {
        "red": red,
        "nir": nir,
        "swir1": swir1,
        "qa": qa,
        "profile": profile,
    }