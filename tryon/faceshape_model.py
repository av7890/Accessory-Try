"""Face-shape classifier from FaceShapeAI (Rob4343/FaceShapeAI on Hugging Face, MIT licence).

Landmark preprocessing follows that project: centre on the midpoint of the two iris centres, rotate to level
the eyes, scale so the inter-ocular distance is 1, then feed the 478 (x, y) pairs to its scikit-learn pipeline.

The model files are pickles from a third party. They are loaded with a restricted unpickler that only
allows an exact list of scikit-learn / numpy classes, so a tampered file cannot run arbitrary code.
"""
import io
import math
import pickle
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent.parent / "models" / "faceshape"
MODEL_PATH = DIR / "face_shape_model.pkl"
ENCODER_PATH = DIR / "label_encoder.pkl"

_ALLOWED = {
    ("sklearn.pipeline", "Pipeline"), ("sklearn.decomposition._pca", "PCA"),
    ("sklearn.preprocessing._data", "StandardScaler"), ("sklearn.svm._classes", "SVC"),
    ("sklearn.preprocessing._label", "LabelEncoder"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "dtype"), ("numpy", "ndarray"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def _safe_load(path: Path):
    return _RestrictedUnpickler(io.BytesIO(path.read_bytes())).load()


def available() -> bool:
    return MODEL_PATH.exists() and ENCODER_PATH.exists()


# iris centres in the 478-point mesh
IRIS_A, IRIS_B = 468, 473


def normalise(pts: np.ndarray) -> np.ndarray:
    """(478,2) pixel landmarks -> flat 956 vector, eye-midpoint at origin, eyes level, eye distance = 1."""
    a, b = pts[IRIS_A], pts[IRIS_B]
    p = pts.astype(np.float64) - (a + b) / 2
    d = a - b
    ang = math.atan2(d[1], d[0])
    c, s = math.cos(-ang), math.sin(-ang)
    p = p @ np.array([[c, s], [-s, c]])            # rotate by -ang
    dist = float(np.hypot(*d))
    return (p / max(dist, 1e-9)).reshape(-1)


class ShapeModel:
    def __init__(self):
        self.model = _safe_load(MODEL_PATH)
        self.encoder = _safe_load(ENCODER_PATH)

    def predict(self, pts: np.ndarray):
        """Return (label, {label: probability}) for one face."""
        x = normalise(pts)[None, :]
        proba = self.model.predict_proba(x)[0]
        classes = list(self.encoder.inverse_transform(self.model.classes_))
        probs = {c: float(p) for c, p in zip(classes, proba)}
        best = max(probs, key=probs.get)
        return best, probs
