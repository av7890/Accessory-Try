"""Face-shape analysis from FaceMesh landmarks and jewellery recommendations (frontal faces, heuristic)."""
import numpy as np

# FaceMesh indices
FOREHEAD_L, FOREHEAD_R = 54, 284
CHEEK_L, CHEEK_R = 234, 454
JAW_L, JAW_R = 172, 397      # lower jaw line (58/288 sit at cheek/mouth level, too high)
HAIRLINE_EXTEND = 1.15       # FaceMesh stops mid-forehead; stretch chin->top by this to approximate the hairline
TOP, CHIN = 10, 152


def _d(f, a, b) -> float:
    return float(np.linalg.norm(f[a] - f[b]))


def extended_top(f: np.ndarray) -> np.ndarray:
    """Estimated hairline point: the mesh top pushed further up along the chin->top axis."""
    return f[CHIN] + (f[TOP] - f[CHIN]) * HAIRLINE_EXTEND


def measure(f: np.ndarray) -> dict:
    cheek = max(_d(f, CHEEK_L, CHEEK_R), 1e-6)
    return {
        "length_to_width": float(np.linalg.norm(extended_top(f) - f[CHIN])) / cheek,
        "forehead_to_cheek": _d(f, FOREHEAD_L, FOREHEAD_R) / cheek,
        "jaw_to_cheek": _d(f, JAW_L, JAW_R) / cheek,
    }


def classify(m: dict) -> str:
    lw, fj, jj = m["length_to_width"], m["forehead_to_cheek"], m["jaw_to_cheek"]
    if lw >= 1.45:
        return "Oblong"
    if jj >= 0.88 and fj >= 0.88:
        return "Square" if lw < 1.32 else "Oblong"
    if fj >= 0.95 and jj < 0.80:
        return "Heart"
    if fj < 0.85 and jj < 0.85:
        return "Diamond"
    if lw < 1.25:
        return "Round"
    return "Oval"


def alternatives(m: dict, margin: float = 0.05) -> list[str]:
    """Other shapes the face would get if each ratio were off by +-margin (measurement noise).
    A non-empty list means the result is borderline."""
    best = classify(m)
    out = []
    for key in m:
        for d in (-margin, margin):
            alt = classify({**m, key: m[key] + d})
            if alt != best and alt not in out:
                out.append(alt)
    return out


def neck_ratio(face: np.ndarray, shoulders) -> float | None:
    """Chin -> shoulder-line distance relative to face width (None without pose)."""
    if shoulders is None:
        return None
    mid = (shoulders[0] + shoulders[1]) / 2
    return float(np.linalg.norm(mid - face[CHIN]) / max(_d(face, CHEEK_L, CHEEK_R), 1e-6))


def neck_class(r: float | None) -> str | None:
    if r is None:
        return None
    return "short" if r < 0.8 else "long" if r > 1.15 else "average"


RECS = {
    "Oval": {
        "earrings": "Almost anything works: studs, hoops, drops. Try statement chandeliers.",
        "necklace": "Pendants and chokers both suit you. Experiment with length.",
        "avoid": "Nothing specific; keep size in proportion to your face.",
    },
    "Round": {
        "earrings": "Long drops, angular or geometric shapes add length and definition.",
        "necklace": "Long pendants, V-shapes, and layered lengths elongate the face.",
        "avoid": "Large round hoops and short chokers that repeat the roundness.",
    },
    "Square": {
        "earrings": "Round or curved earrings, hoops, and teardrops soften the jaw.",
        "necklace": "Round pendants and soft U-shaped chains balance strong angles.",
        "avoid": "Sharp geometric earrings and boxy pendants.",
    },
    "Oblong": {
        "earrings": "Wide studs, button earrings, and medium hoops add width.",
        "necklace": "Chokers and collar-length necklaces shorten the face visually.",
        "avoid": "Very long drops and long vertical pendants.",
    },
    "Heart": {
        "earrings": "Teardrop and triangular shapes wider at the bottom balance a narrow chin.",
        "necklace": "Short to medium chains with rounded pendants.",
        "avoid": "Wide top-heavy earrings such as large studs or bold tops.",
    },
    "Diamond": {
        "earrings": "Studs with a little sparkle, oval or drop shapes. Keep it delicate.",
        "necklace": "Soft U-shaped or round pendants at medium length.",
        "avoid": "Wide hoops that emphasise the cheekbones.",
    },
}

NECK_ADVICE = {
    "short": "Shorter neck line: choose longer necklaces or V-shapes, and skip tight chokers.",
    "long": "Longer neck line: chokers, collars, and layered short chains look great.",
    "average": "Average neck line: most necklace lengths work.",
}


def recommend(shape: str, neck: str | None = None) -> dict:
    r = dict(RECS[shape])
    if neck:
        r["neck"] = NECK_ADVICE[neck]
    return r
