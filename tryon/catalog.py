"""On-disk catalogue of extracted accessories (PNG cutouts + index.json)."""
import json
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "catalog"
TYPES = ("Earrings", "Necklace", "Ring", "Bracelet")
MAX_ITEMS = 200
_ID = re.compile(r"^[a-z0-9_-]{1,48}$")


def _index_path(root: Path) -> Path:
    return root / "index.json"


def _read(root: Path) -> dict:
    try:
        return json.loads(_index_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _png_path(root: Path, item_id: str) -> Path:
    if not _ID.match(item_id):
        raise ValueError("bad id")
    p = (root / f"{item_id}.png").resolve()
    if root.resolve() not in p.parents:
        raise ValueError("bad path")
    return p


def _orig_path(root: Path, item_id: str) -> Path:
    p = _png_path(root, item_id).with_name(f"{item_id}.orig.png")
    return p


def list_items(root: Path = ROOT) -> list[dict]:
    idx = _read(root)
    return [{"id": k, **v} for k, v in sorted(idx.items(), key=lambda kv: kv[1].get("name", "").lower())]


def save_item(name: str, acc_type: str, sprite: np.ndarray, root: Path = ROOT,
              original: np.ndarray | None = None) -> str:
    if acc_type not in TYPES:
        raise ValueError("bad type")
    name = re.sub(r"\s+", " ", name).strip()[:40] or "item"
    root.mkdir(parents=True, exist_ok=True)
    idx = _read(root)
    if len(idx) >= MAX_ITEMS:
        raise ValueError("catalogue is full")
    base = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_")[:40] or "item"
    item_id, n = base, 2
    while item_id in idx:
        item_id, n = f"{base}_{n}", n + 1
    ok, png = cv2.imencode(".png", sprite)
    if not ok:
        raise ValueError("could not encode image")
    _png_path(root, item_id).write_bytes(png.tobytes())
    if original is not None:        # keep the source photo so it can be re-cut with another method later
        ok, o = cv2.imencode(".png", original)
        if ok:
            _orig_path(root, item_id).write_bytes(o.tobytes())
    idx[item_id] = {"name": name, "type": acc_type}
    _index_path(root).write_text(json.dumps(idx, indent=2), encoding="utf-8")
    return item_id


def load_item(item_id: str, root: Path = ROOT) -> np.ndarray | None:
    try:
        p = _png_path(root, item_id)
    except ValueError:
        return None
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else None
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    return img


def load_original(item_id: str, root: Path = ROOT) -> np.ndarray | None:
    try:
        p = _orig_path(root, item_id)
    except ValueError:
        return None
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else None


def delete_item(item_id: str, root: Path = ROOT) -> None:
    idx = _read(root)
    if item_id in idx:
        try:
            _png_path(root, item_id).unlink(missing_ok=True)
            _orig_path(root, item_id).unlink(missing_ok=True)
        except ValueError:
            return
        del idx[item_id]
        _index_path(root).write_text(json.dumps(idx, indent=2), encoding="utf-8")
