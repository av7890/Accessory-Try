"""Turn a product photo into an RGBA cutout (background removed, cropped)."""
import cv2
import numpy as np


def _crop_to_alpha(rgba: np.ndarray, pad: int = 2) -> np.ndarray:
    ys, xs = np.where(rgba[..., 3] > 10)
    if len(xs) == 0:
        return rgba
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgba.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgba.shape[1])
    return rgba[y0:y1, x0:x1]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == keep, 255, 0).astype(np.uint8)


def _clear_holes(bgr: np.ndarray, alpha: np.ndarray, tol: float, min_frac: float = 0.004) -> np.ndarray:
    """Make enclosed regions that match the border colour transparent (ring centre, bracelet loop)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    dist = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    inner = ((alpha > 127) & (dist <= tol)).astype(np.uint8)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(inner, connectivity=4)
    h, w = alpha.shape
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > min_frac * h * w:
            alpha[lab_ == i] = 0
    return alpha


def color_key(bgr: np.ndarray, tol: float = 28.0) -> np.ndarray:
    """Remove a roughly uniform background (typical catalogue shot).

    Background colour = median of the image border; pixels within `tol`
    (Lab distance) of it are background. Interior holes (rings, hoops)
    are kept transparent only if connected to the border.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    fg = (dist > tol).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # flood fill background from the border so enclosed holes stay see-through
    inv = cv2.bitwise_not(fg)
    h, w = inv.shape
    ff = inv.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for sx, sy in seeds:
        if ff[sy, sx] == 255:
            cv2.floodFill(ff, pad, (sx, sy), 128)
    bg_mask = ff == 128
    alpha = np.where(bg_mask, 0, 255).astype(np.uint8)
    alpha = _clear_holes(bgr, alpha, tol)
    alpha = _largest_component(alpha)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return np.dstack([bgr, alpha])


def grabcut(bgr: np.ndarray, tol: float = 28.0, margin: float = 0.04) -> np.ndarray:
    h, w = bgr.shape[:2]
    rect = (int(w * margin), int(h * margin), int(w * (1 - 2 * margin)), int(h * (1 - 2 * margin)))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    alpha = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    alpha = _clear_holes(bgr, alpha, tol)
    alpha = _largest_component(alpha)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return np.dstack([bgr, alpha])


_rembg_session = None


def rembg_cut(bgr: np.ndarray) -> np.ndarray:
    global _rembg_session
    from rembg import new_session, remove

    if _rembg_session is None:
        _rembg_session = new_session("u2netp")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = remove(rgb, session=_rembg_session)  # RGBA
    return cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)


METHODS = ["auto", "color-key", "grabcut", "rembg (AI)"]


def extract_accessory(img: np.ndarray, method: str = "auto", tol: float = 28.0, max_side: int = 800) -> np.ndarray:
    """img: BGR or BGRA uint8. Returns the BGRA cutout cropped to the object."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4 and method == "auto" and img[..., 3].min() < 250:
        return _crop_to_alpha(img)  # already has transparency
    bgr = img[..., :3].copy()
    scale = max_side / max(bgr.shape[:2])
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if method == "auto":
        out = color_key(bgr, tol)
        if (out[..., 3] > 128).mean() > 0.85:  # busy background: key failed
            out = grabcut(bgr, tol)
    elif method == "color-key":
        out = color_key(bgr, tol)
    elif method == "grabcut":
        out = grabcut(bgr, tol)
    else:
        out = rembg_cut(bgr)
    return _crop_to_alpha(out)
