"""Alpha-composite an RGBA sprite onto a frame with scale / rotation / anchor, optional soft shadow."""
import cv2
import numpy as np


def overlay(frame: np.ndarray, sprite: np.ndarray, anchor_frac, target_xy, up_vec, width_px: float,
            opacity: float = 1.0, xscale: float = 1.0, shadow: float = 0.0):
    """Draw `sprite` (BGRA uint8) onto `frame` (BGR uint8) in place.

    anchor_frac : (fx, fy) point of the sprite (fractions of w,h) that lands on target_xy.
    up_vec      : direction in frame space the sprite's "up" (0,-1) should point to.
    width_px    : rendered sprite width in frame pixels (before xscale).
    opacity     : 0..1 global alpha (used to fade accessories that turn out of view).
    xscale      : extra horizontal squash in sprite space (head-yaw foreshortening).
    shadow      : 0..1 strength of a soft contact shadow cast below/right of the sprite.
    Returns the modified region (x0, y0, x1, y1), or None if nothing was drawn.
    """
    sh, sw = sprite.shape[:2]
    if width_px < 2 or opacity <= 0.01:
        return None
    ux, uy = up_vec
    n = np.hypot(ux, uy)
    ux, uy = (0.0, -1.0) if n < 1e-6 else (ux / n, uy / n)
    if width_px < sw * 0.6:  # pre-shrink with area filtering so warpAffine does not alias
        f = (width_px * 1.5) / sw
        sprite = cv2.resize(sprite, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sh, sw = sprite.shape[:2]
    k = width_px / sw
    # R maps (0,-1) -> (ux,uy); column 0 is the sprite x axis
    R = np.array([[-uy, -ux], [ux, -uy]], np.float32) * k
    R[:, 0] *= xscale
    anchor = np.array([anchor_frac[0] * sw, anchor_frac[1] * sh], np.float32)
    t = np.array(target_xy, np.float32) - R @ anchor

    corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
    pts = corners @ R.T + t
    H, W = frame.shape[:2]
    pad = int(0.12 * width_px) + 3 if shadow > 0 else 0
    x0, y0 = np.floor(pts.min(0)).astype(int) - pad
    x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    M = np.hstack([R, (t - (x0, y0))[:, None]])

    a = sprite[..., 3:4].astype(np.float32) / 255.0 * opacity
    pm = np.concatenate([sprite[..., :3].astype(np.float32) * a, a * 255.0], axis=2)
    warped = cv2.warpAffine(pm, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = warped[..., 3:4] / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    if shadow > 0:
        sigma = max(width_px * 0.03, 1.0)
        sh_a = cv2.GaussianBlur(warped[..., 3] / 255.0, (0, 0), sigma)
        shift = np.float32([[1, 0, 0.012 * width_px], [0, 1, 0.03 * width_px]])
        sh_a = cv2.warpAffine(sh_a, shift, (sh_a.shape[1], sh_a.shape[0]))
        roi = roi * (1.0 - shadow * 0.45 * sh_a[..., None] * (1.0 - alpha))
    frame[y0:y1, x0:x1] = np.clip(warped[..., :3] + roi * (1.0 - alpha), 0, 255).astype(np.uint8)
    return int(x0), int(y0), int(x1), int(y1)
