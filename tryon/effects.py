"""Image-processing effects applied to the accessory sprite before compositing."""
import cv2
import numpy as np


class SpriteCache:
    """Tiny cache keyed by (id(sprite), params); keeps a reference so ids cannot be reused."""

    def __init__(self, size: int = 8):
        self.size, self._d = size, {}

    def get(self, sprite, params, build):
        key = (id(sprite), params)
        hit = self._d.get(key)
        if hit is not None and hit[0] is sprite:
            return hit[1]
        val = build()
        if len(self._d) >= self.size:
            self._d.pop(next(iter(self._d)))
        self._d[key] = (sprite, val)
        return val


# ------------------------------------------------------------------ relighting
def frame_light(bgr: np.ndarray):
    """(luminance, mean BGR) of a frame, from a 32x18 thumbnail -- a cheap illumination estimate."""
    small = cv2.resize(bgr, (32, 18), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    mean = small.mean(0)
    return float(0.114 * mean[0] + 0.587 * mean[1] + 0.299 * mean[2]), mean


class Relighter:
    """Match a studio-lit product cutout to the scene.

    gain  = (scene_luminance / sprite_luminance) ** strength      -> brightness match
    tint  = (scene_channel_mean / scene_gray) ** (0.6*strength)   -> colour-temperature match
    """

    def __init__(self):
        self._ref, self._lum = None, 1.0

    def _sprite_lum(self, sprite):
        if self._ref is not sprite:
            m = sprite[..., 3] > 127
            px = sprite[..., :3][m] if m.any() else sprite[..., :3].reshape(-1, 3)
            mean = px.astype(np.float32).mean(0)
            self._lum = float(0.114 * mean[0] + 0.587 * mean[1] + 0.299 * mean[2])
            self._ref = sprite
        return self._lum

    def gains(self, sprite, light, strength):
        lum_s, mean_s = light
        gain = float(np.clip((lum_s / max(self._sprite_lum(sprite), 1.0)) ** strength, 0.5, 1.25))
        tint = np.clip(mean_s / max(lum_s, 1.0), 0.6, 1.6) ** (0.6 * strength)
        tint = tint / np.exp(np.log(tint).mean())
        return (gain * tint).astype(np.float32)

    def __call__(self, sprite, light, strength):
        if strength <= 0:
            return sprite
        g = self.gains(sprite, light, strength)
        out = sprite.copy()
        out[..., :3] = np.clip(sprite[..., :3].astype(np.float32) * g, 0, 255).astype(np.uint8)
        return out


# ------------------------------------------------------------------ shape warps
def wrap_cylinder(sprite: np.ndarray, strength: float) -> np.ndarray:
    """Make a flat front-view band look like it wraps around a finger (cylindrical projection + edge shading)."""
    if strength <= 0.01:
        return sprite
    h, w = sprite.shape[:2]
    phi = (np.pi / 2) * 0.95 * min(strength, 1.0)
    u = np.linspace(-1, 1, w, dtype=np.float32)
    src = np.arcsin(np.clip(u * np.sin(phi), -1, 1)) / phi          # screen column -> texture column
    map_x = np.tile(((src + 1) / 2 * (w - 1)).astype(np.float32), (h, 1))
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    out = cv2.remap(sprite, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shade = 1 - 0.35 * strength * np.abs(u) ** 2
    out[..., :3] = (out[..., :3] * shade[None, :, None]).astype(np.uint8)
    return out


def drape(sprite: np.ndarray) -> np.ndarray:
    """Squeeze the top of a necklace photo so the chain ends meet at the neck, and fade them out."""
    h, w = sprite.shape[:2]
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    squeeze = 0.5 + 0.5 * np.clip(ys / 0.55, 0, 1) ** 0.8
    xs = np.arange(w, dtype=np.float32)
    map_x = (w / 2 + (xs[None, :] - w / 2) / squeeze[:, None]).astype(np.float32)
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    out = cv2.remap(sprite, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fade = np.clip(ys / 0.10, 0, 1)[:, None]
    out[..., 3] = (out[..., 3] * fade).astype(np.uint8)
    return out
