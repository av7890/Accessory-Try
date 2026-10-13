"""Landmark detection (MediaPipe) + accessory placement and rendering.

Pipeline per frame:
    analyse(bgr)  -> Scene   (face / hand / pose landmarks, head yaw, palm side)   [heavy, once]
    render(scene) -> image   (placement, relight, shadow, occlusion)                [cheap, per item]
Splitting the two lets a compare view render two products from one detection pass.
"""
import math
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .effects import Relighter, SpriteCache, drape, frame_light, wrap_cylinder
from .overlay import overlay
from .physics import Pendulum

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"        # optional
SEG_MODEL = MODEL_DIR / "selfie_multiclass_256x256.tflite"  # optional
HAIR, BODY_SKIN, FACE_SKIN = 1, 2, 3                        # selfie_multiclass label ids

ACCESSORIES = ["Earrings", "Necklace", "Ring", "Bracelet"]
FINGERS = {"Index": (5, 6), "Middle": (9, 10), "Ring": (13, 14), "Pinky": (17, 18)}

# FaceMesh indices
LOBE_L = (93, 132)    # the two jaw points just under the ear on the image-left side: they bracket the earlobe
LOBE_R = (323, 361)   # same on the image-right side
FACE_LEFT, FACE_RIGHT = 234, 454   # face edge at ear level (image-left / image-right)
CHIN, FOREHEAD, NOSE = 152, 10, 1


@dataclass
class Config:
    accessory: str = "Earrings"
    scale: float = 1.0           # size multiplier
    offset_y: float = 0.0        # vertical nudge
    finger: str = "Ring"
    mirror_pair: bool = True     # earrings: flip sprite for the second ear
    use_pose: bool = True        # necklace: anchor to shoulders
    hair_occlusion: bool = True  # hair covers earrings / necklace
    hair_strength: float = 0.6   # 1 = hair fully hides the jewellery, 0 = no occlusion
    relight: bool = True         # match brightness / colour temperature of the scene
    relight_strength: float = 0.6
    shadow: bool = True          # soft contact shadow
    swing: bool = True           # earrings: pendulum physics (live only)
    wrap: float = 0.6            # rings: cylindrical wrap strength
    skin_clip: bool = True       # rings / bracelets: keep only pixels on skin
    back_only: bool = True       # rings: hide when palm faces the camera
    flip_palm: bool = False      # escape hatch if palm/back detection is inverted
    show_fps: bool = False


@dataclass
class Scene:
    bgr: np.ndarray
    t: float
    faces: list = field(default_factory=list)    # list of (478,2) px arrays
    yaws: list = field(default_factory=list)     # signed head yaw per face, degrees (+ = towards image right)
    hands: list = field(default_factory=list)    # list of (21,2) px arrays
    palms: list = field(default_factory=list)    # True when the palm faces the camera
    shoulders: tuple | None = None
    _cat: np.ndarray | None = None               # cached segmentation category mask


def extras_available() -> dict:
    return {"pose": POSE_MODEL.exists(), "hair": SEG_MODEL.exists()}


def models_ready() -> bool:
    return FACE_MODEL.exists() and HAND_MODEL.exists()


class OneEuro:
    """One Euro filter: strong smoothing when still, low lag when moving."""

    def __init__(self, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None or self.x.shape != x.shape:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        speed = float(np.abs(self.dx).mean())
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = self.x + a * (x - self.x)
        self.t = t
        return self.x


def palm_facing(pts: np.ndarray, label: str) -> bool:
    """Palm toward the camera?  The 2-D cross product of (wrist->index MCP) x (wrist->pinky MCP) gives the
    hand's orientation in the image. The MediaPipe Tasks label names the actual hand seen in the image, so the
    palm shows exactly when the cross-product sign disagrees with the right-hand sign. Checked on a photo
    of the back of a left hand (nails visible) and on its mirror image."""
    a, b = pts[5] - pts[0], pts[17] - pts[0]
    cross = a[0] * b[1] - a[1] * b[0]
    return bool((cross > 0) != (label == "Right"))


def estimate_yaw(f: np.ndarray, matrix) -> float:
    """Signed head yaw (deg). Magnitude fuses the face transformation matrix with a landmark proxy
    (nose position between the face edges); the sign comes from the proxy."""
    fw = max(float(np.linalg.norm(f[FACE_RIGHT] - f[FACE_LEFT])), 1e-6)
    side = (f[FACE_RIGHT] - f[FACE_LEFT]) / fw
    ratio = float(np.dot(f[NOSE] - f[FACE_LEFT], side) / fw)
    proxy = abs(ratio - 0.5) * 110.0
    mag = proxy
    if matrix is not None:
        R = np.asarray(matrix, np.float64)[:3, :3]
        R = R / max(float(np.linalg.norm(R[:, 0])), 1e-6)
        m = abs(math.degrees(math.asin(float(np.clip(R[0, 2], -1, 1)))))
        if np.isfinite(m):
            mag = 0.5 * (m + proxy)
    return mag if ratio >= 0.5 else -mag


class TryOnEngine:
    DETECT_MAX = 640      # landmarks are normalised: detect on a small frame, draw on the full one
    SLOW_EVERY = 3        # pose refresh interval (frames) in live mode
    SEG_SIDE = 256        # long side of the image given to the segmenter (its model works at 256 px)

    def __init__(self, live: bool = False):
        self.live = live
        self.timings: dict[str, float] = {}  # EMA ms per stage
        self.fps = 0.0
        self._filters, self._cache, self._pend = {}, {}, {}
        self._ts, self._frame, self._last = 0, 0, None
        self._relight = Relighter()
        self._sprite_cache = SpriteCache()
        self._seg_in, self._seg_event, self._seg_thread = None, threading.Event(), None
        self._boxes = []
        self._relit = {}
        self._light = None
        mode = vision.RunningMode.VIDEO if live else vision.RunningMode.IMAGE
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_MODEL)), running_mode=mode, num_faces=1,
            output_facial_transformation_matrixes=True))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)), running_mode=mode, num_hands=2))
        self.pose = self.seg = None
        if POSE_MODEL.exists():
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(POSE_MODEL)), running_mode=mode))
        if SEG_MODEL.exists():
            self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(SEG_MODEL)), running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True))

    # ------------------------------------------------------------------ utils
    @contextmanager
    def _time(self, name):
        t0 = time.perf_counter()
        yield
        dt = (time.perf_counter() - t0) * 1000
        self.timings[name] = dt if name not in self.timings else 0.9 * self.timings[name] + 0.1 * dt

    def _small(self, bgr):
        m = max(bgr.shape[:2])
        if m <= self.DETECT_MAX:
            return bgr
        k = self.DETECT_MAX / m
        return cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _mp_image(bgr):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _tick(self):
        """Strictly increasing millisecond timestamp for VIDEO mode."""
        self._ts = max(self._ts + 1, int(time.perf_counter() * 1000))

    def _smooth(self, key, pts):
        if not self.live:
            return pts
        return self._filters.setdefault(key, OneEuro())(pts, time.perf_counter())

    # ------------------------------------------------------------------ analysis
    def analyse(self, bgr: np.ndarray, cfg: Config, want_pose: bool = False) -> Scene:
        """Run all detectors needed for `cfg.accessory` once."""
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-4)
            self.fps = 0.9 * self.fps + 0.1 * inst if self.fps else inst
        self._last = now
        self._frame += 1
        h, w = bgr.shape[:2]
        scene = Scene(bgr, time.perf_counter())
        face_acc = cfg.accessory in ("Earrings", "Necklace")
        mp_img = self._mp_image(self._small(bgr))
        if self.live:
            self._tick()
        with self._time("detect"):
            if face_acc or want_pose:
                r = self.face.detect_for_video(mp_img, self._ts) if self.live else self.face.detect(mp_img)
                mats = r.facial_transformation_matrixes or []
                for i, lm in enumerate(r.face_landmarks):
                    pts = self._smooth(f"f{i}", np.array([(p.x * w, p.y * h) for p in lm], np.float32))
                    scene.faces.append(pts)
                    scene.yaws.append(estimate_yaw(pts, mats[i] if i < len(mats) else None))
            if not face_acc:
                r = self.hand.detect_for_video(mp_img, self._ts) if self.live else self.hand.detect(mp_img)
                for i, lm in enumerate(r.hand_landmarks):
                    pts = self._smooth(f"h{i}", np.array([(p.x * w, p.y * h) for p in lm], np.float32))
                    label = r.handedness[i][0].category_name if i < len(r.handedness) else "Right"
                    scene.hands.append(pts)
                    scene.palms.append(palm_facing(pts, label))
        if (cfg.accessory == "Necklace" and cfg.use_pose or want_pose) and self.pose is not None and scene.faces:
            with self._time("pose"):
                scene.shoulders = self._shoulders(bgr)
        return scene

    def _shoulders(self, bgr):
        if self.live and self._frame % self.SLOW_EVERY and "pose" in self._cache:
            pts = self._cache["pose"]
            return None if pts is None else self._order(self._smooth("pose", pts))
        h, w = bgr.shape[:2]
        img = self._mp_image(self._small(bgr))
        r = self.pose.detect_for_video(img, self._ts) if self.live else self.pose.detect(img)
        if not r.pose_landmarks:
            self._cache["pose"] = None
            return None
        lm = r.pose_landmarks[0]
        if min(lm[11].visibility, lm[12].visibility) < 0.5:
            self._cache["pose"] = None
            return None
        pts = np.array([(lm[i].x * w, lm[i].y * h) for i in (11, 12)], np.float32)
        self._cache["pose"] = pts
        return self._order(self._smooth("pose", pts))

    @staticmethod
    def _order(pts):
        return (pts[0], pts[1]) if pts[0][0] < pts[1][0] else (pts[1], pts[0])

    def _seg_small(self, bgr):
        k = self.SEG_SIDE / max(bgr.shape[:2])
        return bgr if k >= 1 else cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)

    def _segment(self, bgr):
        """uint8 class map (0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 other) at the input's size."""
        r = self.seg.segment(self._mp_image(bgr))
        return None if r.category_mask is None else np.array(r.category_mask.numpy_view(), copy=True)

    @staticmethod
    def _seg_worker(ref, event):
        """Live mode: segmentation takes ~100+ ms on a CPU, so it runs here and the video loop never waits."""
        while True:
            event.wait(timeout=2.0)
            eng = ref()
            if eng is None:                       # engine was discarded: stop the thread
                return
            event.clear()
            img, eng._seg_in = eng._seg_in, None
            if img is not None:
                try:
                    with eng._time("segment"):
                        m = eng._segment(img)
                    if m is not None:
                        eng._cache["cat"] = m
                except Exception:
                    pass
            del eng

    def _submit_segmentation(self, bgr):
        if self._seg_thread is None:
            self._seg_thread = threading.Thread(target=self._seg_worker, daemon=True,
                                                args=(weakref.ref(self), self._seg_event))
            self._seg_thread.start()
        if self._seg_in is None:                  # worker idle: give it the newest frame (older ones are dropped)
            self._seg_in = self._seg_small(bgr).copy()
            self._seg_event.set()

    def _category(self, scene: Scene):
        """Full-resolution uint8 class map, or None while the first live result is still being computed."""
        if scene._cat is not None:
            return scene._cat
        H, W = scene.bgr.shape[:2]
        if self.live:
            self._submit_segmentation(scene.bgr)
            m = self._cache.get("cat")             # newest finished result (may be a few frames old)
        else:
            with self._time("segment"):
                m = self._segment(self._seg_small(scene.bgr))
        if m is None:
            return None
        scene._cat = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return scene._cat

    # ------------------------------------------------------------------ rendering
    def render_compare(self, bgr, scene, sprite_a, sprite_b, cfg):
        """Split view: left half shows A, right half shows B (one detection pass)."""
        a, found = self.render(bgr, scene, sprite_a, cfg, slot=0)
        b, _ = self.render(bgr, scene, sprite_b, cfg, slot=1)
        h, w = bgr.shape[:2]
        m = w // 2
        out = a.copy()
        out[:, m:] = b[:, m:]
        cv2.line(out, (m, 0), (m, h), (255, 255, 255), 2, cv2.LINE_AA)
        for txt, x in (("A", 12), ("B", m + 12)):
            cv2.putText(out, txt, (x, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, txt, (x, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return out, found

    def render(self, bgr, scene: Scene, sprite, cfg: Config, slot: int = 0):
        out = bgr.copy()
        if sprite is None:
            return out, False
        with self._time("render"):
            found = self._render(out, bgr, scene, sprite, cfg, slot)
        if cfg.show_fps and slot == 0:
            txt = f"{self.fps:4.1f} FPS"
            cv2.putText(out, txt, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, txt, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return out, found

    def _prepared(self, sprite, cfg, scene):
        """Shape-modified (cached) then relit sprite."""
        acc = cfg.accessory
        if acc == "Necklace":
            base = self._sprite_cache.get(sprite, "drape", lambda: drape(sprite))
        elif acc == "Ring" and cfg.wrap > 0.01:
            base = self._sprite_cache.get(sprite, ("wrap", round(cfg.wrap, 2)), lambda: wrap_cylinder(sprite, cfg.wrap))
        else:
            base = sprite
        if cfg.relight and cfg.relight_strength > 0:
            lum, mean = frame_light(scene.bgr)
            if self._light is not None and self.live:      # damp flicker from auto-exposure
                lum = 0.85 * self._light[0] + 0.15 * lum
                mean = 0.85 * self._light[1] + 0.15 * mean
            self._light = (lum, mean)
            g = self._relight.gains(base, self._light, cfg.relight_strength)
            q = tuple(int(v) for v in np.round(g / 0.03))        # gains quantised to 3%: room light changes slowly,
            key = (id(base), q)                                  # so the relit sprite is reused between frames
            hit = self._relit.get(key)
            if hit is not None and hit[0] is base:
                return hit[1]
            out = Relighter.apply(base, np.array(q, np.float32) * 0.03)
            if len(self._relit) >= 8:
                self._relit.pop(next(iter(self._relit)))
            self._relit[key] = (base, out)
            return out
        return base

    def _draw(self, box):
        """Remember the region an overlay touched, so masks are only applied there."""
        if box is not None:
            self._boxes.append(box)

    def _regions(self, shape, pad=8):
        H, W = shape[:2]
        return [(max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)) for x0, y0, x1, y1 in self._boxes]

    def _render(self, out, bgr, scene, sprite, cfg, slot):
        spr = self._prepared(sprite, cfg, scene)
        shadow = 1.0 if cfg.shadow else 0.0
        face_acc = cfg.accessory in ("Earrings", "Necklace")
        self._boxes = []
        if face_acc:
            for i, f in enumerate(scene.faces):
                if cfg.accessory == "Earrings":
                    self._earrings(out, spr, f, scene.yaws[i], cfg, slot, scene.t, shadow)
                else:
                    self._necklace(out, spr, f, cfg, scene.shoulders, shadow)
            if self._boxes and cfg.hair_occlusion and self.seg is not None:
                cat = self._category(scene)
                if cat is not None:
                    for x0, y0, x1, y1 in self._regions(out.shape):      # hair is blended back only where jewellery was drawn
                        a = cv2.GaussianBlur((cat[y0:y1, x0:x1] == HAIR).astype(np.float32), (7, 7), 0)[..., None]
                        a *= cfg.hair_strength
                        out[y0:y1, x0:x1] = (bgr[y0:y1, x0:x1] * a + out[y0:y1, x0:x1] * (1 - a)).astype(np.uint8)
            return bool(scene.faces)
        for pts, palm in zip(scene.hands, scene.palms):
            if cfg.accessory == "Ring":
                if cfg.back_only and (palm != cfg.flip_palm):
                    continue                       # palm side: the stone / band is not visible
                self._ring(out, spr, pts, cfg, shadow)
            else:
                self._bracelet(out, spr, pts, cfg, shadow)
        if self._boxes and cfg.skin_clip and self.seg is not None:
            cat = self._category(scene)
            if cat is not None:
                for x0, y0, x1, y1 in self._regions(out.shape):          # keep the jewellery only on skin
                    c = cat[y0:y1, x0:x1]
                    skin = ((c == BODY_SKIN) | (c == FACE_SKIN)).astype(np.uint8)
                    skin = cv2.dilate(skin, np.ones((9, 9), np.uint8))
                    a = cv2.GaussianBlur(skin.astype(np.float32), (5, 5), 0)[..., None]
                    out[y0:y1, x0:x1] = (out[y0:y1, x0:x1] * a + bgr[y0:y1, x0:x1] * (1 - a)).astype(np.uint8)
        return bool(scene.hands)

    # ------------------------------------------------------------------ placement
    @staticmethod
    def _face_frame(f):
        """(up unit vector, face width, face height) in pixels."""
        up = f[FOREHEAD] - f[CHIN]
        h = float(np.linalg.norm(up))
        up = up / max(h, 1e-6)
        w = float(np.linalg.norm(f[FACE_RIGHT] - f[FACE_LEFT]))
        return up, w, h

    def _pendulum(self, key) -> Pendulum:
        return self._pend.setdefault(key, Pendulum())

    @staticmethod
    def _earring_geometry(sprite, fw, scale):
        """Size and anchor from the sprite's shape, because earrings range from round studs to long drops.

        long side: 0.085 x face width for a compact stud (~1.3 cm) up to 0.33 x face width for a long drop.
        anchor   : centre of the sprite for a stud (it sits on the lobe), top centre for a hanging earring.
        """
        sh, sw = sprite.shape[:2]
        aspect = sh / max(sw, 1)
        t = float(np.clip(aspect - 1.0, 0.0, 1.2) / 1.2)
        long_side = fw * (0.085 + 0.245 * t) * scale
        width = long_side / max(aspect, 1.0)
        anchor_y = 0.04 + 0.46 * float(np.clip((1.6 - aspect) / 0.4, 0.0, 1.0))
        return width, anchor_y, width * aspect

    def _earrings(self, frame, sprite, f, yaw, cfg, slot, t, shadow):
        up, fw, fh = self._face_frame(f)
        side = (f[FACE_RIGHT] - f[FACE_LEFT]) / max(fw, 1e-6)    # unit vector image-left -> image-right
        mag = abs(yaw)
        xs = max(0.3, math.cos(math.radians(min(mag, 70))))      # foreshortening when the head turns
        base_w, anchor_y, length = self._earring_geometry(sprite, fw, cfg.scale)
        drop = cfg.offset_y * 0.30 * fh
        far_fade = float(np.clip(1.0 - (mag - 15.0) / 15.0, 0.0, 1.0))   # far ear disappears 15..30 deg
        flipped = cv2.flip(sprite, 1) if cfg.mirror_pair else sprite
        # yaw>0: head turned to image-right -> the image-right ear (FACE_RIGHT) is the far one.
        # Earlobe: midway between the two jaw points under the ear, a little outside the jaw line and a little
        # above the lobe's lower edge (the piercing sits in the lobe, not at its bottom).
        lobe_l = (f[LOBE_L[0]] + f[LOBE_L[1]]) / 2
        lobe_r = (f[LOBE_R[0]] + f[LOBE_R[1]]) / 2
        for tag, lobe, spr, out_dir, far in (("L", lobe_l, flipped, -side, yaw < 0),
                                             ("R", lobe_r, sprite, side, yaw > 0)):
            opacity = far_fade if far else 1.0
            if opacity <= 0.01:
                continue
            pivot = lobe + up * 0.05 * fh - up * drop + out_dir * 0.02 * fw
            theta = 0.0
            if cfg.swing and self.live:
                theta = self._pendulum((slot, tag)).step(float(pivot[0]), length, 65.0 * fw, t)
            hang = (-math.sin(theta), -math.cos(theta))          # sprite "up" points back to the pivot
            self._draw(overlay(frame, spr, (0.5, anchor_y), pivot, hang, base_w, opacity=opacity, xscale=xs, shadow=shadow))

    def _necklace(self, frame, sprite, f, cfg, shoulders, shadow):
        up, fw, fh = self._face_frame(f)
        if shoulders is not None:
            l, r = shoulders
            v = r - l
            sw_ = float(np.linalg.norm(v))
            sup = np.array([v[1], -v[0]], np.float32) / max(sw_, 1e-6)    # perpendicular, points up
            mid = (l + r) / 2
            neck = f[CHIN] + (mid - f[CHIN]) * 0.30 - up * cfg.offset_y * 0.6 * fh   # + = down
            self._draw(overlay(frame, sprite, (0.5, 0.0), neck, sup, 0.95 * fw * cfg.scale, shadow=shadow))   # neck ~ face width
            return
        width = 0.95 * fw * cfg.scale
        target = f[CHIN] - up * (0.15 + cfg.offset_y * 0.6) * fh
        self._draw(overlay(frame, sprite, (0.5, 0.0), target, up, width, shadow=shadow))

    def _ring(self, frame, sprite, hd, cfg, shadow):
        a, b = FINGERS[cfg.finger]
        seg = hd[b] - hd[a]                      # knuckle -> next joint (toward fingertip)
        length = float(np.linalg.norm(seg))
        centre = hd[a] + seg * (0.45 - cfg.offset_y * 0.7)       # + = towards the knuckle (down)
        self._draw(overlay(frame, sprite, (0.5, 0.5), centre, seg, 0.50 * length * cfg.scale, shadow=shadow))

    def _bracelet(self, frame, sprite, hd, cfg, shadow):
        palm_w = float(np.linalg.norm(hd[5] - hd[17]))
        d = hd[9] - hd[0]                        # wrist -> middle knuckle
        dl = float(np.linalg.norm(d))
        un = d / max(dl, 1e-6)
        centre = hd[0] - un * (0.12 + cfg.offset_y * 0.5) * dl   # + = towards the forearm (down)
        self._draw(overlay(frame, sprite, (0.5, 0.5), centre, d, 0.85 * palm_w * cfg.scale, shadow=shadow))
