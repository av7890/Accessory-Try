"""Virtual accessory try-on  --  streamlit run app.py"""
import logging
import threading
from collections import Counter, deque

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from tryon import catalog, faceshape_model
from tryon.engine import ACCESSORIES, FINGERS, Config, TryOnEngine, extras_available, models_ready
from tryon.net import FetchError, safe_decode, safe_fetch
from tryon.segment import METHODS, extract_accessory
from tryon.shape import CHEEK_L, CHEEK_R, CHIN, FOREHEAD_L, FOREHEAD_R, JAW_L, JAW_R, TOP
from tryon.shape import alternatives, classify, extended_top, measure, neck_class, neck_ratio, recommend

log = logging.getLogger("tryon")

# aiortc defaults to ~0.5 Mbps start / 1.5 Mbps max, which makes the returned video blocky.
try:
    import aiortc.codecs.h264 as _h264
    import aiortc.codecs.vpx as _vpx
    for _m in (_h264, _vpx):
        _m.DEFAULT_BITRATE, _m.MIN_BITRATE, _m.MAX_BITRATE = 4_000_000, 2_000_000, 8_000_000
except Exception:
    pass

st.set_page_config(page_title="Accessory Try-On", layout="wide")
st.title("Virtual Accessory Try-On")

if not models_ready():
    st.error("Model files missing. Run `python download_models.py` once, then reload.")
    st.stop()

SS = st.session_state


# ---------------------------------------------------------------- helpers
@st.cache_resource
def photo_engine():
    return TryOnEngine(live=False), threading.Lock()   # MediaPipe task objects are not thread-safe


@st.cache_resource
def shape_model():
    try:
        return faceshape_model.ShapeModel() if faceshape_model.available() else None
    except Exception:
        log.exception("face-shape model failed to load")
        return None


def pick_shape(rule: str, probs: dict | None):
    """The model scored ~67% on 1,200 labelled test images; our hand-written rules scored ~20% (chance).
    So the model decides; the rules are only a fallback when the model is unavailable."""
    return max(probs, key=probs.get) if probs else rule


@st.cache_data(show_spinner=False, ttl=900, max_entries=8)
def fetch_cached(url: str) -> bytes:
    return safe_fetch(url)


@st.cache_data(show_spinner=False, max_entries=16)
def cutout_cached(raw: bytes, method: str, tol: float) -> np.ndarray:
    return extract_accessory(safe_decode(raw), method, tol)


@st.cache_data(show_spinner=False, max_entries=32)
def recut_cached(item_id: str, method: str, tol: float) -> np.ndarray | None:
    orig = catalog.load_original(item_id)
    return None if orig is None else extract_accessory(orig, method, tol)


def shrink(sprite, max_side=600):
    s = max_side / max(sprite.shape[:2])
    return cv2.resize(sprite, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else sprite


def on_light(sprite, bg=235):
    a = sprite[..., 3:4] / 255.0
    return cv2.cvtColor((sprite[..., :3] * a + bg * (1 - a)).astype(np.uint8), cv2.COLOR_BGR2RGB)


def decode_user_image(data: bytes):
    """Decode an uploaded/snapshot image to BGR (drops alpha). Returns None and shows a message on failure."""
    try:
        img = safe_decode(data)
    except ValueError as e:
        st.warning(str(e))
        return None
    return img[..., :3].copy() if img.ndim == 3 and img.shape[2] == 4 else (
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img)


# ---------------------------------------------------------------- sidebar: source
items = catalog.list_items()
names = {i["id"]: f'{i["name"]} ({i["type"]})' for i in items}
type_of = {i["id"]: i["type"] for i in items}

sprite_full = None          # extracted cutout of a NEW image (for saving)
sprite = sprite_b = None    # what gets rendered
compare = False
with st.sidebar:
    st.header("1. Accessory")
    src = st.radio("Source", ["New image", "Catalogue"], horizontal=True, key="src")
    if src == "Catalogue" and not items:
        st.info("Catalogue is empty. Add items from 'New image'.")
        src = "New image"

    if src == "New image":
        acc_type = st.selectbox("Type", ACCESSORIES)
        how = st.radio("Product image from", ["Upload", "Image URL"], horizontal=True)
        raw = None
        if how == "Upload":
            up = st.file_uploader("Product photo", type=["png", "jpg", "jpeg", "webp"])
            raw = up.getvalue() if up else None
        else:
            url = st.text_input("Direct image URL (right-click image on shop site > copy image address)")
            if url:
                try:
                    raw = fetch_cached(url)
                except FetchError as e:
                    st.warning(str(e))
                except Exception:
                    log.exception("fetch failed")
                    st.warning("Could not load that image.")
        method = st.selectbox("Background removal", METHODS,
                              help="auto = colour key (plain backgrounds) with GrabCut fallback. "
                                   "'rembg (AI)' downloads a ~4 MB model on first use.")
        tol = st.slider("Colour-key tolerance", 5, 80, 28)
        if raw is not None:
            try:
                sprite_full = cutout_cached(raw, method, float(tol))
            except ValueError as e:
                st.warning(str(e))
            except Exception:
                log.exception("segmentation failed")
                st.warning("Could not cut out that image. Try another method.")
        if sprite_full is not None:
            cov = float((sprite_full[..., 3] > 127).mean())
            if cov < 0.12 or cov > 0.97:
                st.warning("Cutout looks wrong. Use a clean catalogue shot of the product alone "
                           "(not a photo of it being worn), or try another removal method.")
            st.caption("Extracted cutout")
            st.image(on_light(sprite_full))
            c1, c2 = st.columns([2, 1])
            nm = c1.text_input("Save as", key="save_name", label_visibility="collapsed", placeholder="Name")
            if c2.button("Save"):
                try:
                    orig = safe_decode(raw)
                    k = 1600 / max(orig.shape[:2])
                    if k < 1:
                        orig = cv2.resize(orig, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
                    catalog.save_item(nm or "item", acc_type, sprite_full, original=orig)
                    st.success("Saved to catalogue.")
                except ValueError as e:
                    st.warning(str(e))
            sprite = shrink(sprite_full)
    else:
        ids = [i["id"] for i in items]
        if SS.get("item_a") not in ids:
            SS["item_a"] = ids[0]
        a_id = st.selectbox("Item A", ids, format_func=names.get, key="item_a")
        acc_type = type_of[a_id]
        st.caption(f"Type: {acc_type}")
        has_orig = catalog.load_original(a_id) is not None
        cmethod, ctol = "Saved cut-out", 28
        if has_orig:
            cmethod = st.selectbox("Cut-out method", ["Saved cut-out"] + METHODS, key="cat_method",
                                   help="Re-cut the saved original photo with another method. "
                                        "'rembg (AI)' downloads a ~4 MB model on first use.")
            if cmethod in ("auto", "color-key"):
                ctol = st.slider("Colour-key tolerance", 5, 80, 28, key="cat_tol")
        else:
            st.caption("No original photo saved for this item, so it cannot be re-cut. "
                       "Add it again from 'New image' and press Save to keep the original.")

        def item_sprite(i):
            if cmethod != "Saved cut-out":
                try:
                    s_ = recut_cached(i, cmethod, float(ctol))
                    if s_ is not None:
                        return s_
                except Exception:
                    log.exception("re-cut failed")
                    st.warning("Could not re-cut with that method; using the saved cut-out.")
            return catalog.load_item(i)

        sa = item_sprite(a_id)
        sprite = shrink(sa) if sa is not None else None
        same = [i for i in ids if type_of[i] == acc_type]
        if len(same) > 1:
            compare = st.checkbox("Compare with another item (split view)")
            if compare:
                if SS.get("item_b") not in same:
                    SS["item_b"] = next(i for i in same if i != a_id)
                b_id = st.selectbox("Item B", same, format_func=names.get, key="item_b")
                sb = item_sprite(b_id)
                sprite_b = shrink(sb) if sb is not None else None
        if sprite is not None:
            st.image(on_light(sprite), width=160)

    st.header("2. Fit")
    scale = st.slider("Size", 0.3, 2.5, 1.0, 0.05)
    offset = st.slider("Vertical offset", -1.0, 1.0, 0.0, 0.05)
    finger = st.selectbox("Finger (rings)", list(FINGERS), index=2) if acc_type == "Ring" else "Ring"
    mirror = st.checkbox("Mirror 2nd earring", True) if acc_type == "Earrings" else True

    extras = extras_available()
    st.header("3. Realism")
    use_pose = st.checkbox("Shoulder tracking (necklace)", extras["pose"], disabled=not extras["pose"])
    hair_occ = st.checkbox("Hair occlusion (earrings/necklace)", extras["hair"], disabled=not extras["hair"])
    hair_str = st.slider("Hair cover strength", 0.0, 1.0, 0.6, 0.05, disabled=not (hair_occ and extras["hair"]),
                         help="How strongly hair hides earrings/necklace. Lower it if long hair swallows the jewellery."
                         ) if extras["hair"] else 0.6
    skin_clip = st.checkbox("Clip to skin (ring/bracelet)", extras["hair"], disabled=not extras["hair"],
                            help="Uses the segmenter so the band stays on the finger/wrist.")
    relight_on = st.checkbox("Relight to match the room", True,
                             help="Matches brightness and colour temperature of your video.")
    relight_s = st.slider("Relight strength", 0.0, 1.0, 0.6, 0.05) if relight_on else 0.0
    shadow = st.checkbox("Contact shadow", True)
    swing = st.checkbox("Earring swing physics (live)", True)
    wrap = st.slider("Ring wrap", 0.0, 1.0, 0.6, 0.05) if acc_type == "Ring" else 0.6
    back_only = st.checkbox("Hide ring when palm faces camera", True) if acc_type == "Ring" else True
    flip_palm = st.checkbox("Palm/back detection is inverted", False) if acc_type == "Ring" else False
    show_fps = st.checkbox("Show FPS overlay", False)
    if not all(extras.values()):
        st.caption("Missing optional models: run `python download_models.py`.")

cfg = Config(accessory=acc_type, scale=scale, offset_y=offset, finger=finger, mirror_pair=mirror,
             use_pose=use_pose, hair_occlusion=hair_occ, hair_strength=hair_str, relight=relight_on, relight_strength=relight_s,
             shadow=shadow, swing=swing, wrap=wrap, skin_clip=skin_clip, back_only=back_only,
             flip_palm=flip_palm, show_fps=show_fps)

if sprite is None:
    st.info("Pick or add an accessory in the sidebar to start.")

tab_photo, tab_live, tab_cat, tab_style = st.tabs(["Photo", "Live webcam", "Catalogue", "Style advisor"])

# ---------------------------------------------------------------- photo
with tab_photo:
    c1, c2 = st.columns(2)
    with c1:
        mode = st.radio("Your photo", ["Upload", "Take picture"], horizontal=True)
        if mode == "Upload":
            f = st.file_uploader("Photo of you", type=["png", "jpg", "jpeg", "webp"], key="me")
        else:
            f = st.camera_input("Camera")
        data = f.getvalue() if f else None
    with c2:
        if data is not None and sprite is not None:
            me = decode_user_image(data)
            if me is not None:
                engine, lock = photo_engine()
                try:
                    with lock:
                        scene = engine.analyse(me, cfg)
                        if compare and sprite_b is not None:
                            res, found = engine.render_compare(me, scene, sprite, sprite_b, cfg)
                        else:
                            res, found = engine.render(me, scene, sprite, cfg)
                    if not found:
                        st.warning("No face found." if acc_type in ("Earrings", "Necklace") else "No hand found.")
                    elif acc_type == "Ring" and not scene.hands:
                        st.warning("No hand found.")
                    st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), caption="Try-on result")
                    ok, png = cv2.imencode(".png", res)
                    st.download_button("Download result", png.tobytes(), "tryon.png", "image/png")
                except Exception:
                    log.exception("photo try-on failed")
                    st.error("Something went wrong processing that photo.")


# ---------------------------------------------------------------- live
class Processor(VideoProcessorBase):
    def __init__(self):
        self.engine = TryOnEngine(live=True)
        self.sprite = self.sprite_b = None
        self.cfg = Config()
        self.shapes = deque(maxlen=90)
        self.necks = deque(maxlen=90)
        self.sm = faceshape_model.ShapeModel() if faceshape_model.available() else None
        self._n = 0

    def update(self, sprite, sprite_b, cfg):
        self.sprite, self.sprite_b, self.cfg = sprite, sprite_b, cfg

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)   # mirror like a real mirror
        cfg, sa, sb = self.cfg, self.sprite, self.sprite_b
        if sa is not None:
            scene = self.engine.analyse(img, cfg)
            if scene.faces:
                self._n += 1
                if self.sm is not None:
                    if self._n % 5 == 0:                      # the model is cheap but needn't run every frame
                        self.shapes.append(self.sm.predict(scene.faces[0])[0])
                else:
                    self.shapes.append(classify(measure(scene.faces[0])))
                r = neck_ratio(scene.faces[0], scene.shoulders)
                if r is not None:
                    self.necks.append(r)
            if sb is not None:
                img, _ = self.engine.render_compare(img, scene, sa, sb, cfg)
            else:
                img, _ = self.engine.render(img, scene, sa, cfg)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


ctx = None
with tab_live:
    st.caption("Allow camera access. Sidebar changes apply live. Sit back so your head and both shoulders "
               "are in frame; light your face from the front.")
    ctx = webrtc_streamer(key="live", video_processor_factory=Processor,
                          media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720},
                                                             "frameRate": {"ideal": 30}}, "audio": False},
                          async_processing=True)
    if ctx.video_processor:
        ctx.video_processor.update(sprite, sprite_b, cfg)
        t = ctx.video_processor.engine.timings
        if t:
            st.caption("Stage time (ms): " + "  |  ".join(f"{k} {v:.1f}" for k, v in t.items()) +
                       f"  |  {ctx.video_processor.engine.fps:.1f} FPS (refresh page to update)")

# ---------------------------------------------------------------- catalogue
with tab_cat:
    items = catalog.list_items()
    if not items:
        st.info("Nothing saved yet. Extract a product in the sidebar and press Save.")
    cols = st.columns(4)
    for n, it in enumerate(items):
        with cols[n % 4]:
            img = catalog.load_item(it["id"])
            if img is not None:
                st.image(on_light(shrink(img, 300)))
            st.caption(f'**{it["name"]}** · {it["type"]}')

            def _use(i=it["id"]):
                SS["src"], SS["item_a"] = "Catalogue", i

            b1, b2 = st.columns(2)
            b1.button("Use", key=f"use_{it['id']}", on_click=_use)
            if b2.button("Delete", key=f"del_{it['id']}"):
                catalog.delete_item(it["id"])
                recut_cached.clear()
                st.rerun()

# ---------------------------------------------------------------- style advisor
with tab_style:
    st.caption("Estimates your face shape from FaceMesh landmarks (works best on a frontal, relaxed face) "
               "and suggests jewellery. Uses the FaceShapeAI model (Rob4343, MIT licence) together with simple "
               "measurement rules. A suggestion, not a measurement standard.")
    sc1, sc2 = st.columns(2)
    with sc1:
        smode = st.radio("Input", ["Upload", "Take picture"], horizontal=True, key="smode")
        sf = st.file_uploader("Frontal photo", type=["png", "jpg", "jpeg", "webp"], key="sphoto") \
            if smode == "Upload" else st.camera_input("Camera", key="scam")
        live_read = st.button("Use live camera reading instead")
    shape = neck = None
    m = None
    vis = None
    model_probs = None
    rule_shape = None
    if live_read and ctx is not None and ctx.video_processor and ctx.video_processor.shapes:
        shape = Counter(ctx.video_processor.shapes).most_common(1)[0][0]
        necks = ctx.video_processor.necks
        neck = neck_class(float(np.median(necks))) if necks else None
        sc2.info("Using the last ~3 seconds of the live camera (needs a face accessory selected).")
    elif sf is not None:
        img = decode_user_image(sf.getvalue())
        if img is not None:
            engine, lock = photo_engine()
            with lock:
                scene = engine.analyse(img, Config(accessory="Necklace"), want_pose=True)
            if scene.faces:
                f = scene.faces[0]
                m = measure(f)
                rule_shape = classify(m)
                sm = shape_model()
                if sm is not None:
                    try:
                        _, model_probs = sm.predict(f)
                    except Exception:
                        log.exception("face-shape model failed")
                shape = pick_shape(rule_shape, model_probs)
                neck = neck_class(neck_ratio(f, scene.shoulders))
                vis = img.copy()
                pts = {CHIN: f[CHIN], TOP: extended_top(f)}
                for a, b in ((CHEEK_L, CHEEK_R), (FOREHEAD_L, FOREHEAD_R), (JAW_L, JAW_R), (TOP, CHIN)):
                    pa, pb = pts.get(a, f[a]), pts.get(b, f[b])
                    cv2.line(vis, tuple(map(int, pa)), tuple(map(int, pb)), (0, 255, 255), 2, cv2.LINE_AA)
            else:
                sc2.warning("No face found.")
    if shape:
        with sc2:
            if vis is not None:
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Measurements used")
            st.subheader(f"Face shape: {shape}")
            if model_probs:
                top = sorted(model_probs.items(), key=lambda kv: -kv[1])[:3]
                st.markdown("**Model confidence:** " + " · ".join(f"{k} {v:.0%}" for k, v in top))
                if top[0][1] < 0.5:
                    st.warning(f"Low confidence: this face is between {top[0][0]} and {top[1][0]}. "
                               "Face shape is a spectrum; advice for both is equally fine.")
                st.caption("Model accuracy on 1,200 labelled test photos was about 67% (5 classes); "
                           "about 82% when it is at least 70% confident.")
            else:
                st.warning("Face-shape model not available (run download_models.py). Falling back to simple "
                           "measurement rules, which are not validated and are often wrong.")
                alt = alternatives(m) if m else []
                if alt:
                    st.caption("Also close to: " + ", ".join(alt))
            if m:
                st.caption(f'length/width {m["length_to_width"]:.2f} · forehead/cheek {m["forehead_to_cheek"]:.2f} '
                           f'· jaw/cheek {m["jaw_to_cheek"]:.2f}')
            r = recommend(shape, neck)
            st.markdown(f"**Earrings:** {r['earrings']}\n\n**Necklaces:** {r['necklace']}\n\n**Avoid:** {r['avoid']}")
            if "neck" in r:
                st.markdown(f"**Neck line:** {r['neck']}")
