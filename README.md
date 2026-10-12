# Virtual Accessory Try-On

A Streamlit app that places earrings, necklaces, rings and bracelets from a product photo onto a person in a
photo or a live webcam. Course project for Image & Video Processing.

## 1. How to run

**Windows, one click:** double-click `setup_and_run.bat`. On the first run it creates a virtual environment,
installs the packages, downloads the models and starts the app. Later runs skip the installation.

**Manually:**

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
.venv/Scripts/python download_models.py                     # downloads the model files into models/
.venv/Scripts/streamlit run app.py                          # opens http://localhost:8501
```

**How to use the app:**
1. Sidebar, "1. Accessory": choose *New image* (upload a product photo or paste a direct image URL) or
   *Catalogue* (a saved item).
2. Check the extracted cut-out in the sidebar. Change the removal method or tolerance if the edges look wrong.
   Press *Save* to keep it in the catalogue.
3. Use the *Photo* tab (upload or take a picture) or the *Live webcam* tab (allow camera access).
4. Adjust *Fit* and *Realism* settings in the sidebar. Use the *Style advisor* tab for face shape and
   jewellery suggestions.

**Tests:** `.venv/Scripts/python -m pip install -r requirements-dev.txt`, then `.venv/Scripts/python -m pytest`.

## 2. Requirements

| Item | Detail |
|---|---|
| OS | Developed and run on Windows 11. `setup_and_run.bat` is Windows only. |
| Python | 3.14 |
| Browser | A current Chrome or Edge. Camera access works only on `localhost` or HTTPS. |
| Camera | Optional; needed only for the *Live webcam* tab and *Take picture* |
| Internet | Needed for the first install and model download, and for the *Image URL* option |
| Disk | About 500 MB for packages, about 35 MB for models |

Python packages (`requirements.txt`): mediapipe 1.1.0, streamlit 1.65.0, streamlit-webrtc 0.78.1,
aiortc 1.15.0, av 17.1.0, opencv-python 5.0.0.93, numpy 2.5.3, pillow 12.3.0, rembg 2.0.85,
onnxruntime 1.30.0, scikit-learn 1.9.1. Dev only: pytest 9.1.1.

## 3. Directory format

```
Try-On/
  app.py                     Streamlit user interface (all tabs, sidebar, live video processor)
  download_models.py         downloads and checksum-verifies the model files
  setup_and_run.bat          one-click setup and start (Windows)
  requirements.txt           runtime packages        requirements-dev.txt   adds pytest
  pytest.ini                 test settings
  .streamlit/config.toml     server settings (localhost only, upload limit)
  tryon/
    engine.py                landmark detection + placement + rendering of each accessory
    segment.py               product photo -> transparent cut-out
    overlay.py               draws a sprite onto a frame (rotate, scale, shadow, opacity)
    effects.py               relighting, ring wrap warp, necklace drape warp, small cache
    physics.py               pendulum used for earring swing
    shape.py                 face-shape measurements, fallback rules, neck length, style advice text
    faceshape_model.py       loads and runs the third-party face-shape model
    catalog.py               saves/loads cut-outs (and originals) on disk
    net.py                   URL download and image decoding with safety checks
  models/                    downloaded model files (see section 4)
    faceshape/               third-party face-shape model files
  catalog/                   saved accessories: index.json, <id>.png, <id>.orig.png
  tests/                     unit tests
```

## 4. Technology and models used

| Technology | Used for |
|---|---|
| Python | Everything |
| Streamlit | Web interface |
| streamlit-webrtc (aiortc, av) | Sending webcam video to Python and back |
| OpenCV, NumPy, Pillow | Image processing, drawing, image checks |
| MediaPipe Tasks | Landmark and segmentation models below |
| scikit-learn | Runs the face-shape model |
| rembg + onnxruntime | Optional AI background removal (downloads its own model on first use) |

| Model file | Source | Used for |
|---|---|---|
| `models/face_landmarker.task` | Google MediaPipe | 478 face landmarks (including irises) and head-pose matrix |
| `models/hand_landmarker.task` | Google MediaPipe | 21 hand landmarks and left/right label |
| `models/pose_landmarker_lite.task` | Google MediaPipe | Shoulder positions (necklace) |
| `models/selfie_multiclass_256x256.tflite` | Google MediaPipe | Per-pixel classes: background, hair, body-skin, face-skin, clothes, other |
| `models/faceshape/face_shape_model.pkl`, `label_encoder.pkl` | [Rob4343/FaceShapeAI](https://huggingface.co/Rob4343/FaceShapeAI), MIT licence | Face shape: Heart, Oblong, Oval, Round, Square |

The face-shape files are pickles, so they are loaded with a restricted unpickler that allows only an exact list
of scikit-learn and numpy classes. `download_models.py` verifies each file's SHA-256.

## 5. Methods and functions

"Defined in" is the file that contains the function. "Used in" is where it is called from.

### Cut-out (product photo to transparent image)

| Function | What it does | Defined in | Used in |
|---|---|---|---|
| `extract_accessory` | Chooses the removal method and returns the cropped RGBA cut-out | `tryon/segment.py` | `app.py` (`cutout_cached`, `recut_cached`) |
| `color_key` | Removes a plain background by colour distance to the border colour | `tryon/segment.py` | `extract_accessory` |
| `grabcut` | Removes the background with OpenCV GrabCut | `tryon/segment.py` | `extract_accessory` |
| `rembg_cut` | Removes the background with the rembg AI model | `tryon/segment.py` | `extract_accessory` |
| `_clear_holes` | Makes enclosed background areas (ring centre, bracelet loop) transparent | `tryon/segment.py` | `color_key`, `grabcut` |
| `_largest_component`, `_crop_to_alpha` | Keeps the biggest object; crops to its bounding box | `tryon/segment.py` | `color_key`, `grabcut`, `extract_accessory` |

### Detection and rendering (the engine)

| Function | What it does | Defined in | Used in |
|---|---|---|---|
| `TryOnEngine.analyse` | Runs the needed detectors on a frame and returns a `Scene` | `tryon/engine.py` | `app.py` (photo tab, live `Processor`, advisor) |
| `TryOnEngine._shoulders` | Finds the two shoulder points with the pose model | `tryon/engine.py` | `analyse` |
| `TryOnEngine._category` | Gets the per-pixel class mask (hair, skin) from the segmenter | `tryon/engine.py` | `_render` |
| `TryOnEngine.render` | Draws one accessory onto a copy of the frame | `tryon/engine.py` | `app.py`, `render_compare` |
| `TryOnEngine.render_compare` | Draws item A on the left half and item B on the right half | `tryon/engine.py` | `app.py` |
| `TryOnEngine._prepared` | Applies the shape warp, then relighting, to the sprite | `tryon/engine.py` | `_render` |
| `TryOnEngine._render` | Per accessory type: places it, then applies hair and skin masks | `tryon/engine.py` | `render` |
| `TryOnEngine._earrings` | Places earrings at the earlobes with yaw squash, fade and swing | `tryon/engine.py` | `_render` |
| `TryOnEngine._necklace` | Places the necklace at the neck using shoulders or the face | `tryon/engine.py` | `_render` |
| `TryOnEngine._ring` | Places the ring on the chosen finger segment | `tryon/engine.py` | `_render` |
| `TryOnEngine._bracelet` | Places the bracelet at the wrist | `tryon/engine.py` | `_render` |
| `estimate_yaw` | Estimates left/right head turn in degrees | `tryon/engine.py` | `analyse` |
| `palm_facing` | Decides whether the palm or the back of the hand faces the camera | `tryon/engine.py` | `analyse` |
| `OneEuro` | Smooths landmark positions over time | `tryon/engine.py` | `TryOnEngine._smooth` |
| `Pendulum.step` | Advances the earring swing by one time step | `tryon/physics.py` | `_earrings` |

### Drawing and effects

| Function | What it does | Defined in | Used in |
|---|---|---|---|
| `overlay` | Rotates, scales and blends a sprite onto a frame (with shadow, opacity, squash) | `tryon/overlay.py` | `_earrings`, `_necklace`, `_ring`, `_bracelet` |
| `Relighter` (`__call__`) | Scales sprite brightness and colour toward the scene's lighting | `tryon/effects.py` | `_prepared` |
| `frame_light` | Estimates scene brightness and colour from a small thumbnail | `tryon/effects.py` | `_prepared` |
| `wrap_cylinder` | Warps a flat ring image so it looks wrapped round a finger | `tryon/effects.py` | `_prepared` |
| `drape` | Narrows and fades the top of a necklace image so the chain ends meet at the neck | `tryon/effects.py` | `_prepared` |
| `SpriteCache.get` | Remembers warped sprites so they are not recomputed each frame | `tryon/effects.py` | `_prepared` |

### Style advisor

| Function | What it does | Defined in | Used in |
|---|---|---|---|
| `normalise` | Centres, levels and scales the 478 landmarks for the model | `tryon/faceshape_model.py` | `ShapeModel.predict` |
| `ShapeModel.predict` | Returns the face shape and a probability for each shape | `tryon/faceshape_model.py` | `app.py` (advisor, live `Processor`) |
| `measure`, `extended_top` | Computes length/width, forehead/cheek and jaw/cheek ratios | `tryon/shape.py` | `classify`, `app.py` |
| `classify`, `alternatives` | Fallback rules that name a shape and any close alternatives | `tryon/shape.py` | `app.py` (only when the model is missing) |
| `neck_ratio`, `neck_class` | Chin-to-shoulder distance relative to face width, and short/average/long | `tryon/shape.py` | `app.py` |
| `recommend` | Returns the earring/necklace advice text for a shape and neck class | `tryon/shape.py` | `app.py` |

### Catalogue, input safety and setup

| Function | What it does | Defined in | Used in |
|---|---|---|---|
| `save_item`, `load_item`, `load_original`, `list_items`, `delete_item` | Store and read cut-outs and their original photos on disk | `tryon/catalog.py` | `app.py` |
| `safe_fetch` | Downloads an image from a URL with address, size, redirect and time limits | `tryon/net.py` | `app.py` (`fetch_cached`) |
| `validate_url` | Checks scheme, port and that every resolved address is public | `tryon/net.py` | `safe_fetch` |
| `safe_decode` | Checks image format and size with Pillow, then decodes with OpenCV | `tryon/net.py` | `app.py` (`cutout_cached`, `decode_user_image`) |
| `download_models.main` | Downloads each model file and verifies its SHA-256 | `download_models.py` | `setup_and_run.bat`, manual run |

### Interface helpers (`app.py`)

| Function | What it does |
|---|---|
| `photo_engine` | Creates one shared engine for still photos (with a lock) |
| `shape_model` | Loads the face-shape model once |
| `pick_shape` | Chooses the model's top shape, or the rule shape if the model is missing |
| `fetch_cached`, `cutout_cached`, `recut_cached` | Cache URL downloads and cut-outs between app reruns |
| `shrink`, `on_light`, `decode_user_image` | Resize a sprite, preview it on a light background, decode a user photo |
| `Processor.recv` | Handles each live video frame: analyse, render, return the frame |

### Filters and image operations

| Operation | What it does here | Defined in |
|---|---|---|
| Gaussian blur (3×3) | Softens the edge of the cut-out mask | `tryon/segment.py` (`color_key`, `grabcut`) |
| Gaussian blur (7×7) | Softens the hair mask before it is blended over the jewellery | `tryon/engine.py` (`_render`) |
| Gaussian blur (5×5) | Softens the skin mask used to clip rings and bracelets | `tryon/engine.py` (`_render`) |
| Gaussian blur (sigma = 3% of sprite width) | Spreads the contact shadow | `tryon/overlay.py` (`overlay`) |
| Morphological closing (5×5) | Fills small gaps in the object mask | `tryon/segment.py` (`color_key`) |
| Dilation (9×9) | Grows the skin mask so the band is not cut at the finger edge | `tryon/engine.py` (`_render`) |
| Colour distance in Lab space | Separates the object from the background colour | `tryon/segment.py` (`color_key`, `_clear_holes`) |
| Flood fill | Marks the background connected to the image corners | `tryon/segment.py` (`color_key`) |
| GrabCut | Foreground/background segmentation from a rectangle | `tryon/segment.py` (`grabcut`) |
| Connected components | Keeps the largest object; finds enclosed holes | `tryon/segment.py` |
| Affine warp (`warpAffine`, bilinear) | Rotates, scales and places the sprite on the frame | `tryon/overlay.py` |
| Remap (bilinear) | Cylindrical wrap of rings; squeeze of the top of necklaces | `tryon/effects.py` (`wrap_cylinder`, `drape`) |
| Area-averaging resize | Shrinks images without jagged edges (sprites, detection frame, brightness thumbnail) | `segment.py`, `overlay.py`, `engine.py`, `effects.py`, `app.py` |
| Nearest-neighbour resize | Scales the class mask to the frame size | `tryon/engine.py` (`_category`) |
| Alpha blending | Composites the sprite over the frame | `tryon/overlay.py` |
| Per-channel gain | Brightness and colour-temperature match (relighting) | `tryon/effects.py` (`Relighter`) |
| One Euro filter | Smooths landmark positions over time (adaptive low-pass in time) | `tryon/engine.py` (`OneEuro`) |
| Horizontal flip | Mirror view of the webcam; mirrored second earring | `app.py`, `tryon/engine.py` |
| Colour conversions (BGR↔RGB, BGR→Lab, gray→BGR) | Input format for MediaPipe, colour-key and rembg | `engine.py`, `segment.py`, `app.py` |

## 6. Flow of each function

**Add a product (sidebar, New image):** upload or URL → `safe_fetch` (URL only) → `safe_decode` →
`extract_accessory` → cut-out preview → optional *Save* → `save_item` stores the cut-out and the original.

**`extract_accessory`:** if the image already has transparency, crop it and stop → shrink to at most 800 px →
`color_key`, `grabcut` or `rembg_cut` (auto = `color_key`, then `grabcut` if the result covers more than 85% of
the image) → `_clear_holes` → `_largest_component` → soft edge → `_crop_to_alpha`.

**`color_key`:** take the median colour of the border → mark pixels far from it in Lab space as object → close
small gaps → flood-fill the background from the corners → `_clear_holes`.

**Catalogue re-cut:** choose an item → `load_original` → `extract_accessory` with the chosen method →
the new cut-out replaces the saved one for this session.

**`analyse`:** (live mode: timestamp) → shrink frame to 640 px → face accessories: face landmarker →
`estimate_yaw`; hand accessories: hand landmarker → `palm_facing` → landmarks pass through `OneEuro` in live
mode → necklace with shoulders on: `_shoulders` → returns `Scene`.

**`estimate_yaw`:** compare the nose position between the two face edges (sign and size) with the head-pose
matrix angle → average the two sizes.

**`palm_facing`:** cross product of wrist→index knuckle and wrist→pinky knuckle, compared with the left/right
label.

**`OneEuro`:** low smoothing cut-off when the point moves slowly, higher cut-off when it moves fast.

**`render`:** copy the frame → `_prepared` → `_render` → optional FPS text.

**`_prepared`:** necklace: `drape`; ring: `wrap_cylinder`; both cached in `SpriteCache` → if relighting is on:
`frame_light` → `Relighter` multiplies the sprite's colour channels.

**`_render` (face accessories):** `_earrings` or `_necklace` for each face → if hair occlusion is on: `_category`
→ blend the original pixels back where the mask says hair, scaled by the hair strength.

**`_render` (hand accessories):** for each hand: ring hidden if the palm faces the camera (when enabled), else
`_ring`; or `_bracelet` → if skin clip is on: `_category` → keep the accessory only on skin pixels (mask dilated).

**`_earrings`:** face height and width → head-yaw squash and far-ear fade → earlobe point = face edge moved
down to nose-base height, then offset slider → pendulum angle (live only) → `overlay` for each ear.

**`Pendulum.step`:** horizontal speed of the ear → acceleration → a few small physics sub-steps with damping →
angle, limited to ±1 rad.

**`_necklace`:** with shoulders: neck point between chin and shoulder midpoint, width from shoulder distance,
rotated with the shoulder line; without: below the chin, width from face width → `overlay`.

**`_ring`:** finger joint pair chosen in the sidebar → point along that segment → rotate along the finger →
width from segment length → `overlay`.

**`_bracelet`:** wrist point moved slightly toward the forearm → width from palm width → `overlay`.

**`overlay`:** build rotation/scale/anchor matrix → shrink large sprites with area filtering → warp to the
affected region only → optional soft shadow from the blurred alpha → alpha-blend into the frame.

**`render_compare`:** `render` with item A and with item B on the same `Scene` → left half from A, right half
from B → divider line and labels.

**Live video (`Processor.recv`):** frame from browser → mirror → `analyse` → (every 5th frame with a face:
`ShapeModel.predict`, stored for the advisor) → `render` or `render_compare` → frame back to the browser.

**Style advisor tab:** photo → `analyse` (with pose) → `ShapeModel.predict` → top shape and probabilities →
`measure` for the displayed ratios → `neck_ratio` → `neck_class` → `recommend` → text. If the model files are
missing: `classify` and `alternatives` instead.

**`ShapeModel.predict`:** `normalise` → scikit-learn pipeline (scaler, PCA, SVM) → probabilities per shape.

**`normalise`:** subtract the eye midpoint → rotate so the eyes are level → divide by the eye distance → flatten
the 478 points to 956 numbers.

**`safe_fetch`:** `validate_url` → connect to the checked IP → follow up to 3 redirects, re-checking each →
require an `image/*` content type → read up to the size limit.

**`safe_decode`:** check byte size → Pillow reads the header (format, width, height) → reject by limits →
OpenCV decodes.

**Catalogue functions:** `save_item` writes `<id>.png`, optionally `<id>.orig.png` and updates `index.json`;
ids are restricted to lowercase letters, digits, `_` and `-`, and paths are checked to stay inside `catalog/`.

**`download_models.main`:** for each model: download if missing → compare SHA-256 → delete the file on mismatch.

## 7. Claims:
### Accuracy of each part

| Part | Result | How it was tested |
|---|---|---|
| Face shape (Style advisor) | Right about 2 times out of 3 across the 5 shapes (random guessing would be 1 in 5). When the model is at least 70% sure, it is right about 4 times out of 5. Oblong is the easiest shape; Oval is the hardest. | 1,200 labelled photos: about 66% on 400 test photos and about 69% on 801 validation photos; the 4-in-5 figure comes from the test photos |
| Background removal on plain test images | Exact outline, including the hole in a ring | Drawn test images with a known outline |
| Smoothing (less shaking) | Much steadier landmarks | Noisy made-up position signal |
| Unsafe web addresses and bad uploads | Rejected | Unit tests |

### Future applications

- Online jewellery shops: let customers try items on from the product page before buying.
- In-store mirrors or kiosks that show many items without handling them.
- Phone apps and social-media filters for jewellery brands.
- Personal style assistants that recommend items from face shape and neck length.
- Ring and bracelet size estimation from hand landmarks.
- Other accessories such as glasses, hats, watches and scarves, using the same landmark approach.
- Better realism with 3-D head and hand models, true depth ordering, and models trained on jewellery photos.
- A shared catalogue for stores, with faster cloud processing.
