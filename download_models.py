"""Download MediaPipe models (~33 MB total) into ./models and verify their SHA-256.

The hashes were recorded from the first verified download (trust-on-first-use); a mismatch means the file
changed upstream or was tampered with, and the file is deleted instead of being loaded.
"""
import hashlib
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "face_landmarker.task": (f"{BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                             "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    "hand_landmarker.task": (f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                             "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
    # optional extras: shoulder-based necklace placement, hair / skin segmentation
    "pose_landmarker_lite.task": (f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                                  "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    "selfie_multiclass_256x256.tflite": (
        f"{BASE}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"),
}

# Third-party face-shape model (Rob4343/FaceShapeAI, MIT). Pickles: loaded only by tryon/faceshape_model.py,
# which uses a restricted unpickler with an exact class allow-list.
HF = "https://huggingface.co/Rob4343/FaceShapeAI/resolve/main"
FACESHAPE = {
    "face_shape_model.pkl": (f"{HF}/face_shape_model.pkl",
                             "790585e7c236f108d1131d4d295f28f68b2cbe2702f81ba500c5b8a9ec3294f2"),
    "label_encoder.pkl": (f"{HF}/label_encoder.pkl",
                          "4ed219dcc5c0195ddd7f21aed8a86714ba9bada51c3684961a9331bf072af1d7"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    out = Path(__file__).parent / "models"
    (out / "faceshape").mkdir(parents=True, exist_ok=True)
    bad = 0
    jobs = [(out / n, u, d) for n, (u, d) in MODELS.items()] +            [(out / "faceshape" / n, u, d) for n, (u, d) in FACESHAPE.items()]
    for dest, url, digest in jobs:
        name = dest.name
        if not dest.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, dest)
        if sha256(dest) != digest:
            print(f"CHECKSUM MISMATCH for {name}; deleting it")
            dest.unlink()
            bad += 1
        else:
            print(f"ok   {name} ({dest.stat().st_size // 1024} KB)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
