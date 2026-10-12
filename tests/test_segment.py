import cv2
import numpy as np
from tryon.segment import extract_accessory


def ring_photo():
    img = np.full((400, 400, 3), 245, np.uint8)
    cv2.ellipse(img, (200, 220), (120, 120), 0, 0, 360, (40, 160, 210), 28)
    return img


def test_color_key_clears_ring_hole_and_crops():
    out = extract_accessory(ring_photo(), "color-key")
    h, w = out.shape[:2]
    assert out.shape[2] == 4 and h < 400
    assert out[h // 2, w // 2, 3] == 0                # centre of the ring is see-through
    assert (out[..., 3] > 127).mean() < 0.5


def test_grabcut_also_clears_hole():
    out = extract_accessory(ring_photo(), "grabcut")
    h, w = out.shape[:2]
    assert out[h // 2, w // 2, 3] == 0


def test_alpha_input_is_passed_through():
    rgba = np.zeros((50, 50, 4), np.uint8)
    rgba[10:30, 10:30] = (1, 2, 3, 255)
    out = extract_accessory(rgba)
    assert out.shape[:2] == (24, 24)
