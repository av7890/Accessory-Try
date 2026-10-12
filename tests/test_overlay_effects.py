import numpy as np
from tryon.effects import Relighter, drape, frame_light, wrap_cylinder
from tryon.overlay import overlay


def sprite(color=(0, 0, 255), size=40):
    s = np.zeros((size, size, 4), np.uint8)
    s[...] = (*color, 255)
    return s


def test_overlay_lands_on_target():
    f = np.zeros((200, 200, 3), np.uint8)
    overlay(f, sprite(), (0.5, 0.5), (100, 100), (0, -1), 40)
    assert f[100, 100].tolist() == [0, 0, 255]
    assert f[10, 10].sum() == 0


def test_overlay_opacity_and_offscreen():
    f = np.zeros((100, 100, 3), np.uint8)
    overlay(f, sprite(), (0.5, 0.5), (50, 50), (0, -1), 30, opacity=0.0)
    assert f.sum() == 0
    overlay(f, sprite(), (0.5, 0.5), (-500, -500), (0, -1), 30)   # fully off screen: no crash
    assert f.sum() == 0


def test_overlay_rotation_moves_pixels():
    s = np.zeros((40, 40, 4), np.uint8)
    s[:10, :, :] = (0, 255, 0, 255)                    # green band on the sprite's top
    f = np.zeros((200, 200, 3), np.uint8)
    overlay(f, s, (0.5, 0.5), (100, 100), (1, 0), 80)  # up -> right: green ends up on the right
    ys, xs = np.nonzero(f[..., 1])
    assert xs.mean() > 100


def test_shadow_darkens_background_but_not_sprite():
    f = np.full((200, 200, 3), 200, np.uint8)
    overlay(f, sprite(), (0.5, 0.5), (100, 100), (0, -1), 60, shadow=1.0)
    assert f.min() < 200 and f[100, 100].tolist() == [0, 0, 255]


def test_wrap_and_drape_shapes():
    s = sprite(size=60)
    w = wrap_cylinder(s, 0.8)
    assert w.shape == s.shape and w[30, 30, 3] == 255
    assert wrap_cylinder(s, 0.0) is s
    d = drape(s)
    assert d.shape == s.shape and d[0, :, 3].max() == 0 and d[-1, :, 3].min() == 255


def test_relight_darkens_for_dark_scene_and_brightens_for_bright():
    s = sprite((200, 200, 200))
    r = Relighter()
    dark = r(s, frame_light(np.full((90, 160, 3), 40, np.uint8)), 1.0)
    bright = Relighter()(sprite((60, 60, 60)), frame_light(np.full((90, 160, 3), 200, np.uint8)), 1.0)
    assert dark[10, 10, :3].mean() < 200
    assert bright[10, 10, :3].mean() > 60
    assert r(s, frame_light(np.zeros((9, 16, 3), np.uint8)), 0.0) is s
