import numpy as np
import pytest

from tryon.engine import ACCESSORIES, Config, OneEuro, TryOnEngine, estimate_yaw, models_ready, palm_facing

needs_models = pytest.mark.skipif(not models_ready(), reason="run download_models.py")


def _rms_second_difference(series):
    s = np.asarray(series)
    return float(np.sqrt(((s[2:] - 2 * s[1:-1] + s[:-2]) ** 2).sum(1).mean()))


def test_one_euro_reduces_jitter_on_noisy_static_signal():
    rng = np.random.default_rng(1)
    f = OneEuro()
    raw = [(np.array([[100.0, 100.0]]) + rng.normal(scale=1.5, size=(1, 2))).astype(np.float32) for _ in range(150)]
    flt = [np.array(f(x, i / 30)) for i, x in enumerate(raw)]
    assert _rms_second_difference([x[0] for x in flt]) < 0.4 * _rms_second_difference([x[0] for x in raw])


def test_one_euro_follows_fast_motion_with_low_lag():
    f = OneEuro()
    out = None
    for i in range(60):
        out = f(np.array([[float(i * 20), 0.0]], np.float32), i / 30)
    assert abs(out[0, 0] - 59 * 20) < 60                      # lag under ~3 frames of motion


def test_palm_detection_is_label_geometry_consistent():
    p = np.zeros((21, 2), np.float32)
    p[5], p[17] = (-1, -2), (1, -2)                           # thumb side left
    assert palm_facing(p, "Right") is True and palm_facing(p, "Left") is False
    p[5], p[17] = (1, -2), (-1, -2)                           # mirrored geometry
    assert palm_facing(p, "Right") is False and palm_facing(p, "Left") is True


def test_yaw_sign_from_nose_position():
    f = np.zeros((478, 2), np.float32)
    f[234], f[454] = (0, 0), (100, 0)
    f[1] = (80, 0)
    assert estimate_yaw(f, None) > 0
    f[1] = (20, 0)
    assert estimate_yaw(f, None) < 0
    f[1] = (50, 0)
    assert abs(estimate_yaw(f, None)) < 1e-3


@needs_models
@pytest.mark.parametrize("acc", ACCESSORIES)
@pytest.mark.parametrize("live", [False, True])
def test_engine_runs_on_blank_frame(acc, live):
    eng = TryOnEngine(live=live)
    spr = np.zeros((60, 60, 4), np.uint8)
    spr[10:50, 10:50] = (0, 0, 255, 255)
    frame = np.full((240, 320, 3), 100, np.uint8)
    cfg = Config(accessory=acc)
    for _ in range(3):
        scene = eng.analyse(frame, cfg)
        out, found = eng.render(frame, scene, spr, cfg)
    assert out.shape == frame.shape and found is False
    assert (out == frame).all()                                # nothing detected -> frame untouched
