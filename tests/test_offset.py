import numpy as np
import pytest

from tryon.engine import Config, Scene, TryOnEngine, models_ready

pytestmark = pytest.mark.skipif(not models_ready(), reason="run download_models.py")


def _face():
    f = np.zeros((478, 2), np.float32)
    f[234], f[454] = (200, 200), (400, 200)      # face edges
    f[10], f[152] = (300, 80), (300, 330)        # forehead, chin  (face height 250)
    f[1], f[2] = (300, 220), (300, 245)          # nose tip, nose base
    f[93], f[132] = (190, 250), (195, 275)       # jaw points under the left ear
    f[323], f[361] = (410, 250), (405, 275)      # ... and the right ear
    return f


def _centroid_y(out, base):
    d = np.abs(out.astype(int) - base.astype(int)).sum(2) > 30
    ys = np.nonzero(d)[0]
    return float(ys.mean()) if len(ys) else None


@pytest.mark.parametrize("acc", ["Necklace", "Earrings"])
@pytest.mark.parametrize("with_shoulders", [True, False])
def test_vertical_offset_moves_accessory_clearly(acc, with_shoulders):
    if acc == "Earrings" and with_shoulders:
        pytest.skip("shoulders only affect the necklace")
    eng = TryOnEngine(live=False)
    base = np.full((600, 600, 3), 90, np.uint8)
    spr = np.zeros((120, 160, 4), np.uint8)
    spr[:] = (0, 0, 255, 255)
    shoulders = (np.array([100, 480], np.float32), np.array([500, 480], np.float32)) if with_shoulders else None
    ys = []
    for off in (-1.0, 0.0, 1.0):
        cfg = Config(accessory=acc, offset_y=off, relight=False, shadow=False, hair_occlusion=False, swing=False)
        scene = Scene(base, 0.0, faces=[_face()], yaws=[0.0], shoulders=shoulders)
        out, _ = eng.render(base, scene, spr, cfg)
        ys.append(_centroid_y(out, base))
    assert ys[0] < ys[1] < ys[2]                         # + offset = down
    assert ys[2] - ys[0] > 60                            # at least ~25% of face height across the slider
