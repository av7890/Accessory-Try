import numpy as np
from tryon.physics import Pendulum
from tryon.shape import CHEEK_L, CHEEK_R, CHIN, FOREHEAD_L, FOREHEAD_R, JAW_L, JAW_R, TOP, RECS
from tryon.shape import alternatives, classify, measure, neck_class, neck_ratio, recommend


def face(length, cheek, fore, jaw):
    f = np.zeros((478, 2), np.float32)
    f[CHEEK_L], f[CHEEK_R] = (-cheek / 2, 0), (cheek / 2, 0)
    f[FOREHEAD_L], f[FOREHEAD_R] = (-fore / 2, -1), (fore / 2, -1)
    f[JAW_L], f[JAW_R] = (-jaw / 2, 1), (jaw / 2, 1)
    f[TOP], f[CHIN] = (0, -length / 2), (0, length / 2)
    return f


def test_classification_rules():
    assert classify(measure(face(118, 100, 85, 85))) == "Oval"
    assert classify(measure(face(100, 100, 86, 80))) == "Round"
    assert classify(measure(face(100, 100, 95, 95))) == "Square"
    assert classify(measure(face(140, 100, 98, 98))) == "Oblong"
    assert classify(measure(face(110, 100, 98, 70))) == "Heart"
    assert classify(measure(face(120, 100, 75, 75))) == "Diamond"


def test_recommendations_cover_all_shapes_and_neck():
    for s in RECS:
        r = recommend(s, "short")
        assert r["earrings"] and r["necklace"] and r["avoid"] and r["neck"]


def test_neck_ratio():
    f = face(100, 100, 90, 90)
    sh = (np.array([-100, 150], np.float32), np.array([100, 150], np.float32))
    r = neck_ratio(f, sh)
    assert neck_class(r) in ("short", "average", "long")
    assert neck_ratio(f, None) is None and neck_class(None) is None


def test_pendulum_rests_swings_and_settles():
    p = Pendulum()
    for i in range(200):
        assert p.step(100.0, 40.0, 20000.0, i * 0.033) == 0.0     # no pivot motion -> stays at rest
    p = Pendulum()
    x = 100.0
    peaks = []
    for i in range(30):
        x += 40.0 if i < 5 else 0.0                               # shove the pivot sideways
        peaks.append(abs(p.step(x, 40.0, 20000.0, i * 0.033)))
    assert max(peaks) > 0.05                                      # it swings
    th = 1.0
    for i in range(30, 400):
        th = p.step(x, 40.0, 20000.0, i * 0.033)
    assert abs(th) < 0.02 and abs(p.omega) < 0.2                  # and settles
    assert all(v <= Pendulum.MAX_THETA for v in peaks)            # bounded


def test_alternatives_flag_borderline_only():
    safe = measure(face(140, 100, 98, 98))                    # clearly Oblong
    assert alternatives(safe, margin=0.01) == []
    edge = {"length_to_width": 1.44, "forehead_to_cheek": 0.85, "jaw_to_cheek": 0.76}
    assert classify(edge) == "Oval" and "Oblong" in alternatives(edge)
