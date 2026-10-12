import os
import pickle

import numpy as np
import pytest

from tryon import faceshape_model as fm


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_restricted_unpickler_blocks_arbitrary_code(tmp_path):
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps(_Evil()))
    with pytest.raises(pickle.UnpicklingError):
        fm._safe_load(p)


def test_restricted_unpickler_blocks_eval_and_other_sklearn(tmp_path):
    class _Eval:
        def __reduce__(self):
            return (eval, ("1+1",))
    p = tmp_path / "e.pkl"
    p.write_bytes(pickle.dumps(_Eval()))
    with pytest.raises(pickle.UnpicklingError):
        fm._safe_load(p)


def test_normalise_puts_irises_at_half_unit_and_is_rotation_scale_invariant():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(478, 2)) * 40 + 300
    pts[468], pts[473] = (260, 300), (340, 300)             # level eyes, 80 px apart
    v = fm.normalise(pts).reshape(-1, 2)
    assert np.allclose(v[468], (0.5, 0.0), atol=1e-6) and np.allclose(v[473], (-0.5, 0.0), atol=1e-6)
    a = np.radians(17)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    moved = (pts - 300) @ R.T * 2.5 + 77                     # rotate, scale, translate
    assert np.allclose(fm.normalise(moved), fm.normalise(pts), atol=1e-6)


@pytest.mark.skipif(not fm.available(), reason="run download_models.py")
def test_model_predicts_a_distribution():
    m = fm.ShapeModel()
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(478, 2)) * 30
    pts[468], pts[473] = (50, 0), (-50, 0)
    label, probs = m.predict(pts)
    assert label in probs and abs(sum(probs.values()) - 1) < 1e-6
    assert set(probs) == {"Heart", "Oblong", "Oval", "Round", "Square"}
