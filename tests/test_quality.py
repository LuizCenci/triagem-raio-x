import numpy as np
import pandas as pd
from src.quality import quality_features, fit_quality_limits, is_inadequate

LIMITS = {"contraste": 5.0, "nitidez": 5.0}


def test_flat_image_is_inadequate():
    flat = np.full((64, 64), 128, dtype=np.uint8)
    f = quality_features(flat)
    assert f["contraste"] == 0 and is_inadequate(f, LIMITS)


def test_random_image_is_adequate():
    img = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
    assert not is_inadequate(quality_features(img), LIMITS)


def test_fit_limits_quantile():
    feats = pd.DataFrame({"contraste": range(100), "nitidez": range(100)})
    limits = fit_quality_limits(feats, quantile=0.01)
    assert abs(limits["contraste"] - 0.99) < 1e-9 and set(limits) == {"contraste", "nitidez"}
