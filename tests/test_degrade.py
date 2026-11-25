import json

import cv2
import numpy as np
import pandas as pd
import pytest
from src.degrade import (DEGRADATIONS, noise, rotate, crop, low_contrast, combined,
                         select_subset, assign_degradations, build_degraded_set)

IMG = np.random.default_rng(1).integers(0, 256, (128, 128), dtype=np.uint8)


@pytest.mark.parametrize("fn", [noise, rotate, crop, low_contrast, combined])
def test_shape_dtype_and_type_key(fn):
    out, params = fn(IMG, np.random.default_rng(0))
    assert out.shape == IMG.shape and out.dtype == np.uint8 and "tipo" in params


@pytest.mark.parametrize("fn", [noise, rotate, crop, low_contrast, combined])
def test_reproducible(fn):
    a, pa = fn(IMG, np.random.default_rng(5))
    b, pb = fn(IMG, np.random.default_rng(5))
    assert pa == pb and np.array_equal(a, b)


def test_param_ranges():
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert 10 <= abs(rotate(IMG, rng)[1]["angulo"]) <= 30
        assert 0.20 <= crop(IMG, rng)[1]["fracao"] <= 0.40
        assert 0.3 <= low_contrast(IMG, rng)[1]["fator"] <= 0.5
        steps = combined(IMG, rng)[1]["etapas"]
        assert len(steps) == 2 and steps[0]["tipo"] != steps[1]["tipo"]


def test_noise_params_in_range():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = noise(IMG, rng)[1]
        if "sigma" in p:
            assert 10 <= p["sigma"] <= 25
        else:
            assert 0.02 <= p["amount"] <= 0.05


def test_low_contrast_reduces_std():
    assert low_contrast(IMG, np.random.default_rng(0))[0].std() < IMG.std()


def test_noise_stays_in_uint8_range_on_saturated_image():
    white = np.full((64, 64), 255, dtype=np.uint8)
    out, _ = noise(white, np.random.default_rng(0))
    assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255


@pytest.fixture
def test_df():
    return pd.DataFrame({"image": [f"{i}.png" for i in range(300)],
                         "patient_id": range(300), "label": [0, 1] * 150})


def test_select_and_assign_balanced(test_df):
    sub = select_subset(test_df)
    assert len(sub) == 150 and sub.label.value_counts().tolist() == [75, 75]
    deg = assign_degradations(sub)
    table = pd.crosstab(deg, sub.label)
    assert set(table.index) == set(DEGRADATIONS) and (table.values == 15).all()


def test_select_raises_when_class_too_small(test_df):
    with pytest.raises(ValueError):
        select_subset(pd.concat([test_df[test_df.label == 0].head(80), test_df[test_df.label == 1].head(10)]))


def test_build_degraded_set_writes_files_and_metadata(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    rows = []
    for i in range(10):
        p = src_dir / f"{i}.png"
        cv2.imwrite(str(p), np.random.default_rng(i).integers(0, 256, (64, 64), dtype=np.uint8))
        rows.append({"image": f"{i}.png", "patient_id": i, "label": i % 2,
                     "class_name": ["NORMAL", "ALTERADA"][i % 2], "path": str(p)})
    sub = pd.DataFrame(rows)
    out_dir = tmp_path / "degraded"
    meta = build_degraded_set(sub, out_dir, seed=1)
    assert list(meta.columns) == ["image", "original_image", "patient_id", "label", "class_name",
                                  "degradation", "params"]
    assert len(meta) == 10 and (out_dir / "degraded_metadata.csv").exists()
    assert all((out_dir / f).exists() for f in meta["image"])
    assert meta["label"].tolist() == sub["label"].tolist()
    assert json.loads(meta["params"].iloc[0])["tipo"] in DEGRADATIONS
    again = build_degraded_set(sub, tmp_path / "degraded2", seed=1)
    assert again["params"].tolist() == meta["params"].tolist()
