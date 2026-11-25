import numpy as np
import pandas as pd
import pytest
from src.split import SPLIT_NAMES, patient_split, assert_no_patient_overlap, save_splits, load_splits


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    patients = np.repeat(np.arange(1000), rng.integers(1, 6, size=1000))
    return pd.DataFrame({"image": [f"{i}.png" for i in range(len(patients))],
                         "patient_id": patients,
                         "label": rng.integers(0, 2, size=len(patients))})


def test_keys_and_full_coverage(df):
    s = patient_split(df)
    assert tuple(s) == SPLIT_NAMES
    assert sum(len(p) for p in s.values()) == len(df)
    assert set(pd.concat(s.values())["image"]) == set(df["image"])


def test_no_patient_in_two_splits(df):
    s = patient_split(df)
    ids = [set(p["patient_id"]) for p in s.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_patient_proportions(df):
    s = patient_split(df)
    n = df["patient_id"].nunique()
    frac = {k: v["patient_id"].nunique() / n for k, v in s.items()}
    assert frac["test"] == pytest.approx(0.20, abs=0.01)
    assert frac["validation"] == pytest.approx(0.10, abs=0.01)
    assert frac["train"] == pytest.approx(0.70, abs=0.01)


def test_same_seed_same_split_and_different_seed_differs(df):
    a, b, c = patient_split(df, seed=42), patient_split(df, seed=42), patient_split(df, seed=7)
    assert a["test"]["image"].tolist() == b["test"]["image"].tolist()
    assert set(a["test"]["patient_id"]) != set(c["test"]["patient_id"])


def test_overlap_raises_with_split_names():
    s = {"train": pd.DataFrame({"patient_id": [1, 2]}),
         "validation": pd.DataFrame({"patient_id": [3]}),
         "test": pd.DataFrame({"patient_id": [2]})}
    with pytest.raises(ValueError, match="train e test: 1"):
        assert_no_patient_overlap(s)


def test_save_and_load_roundtrip(df, tmp_path):
    s = patient_split(df)
    save_splits(s, tmp_path / "splits")
    loaded = load_splits(tmp_path / "splits")
    for name in SPLIT_NAMES:
        pd.testing.assert_frame_equal(loaded[name], s[name])
