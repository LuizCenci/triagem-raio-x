import pandas as pd
import pytest
from src.metadata import binary_label, parse_age, age_group, load_metadata, attach_paths


@pytest.mark.parametrize("raw,expected", [
    ("No Finding", 0), (" No Finding ", 0),
    ("Effusion", 1), ("Hernia|Mass", 1), ("Infiltration|Effusion|Atelectasis", 1),
])
def test_binary_label(raw, expected):
    assert binary_label(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("058Y", 58), ("011M", 0), ("005D", 0), (47, 47), (47.0, 47),
    (414, None), ("abc", None), (None, None), (-1, None),
])
def test_parse_age(raw, expected):
    assert parse_age(raw) == expected


@pytest.mark.parametrize("age,expected", [
    (0, "<18"), (17, "<18"), (18, "18-39"), (39, "18-39"), (40, "40-59"), (60, "60+"), (120, "60+"), (None, None),
])
def test_age_group(age, expected):
    assert age_group(age) == expected


@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({
        "Image Index": ["a.png", "b.png", "c.png"],
        "Finding Labels": ["No Finding", "Mass|Nodule", "Effusion"],
        "Follow-up #": [0, 1, 0],
        "Patient ID": [1, 1, 2],
        "Patient Age": ["030Y", "031Y", "414Y"],
        "Patient Gender": ["M", "M", "F"],
        "View Position": ["PA", "AP", "PA"],
    })
    path = tmp_path / "labels.csv"
    df.to_csv(path, index=False)
    return path


def test_load_metadata_columns_and_values(csv_file):
    df = load_metadata(csv_file)
    assert list(df.columns) == ["image", "finding_labels", "patient_id", "age", "age_group",
                                "sex", "view", "label", "class_name"]
    assert df["label"].tolist() == [0, 1, 1]
    assert df["class_name"].tolist() == ["NORMAL", "ALTERADA", "ALTERADA"]
    assert df["age"].tolist()[:2] == [30, 31] and pd.isna(df["age"].iloc[2])
    assert df["age_group"].tolist()[:2] == ["18-39", "18-39"]


def test_attach_paths_drops_missing_and_finds_nested(csv_file, tmp_path):
    nested = tmp_path / "images_001" / "images"
    nested.mkdir(parents=True)
    (nested / "a.png").write_bytes(b"")
    (tmp_path / "c.png").write_bytes(b"")
    df = load_metadata(csv_file)
    out = attach_paths(df, tmp_path)
    assert sorted(out["image"]) == ["a.png", "c.png"]
    assert out.set_index("image").loc["a.png", "path"].endswith("a.png")
    assert len(df) == 3 and "path" not in df.columns
