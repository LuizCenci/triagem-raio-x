import pytest
from src.download import find_metadata_csv


def test_finds_sample_csv_nested(tmp_path):
    (tmp_path / "sample").mkdir()
    (tmp_path / "sample" / "sample_labels.csv").write_text("x")
    assert find_metadata_csv(tmp_path).name == "sample_labels.csv"


def test_finds_full_dataset_csv(tmp_path):
    (tmp_path / "Data_Entry_2017.csv").write_text("x")
    assert find_metadata_csv(tmp_path).name == "Data_Entry_2017.csv"


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_metadata_csv(tmp_path)
