import pytest
from src import config


def test_split_fractions_sum_below_one():
    assert config.TEST_SIZE + config.VAL_SIZE == pytest.approx(0.30)


def test_classes_order_matches_labels():
    assert config.CLASSES[0] == "NORMAL" and config.CLASSES[1] == "ALTERADA"


def test_age_bins_match_labels():
    assert len(config.AGE_BINS) == len(config.AGE_LABELS) + 1


def test_paths_are_inside_repo():
    assert config.SPLITS_DIR.parent == config.DATA_DIR
    assert (config.ROOT / "requirements.txt").exists()
