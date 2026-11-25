"""Constantes do projeto."""
from pathlib import Path

SEED: int = 42
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.10

CLASSES: tuple[str, str] = ("NORMAL", "ALTERADA")
NORMAL_TAG: str = "No Finding"

AGE_BINS: tuple[int, ...] = (0, 18, 40, 60, 121)
AGE_LABELS: tuple[str, ...] = ("<18", "18-39", "40-59", "60+")

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
SPLITS_DIR: Path = DATA_DIR / "splits"
RESULTS_DIR: Path = ROOT / "results"

KAGGLE_DATASET: str = "nih-chest-xrays/sample"
