"""Download da base NIH ChestX-ray14 (amostra do Kaggle) e localização dos metadados."""
from pathlib import Path

from src.config import KAGGLE_DATASET

_CSV_NAMES = ("sample_labels.csv", "Data_Entry_2017.csv")


def download_dataset(dataset: str = KAGGLE_DATASET) -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(dataset))


def find_metadata_csv(root: str | Path) -> Path:
    root = Path(root)
    for name in _CSV_NAMES:
        found = next(root.rglob(name), None)
        if found:
            return found
    raise FileNotFoundError(f"Nenhum de {_CSV_NAMES} encontrado em {root}")
