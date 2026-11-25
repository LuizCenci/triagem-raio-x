"""Divisão treino/validação/teste por paciente."""
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import SEED, TEST_SIZE, VAL_SIZE

SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")


def assert_no_patient_overlap(splits: dict[str, pd.DataFrame]) -> None:
    for a, b in combinations(SPLIT_NAMES, 2):
        common = set(splits[a]["patient_id"]) & set(splits[b]["patient_id"])
        if common:
            raise ValueError(f"Pacientes em comum entre {a} e {b}: {len(common)}")


def patient_split(df: pd.DataFrame, test_size: float = TEST_SIZE,
                  val_size: float = VAL_SIZE, seed: int = SEED) -> dict[str, pd.DataFrame]:
    groups = df["patient_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(gss.split(df, groups=groups))
    rest, test = df.iloc[rest_idx], df.iloc[test_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(gss_val.split(rest, groups=rest["patient_id"]))
    splits = {
        "train": rest.iloc[train_idx].reset_index(drop=True),
        "validation": rest.iloc[val_idx].reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    assert_no_patient_overlap(splits)
    return splits


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        splits[name].to_csv(out / f"{name}.csv", index=False)


def load_splits(out_dir: str | Path) -> dict[str, pd.DataFrame]:
    splits = {name: pd.read_csv(Path(out_dir) / f"{name}.csv") for name in SPLIT_NAMES}
    assert_no_patient_overlap(splits)
    return splits
