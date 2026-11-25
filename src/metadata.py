"""Carga dos metadados do NIH ChestX-ray14 e rótulo binário."""
import re
from pathlib import Path

import pandas as pd

from src.config import AGE_BINS, AGE_LABELS, CLASSES, NORMAL_TAG

_AGE_RE = re.compile(r"^(\d+)([YMD])$")
_COLUMNS = {
    "Image Index": "image",
    "Finding Labels": "finding_labels",
    "Patient ID": "patient_id",
    "Patient Age": "age",
    "Patient Gender": "sex",
    "View Position": "view",
}


def binary_label(finding_labels: str) -> int:
    """0 (NORMAL) se o único rótulo é 'No Finding'; 1 (ALTERADA) caso contrário."""
    return 0 if str(finding_labels).strip() == NORMAL_TAG else 1


def parse_age(value) -> int | None:
    """Idade em anos. Meses e dias viram 0; valores inválidos ou fora de [0, 120] viram None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        age = int(value)
    else:
        m = _AGE_RE.match(str(value).strip().upper())
        if not m:
            return None
        age = int(m.group(1)) if m.group(2) == "Y" else 0
    return age if 0 <= age <= 120 else None


def age_group(age: int | None) -> str | None:
    if age is None or pd.isna(age):
        return None
    for i, label in enumerate(AGE_LABELS):
        if AGE_BINS[i] <= age < AGE_BINS[i + 1]:
            return label
    return None


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    df = raw[list(_COLUMNS)].rename(columns=_COLUMNS)
    df["patient_id"] = df["patient_id"].astype(int)
    ages = [parse_age(v) for v in df["age"]]
    df["age"] = pd.array(ages, dtype="Int64")
    df["age_group"] = [age_group(a) for a in ages]
    df["label"] = df["finding_labels"].map(binary_label).astype(int)
    df["class_name"] = df["label"].map(lambda i: CLASSES[i])
    return df[["image", "finding_labels", "patient_id", "age", "age_group",
               "sex", "view", "label", "class_name"]]


def attach_paths(df: pd.DataFrame, images_root: str | Path) -> pd.DataFrame:
    """Adiciona a coluna `path` e descarta as linhas cuja imagem não existe em disco."""
    index = {p.name: str(p) for p in Path(images_root).rglob("*.png")}
    out = df.copy()
    out["path"] = out["image"].map(index)
    return out.dropna(subset=["path"]).reset_index(drop=True)
