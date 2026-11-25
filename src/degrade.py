"""Geração reprodutível do conjunto de teste degradado (150 imagens do split de teste)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import CLASSES, SEED

DEGRADATIONS: tuple[str, ...] = ("ruido", "rotacao", "corte", "baixo_contraste", "combinada")
_BASE = ("ruido", "rotacao", "corte", "baixo_contraste")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def noise(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Ruído gaussiano (sigma 10-25) ou sal e pimenta (fração 0,02-0,05), sorteado."""
    if rng.random() < 0.5:
        sigma = float(rng.uniform(10, 25))
        out = _to_uint8(img.astype(np.float64) + rng.normal(0, sigma, img.shape))
        return out, {"tipo": "ruido", "modo": "gaussiano", "sigma": sigma}
    amount = float(rng.uniform(0.02, 0.05))
    out = img.copy()
    mask = rng.random(img.shape)
    out[mask < amount / 2] = 0
    out[(mask >= amount / 2) & (mask < amount)] = 255
    return out, {"tipo": "ruido", "modo": "sal_pimenta", "amount": amount}


def rotate(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Rotação com |ângulo| entre 10° e 30° (sinal aleatório), fundo preto, mesmo tamanho."""
    angle = float(rng.uniform(10, 30) * rng.choice([-1, 1]))
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, {"tipo": "rotacao", "angulo": angle}


def crop(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Remove 20-40% de cada dimensão, repartido entre as bordas opostas, e reduz/amplia ao tamanho original."""
    frac = float(rng.uniform(0.20, 0.40))
    h, w = img.shape
    split_v, split_h = float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.3, 0.7))
    sides = {"top": frac * split_v, "bottom": frac * (1 - split_v),
             "left": frac * split_h, "right": frac * (1 - split_h)}
    y0, y1 = int(round(sides["top"] * h)), h - int(round(sides["bottom"] * h))
    x0, x1 = int(round(sides["left"] * w)), w - int(round(sides["right"] * w))
    out = cv2.resize(img[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LINEAR)
    return out, {"tipo": "corte", "fracao": frac, "lados": sides}


def low_contrast(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Contraste reduzido: out = média + f*(img - média), f entre 0,3 e 0,5."""
    f = float(rng.uniform(0.3, 0.5))
    mean = float(img.mean())
    return _to_uint8(mean + f * (img.astype(np.float64) - mean)), {"tipo": "baixo_contraste", "fator": f}


_FUNCS = {"ruido": noise, "rotacao": rotate, "corte": crop, "baixo_contraste": low_contrast}


def combined(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Aplica duas degradações distintas entre as quatro básicas, em ordem sorteada."""
    names = [str(n) for n in rng.choice(_BASE, size=2, replace=False)]
    steps, out = [], img
    for name in names:
        out, p = _FUNCS[name](out, rng)
        steps.append(p)
    return out, {"tipo": "combinada", "etapas": steps}


_FUNCS["combinada"] = combined


def select_subset(test_df: pd.DataFrame, n_per_class: int = 75, seed: int = SEED) -> pd.DataFrame:
    parts = []
    for label in (0, 1):
        cls = test_df[test_df["label"] == label]
        if len(cls) < n_per_class:
            raise ValueError(f"Classe {label} tem {len(cls)} imagens no teste; são necessárias {n_per_class}")
        parts.append(cls.sample(n=n_per_class, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def assign_degradations(subset: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Distribui os tipos de forma equilibrada dentro de cada classe (n/5 de cada tipo por classe)."""
    rng = np.random.default_rng(seed)
    out = pd.Series(index=subset.index, dtype=object, name="degradation")
    for label in (0, 1):
        idx = subset.index[subset["label"] == label]
        if len(idx) % len(DEGRADATIONS):
            raise ValueError(f"Classe {label}: {len(idx)} imagens não é múltiplo de {len(DEGRADATIONS)}")
        kinds = np.repeat(DEGRADATIONS, len(idx) // len(DEGRADATIONS))
        out.loc[idx] = rng.permutation(kinds)
    return out


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler {path}")
    return img


def build_degraded_set(subset: pd.DataFrame, out_dir: Path, seed: int = SEED) -> pd.DataFrame:
    """Grava as imagens degradadas e o `degraded_metadata.csv` em `out_dir`; devolve os metadados."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = assign_degradations(subset, seed)
    rng = np.random.default_rng(seed)
    rows = []
    for i, row in subset.iterrows():
        kind = kinds.loc[i]
        img, params = _FUNCS[kind](_read_gray(row["path"]), rng)
        name = f"{kind}_{row['image']}"
        ok, buf = cv2.imencode(".png", img)
        buf.tofile(str(out_dir / name))
        class_name = row["class_name"] if "class_name" in subset.columns else CLASSES[int(row["label"])]
        rows.append({"image": name, "original_image": row["image"], "patient_id": row["patient_id"],
                     "label": int(row["label"]), "class_name": class_name, "degradation": kind,
                     "params": json.dumps(params, ensure_ascii=False)})
    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "degraded_metadata.csv", index=False)
    return meta
