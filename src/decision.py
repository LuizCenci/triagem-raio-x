"""Camada de decisão de prioridade baseada em limiares ajustados na validação."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from src.evaluation import threshold_for_precision, threshold_for_recall

RECALL_TARGET: float = 0.90
HIGH_PRECISION: float = 0.80
INDET_MARGIN: float = 0.05

OUTPUTS: tuple[str, ...] = (
    "IMAGEM INADEQUADA",
    "INDETERMINADO",
    "PRIORIDADE ALTA",
    "PRIORIDADE MÉDIA",
    "ROTINA",
)


@dataclass(frozen=True)
class Thresholds:
    """Limiares de decisão de prioridade ajustados na validação."""

    t_rotina: float
    t_alta: float
    margin: float = INDET_MARGIN

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Thresholds":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def fit_thresholds(
    y_val,
    p_val,
    recall_target: float = RECALL_TARGET,
    high_precision: float = HIGH_PRECISION,
) -> Thresholds:
    """Ajusta `t_rotina` e `t_alta` exclusivamente sobre os dados de validação."""
    prob = np.asarray(p_val, dtype=float)
    t_rotina = threshold_for_recall(y_val, prob, recall_target)
    t_alta = threshold_for_precision(y_val, prob, high_precision)
    if t_alta is None:
        t_alta = float(np.percentile(prob, 90))
    t_alta = max(float(t_alta), float(t_rotina))
    return Thresholds(t_rotina=float(t_rotina), t_alta=t_alta)


def decide(p: float, th: Thresholds, inadequate: bool = False) -> str:
    """Converte `p(alterada)` em uma das 5 saídas do sistema na ordem definida."""
    if inadequate:
        return "IMAGEM INADEQUADA"
    prob = float(p)
    if abs(prob - th.t_rotina) < th.margin:
        return "INDETERMINADO"
    if prob >= th.t_alta:
        return "PRIORIDADE ALTA"
    if prob >= th.t_rotina:
        return "PRIORIDADE MÉDIA"
    return "ROTINA"


def decide_many(p, th: Thresholds, inadequate=None) -> np.ndarray:
    """Versão vetorizada de `decide` para arrays de probabilidades."""
    probs = np.asarray(p, dtype=float)
    flags = np.zeros(len(probs), dtype=bool) if inadequate is None else np.asarray(inadequate, dtype=bool)
    return np.array([decide(prob, th, bool(inad)) for prob, inad in zip(probs, flags)])
