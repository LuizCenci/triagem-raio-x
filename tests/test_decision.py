# tests/test_decision.py
import numpy as np
from src.decision import Thresholds, fit_thresholds, decide, decide_many

TH = Thresholds(t_rotina=0.3, t_alta=0.7, margin=0.05)


def test_decide_order():
    assert decide(0.9, TH, inadequate=True) == "IMAGEM INADEQUADA"
    assert decide(0.32, TH) == "INDETERMINADO"
    assert decide(0.7, TH) == "PRIORIDADE ALTA"
    assert decide(0.5, TH) == "PRIORIDADE MÉDIA"
    assert decide(0.1, TH) == "ROTINA"


def test_decide_many_matches_decide():
    p = np.array([0.1, 0.5, 0.9])
    assert decide_many(p, TH).tolist() == ["ROTINA", "PRIORIDADE MÉDIA", "PRIORIDADE ALTA"]
    assert decide_many(p, TH, np.array([False, True, False]))[1] == "IMAGEM INADEQUADA"


def test_fit_thresholds_meets_recall_on_validation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 500), 0, 1)
    th = fit_thresholds(y, p)
    assert (p[y == 1] >= th.t_rotina).mean() >= 0.90
    assert th.t_alta >= th.t_rotina


def test_fit_thresholds_fallback_when_precision_unreachable():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    th = fit_thresholds(y, p)
    assert th.t_alta >= th.t_rotina


def test_json_roundtrip(tmp_path):
    TH.to_json(tmp_path / "t.json")
    assert Thresholds.from_json(tmp_path / "t.json") == TH
