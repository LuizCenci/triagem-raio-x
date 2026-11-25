# Etapa 3 — Avaliação e camada de decisão: Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calcular as métricas da proposta, ajustar os limiares de prioridade na validação e converter `p(alterada)` nas 5 saídas do sistema, reportando tudo no teste.

**Architecture:** Funções puras em `src/evaluation.py` (métricas) e `src/decision.py` (limiares e decisão), testadas com arrays sintéticos. O notebook `03_avaliacao.ipynb` lê as predições da Etapa 2, ajusta os limiares na validação e aplica esses limiares no teste.

**Tech Stack:** numpy, pandas, scikit-learn (`metrics`), matplotlib/seaborn.

**Spec:** `CLAUDE.md` §2, §7 e §8; `docs/spec.md` §2.3.

## Global Constraints

- Limiares definidos **somente** na validação (proposta §2, §7, §8).
- Positivo = ALTERADA (`label == 1`).
- Meta de Recall da classe ALTERADA para PRIORIDADE ALTA + MÉDIA: **`RECALL_TARGET = 0.90`**.
- PRIORIDADE ALTA: `p ≥ t_alta`, em que `t_alta` é o menor limiar com Precisão ≥ **`HIGH_PRECISION = 0.80`** na validação; se nenhum limiar atingir esse valor, usar o percentil 90 de `p` na validação.
- INDETERMINADO: `|p − t_rotina| < INDET_MARGIN = 0.05`.
- Ordem da decisão: IMAGEM INADEQUADA → INDETERMINADO → ALTA → MÉDIA → ROTINA.
- Os textos das saídas são exatamente: `"IMAGEM INADEQUADA"`, `"INDETERMINADO"`, `"PRIORIDADE ALTA"`, `"PRIORIDADE MÉDIA"`, `"ROTINA"`.

## Review Focus

1. **Classe única em um subgrupo** (por exemplo, poucas imagens `<18`): AUROC e PR-AUC ficam `NaN`, sem exceção.
2. **Nenhum limiar atinge a meta de Recall:** `threshold_for_recall` devolve o menor `p` (todos positivos), sem erro.
3. **Empate de probabilidades no limiar:** `p == t` conta como positivo (`>=`) em todas as funções.
4. **`t_alta < t_rotina`** (modelo fraco): `t_alta` passa a valer `max(t_alta, t_rotina)`, para que as faixas não se invertam.

---

### Task 1: `src/evaluation.py`

**Files:**
- Create: `src/evaluation.py`, `tests/test_evaluation.py`

**Interfaces:**
- Produces:
  - `binary_metrics(y_true, p, threshold: float) -> dict[str, float]` com as chaves `recall, precision, f1, balanced_accuracy, specificity, auroc, pr_auc, tp, tn, fp, fn, n`; previsão = `p >= threshold`; `zero_division=0`; AUROC e PR-AUC (`average_precision_score`) valem `nan` se `y_true` tiver uma classe só
  - `threshold_for_recall(y_true, p, target: float) -> float`: o **maior** limiar `t` entre os valores únicos de `p` com recall(`p >= t`) ≥ `target`
  - `threshold_for_precision(y_true, p, target: float) -> float | None`: o **menor** limiar `t` entre os valores únicos de `p` com precisão ≥ `target` e pelo menos 1 positivo previsto; `None` se não houver
  - `subgroup_metrics(df: pd.DataFrame, group_col: str, threshold: float) -> pd.DataFrame`: uma linha por grupo (valores nulos ignorados), com as colunas `group` + chaves de `binary_metrics`; usa `df.label` e `df.p_alterada`

- [ ] **Step 1: Testes que falham**

```python
# tests/test_evaluation.py
import math
import numpy as np
import pandas as pd
from src.evaluation import binary_metrics, threshold_for_recall, threshold_for_precision, subgroup_metrics

Y = np.array([0, 0, 1, 1, 1, 0])
P = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.7])

def test_binary_metrics_counts():
    m = binary_metrics(Y, P, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2)
    assert m["recall"] == 2 / 3 and m["specificity"] == 2 / 3 and m["n"] == 6

def test_threshold_inclusive():
    assert binary_metrics(Y, P, 0.8)["tp"] == 2

def test_single_class_gives_nan_auc():
    m = binary_metrics(np.array([1, 1]), np.array([0.2, 0.9]), 0.5)
    assert math.isnan(m["auroc"]) and math.isnan(m["pr_auc"])

def test_threshold_for_recall():
    assert threshold_for_recall(Y, P, 1.0) == 0.35
    assert threshold_for_recall(Y, P, 0.6) == 0.8

def test_threshold_for_precision():
    assert threshold_for_precision(Y, P, 1.0) == 0.8
    assert threshold_for_precision(np.array([0, 1]), np.array([0.9, 0.1]), 1.0) is None

def test_subgroup_metrics_skips_null():
    df = pd.DataFrame({"label": Y, "p_alterada": P, "sex": ["M", "M", "F", "F", None, "M"]})
    out = subgroup_metrics(df, "sex", 0.5)
    assert sorted(out["group"]) == ["F", "M"] and out["n"].sum() == 5
```

- [ ] **Step 2:** `pytest tests/test_evaluation.py -v` → FAIL
- [ ] **Step 3:** Implementar conforme o bloco Interfaces.
- [ ] **Step 4:** `pytest tests/test_evaluation.py -v` → 6 passed
- [ ] **Step 5:** `git commit -m "feat: métricas de avaliação"`

---

### Task 2: `src/decision.py`

**Files:**
- Create: `src/decision.py`, `tests/test_decision.py`

**Interfaces:**
- Consumes: `threshold_for_recall`, `threshold_for_precision` (Task 1)
- Produces:
  - Constantes `RECALL_TARGET = 0.90`, `HIGH_PRECISION = 0.80`, `INDET_MARGIN = 0.05`, `OUTPUTS: tuple[str, ...]` (os 5 textos, na ordem da decisão)
  - `@dataclass(frozen=True) class Thresholds: t_rotina: float; t_alta: float; margin: float = INDET_MARGIN`, com `to_json(path)` e `Thresholds.from_json(path)`
  - `fit_thresholds(y_val, p_val, recall_target=RECALL_TARGET, high_precision=HIGH_PRECISION) -> Thresholds`: segue as Global Constraints, incluindo o fallback do percentil 90 e `t_alta = max(t_alta, t_rotina)`
  - `decide(p: float, th: Thresholds, inadequate: bool = False) -> str`
  - `decide_many(p: array, th: Thresholds, inadequate: array | None = None) -> np.ndarray[str]`

- [ ] **Step 1: Testes que falham**

```python
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
    y = np.array([1, 0, 1, 0]); p = np.array([0.1, 0.9, 0.2, 0.8])
    th = fit_thresholds(y, p)
    assert th.t_alta >= th.t_rotina

def test_json_roundtrip(tmp_path):
    TH.to_json(tmp_path / "t.json")
    assert Thresholds.from_json(tmp_path / "t.json") == TH
```

- [ ] **Step 2:** `pytest tests/test_decision.py -v` → FAIL
- [ ] **Step 3:** Implementar conforme o bloco Interfaces.
- [ ] **Step 4:** `pytest tests/test_decision.py -v` → 5 passed
- [ ] **Step 5:** `git commit -m "feat: camada de decisão de prioridade"`

---

### Task 3: Notebook `03_avaliacao.ipynb`

**Files:**
- Create: `notebooks/03_avaliacao.ipynb`, `results/thresholds.json`, `results/metrics/{test_overall,test_subgroups,test_priority}.csv`, `results/figures/{confusao_resnet50,roc_pr,prioridade}.png`

**Interfaces:**
- Consumes: `results/predictions/*.csv` (Etapa 2); Tasks 1 e 2
- Produces: `results/thresholds.json` (formato de `Thresholds.to_json`, ajustado na validação da `resnet50`), usado nas Etapas 4 e 5

Células:
- [ ] **Step 1:** Carregar as 4 predições. `th = fit_thresholds(val.label, val.p_alterada)` para a `resnet50`; gravar `results/thresholds.json`. Ajustar um `t_rotina` separado para o baseline com a mesma meta (para uma comparação justa).
- [ ] **Step 2:** `test_overall.csv` com `binary_metrics` no teste, em `t_rotina`, para os 2 modelos, mais uma coluna `limiar`. Imprimir também as métricas em 0,5 como referência.
- [ ] **Step 3:** Figuras: matriz de confusão da `resnet50` no teste (`confusao_resnet50.png`) e curvas ROC e PR dos 2 modelos (`roc_pr.png`).
- [ ] **Step 4:** `test_subgroups.csv`: `subgroup_metrics` para `sex`, `age_group` e `view` (coluna `variavel` + saída da função).
- [ ] **Step 5:** `test_priority.csv`: `decide_many` no teste (sem imagens inadequadas nesta etapa) e a tabela saída × classe (contagem e % das ALTERADAS em cada faixa); o recall de ALTA + MÉDIA fica em destaque. Gráfico de barras em `prioridade.png`.
- [ ] **Step 6:** Markdown de conclusões: o ganho da ResNet sobre o baseline, se a meta de 90% foi mantida no teste e os subgrupos com desempenho pior.
- [ ] **Step 7:** `pytest -v` → todos passam
- [ ] **Step 8:** `git add notebooks/03_avaliacao.ipynb results/ && git commit -m "feat: avaliação e limiares"`
