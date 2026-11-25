# Etapa 4 — Conjunto de teste degradado (Opção B): Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gerar, de forma reproduzível, 150 imagens degradadas a partir do teste, medir a queda de desempenho em relação às originais e alimentar a saída IMAGEM INADEQUADA com uma checagem simples de qualidade.

**Architecture:** `src/degrade.py` (seleção e transformações sobre arrays `uint8` em escala de cinza) e `src/quality.py` (indicadores de qualidade e limites calibrados na validação), ambos testados. O notebook `04_degradado.ipynb` gera o conjunto, roda a `resnet50` nas versões originais e degradadas e compara os resultados.

**Tech Stack:** numpy, OpenCV (`cv2`), pandas; `src.pipeline`, `src.evaluation` e `src.decision`.

**Spec:** `CLAUDE.md` §5; `docs/spec.md` §2.2 (Opção B; alternativa para imagens médicas, **a confirmar com o professor**).

## Global Constraints

- 150 imagens, **somente do split de teste**: 75 NORMAL + 75 ALTERADA; rótulos preservados.
- 30 por tipo: `ruido`, `rotacao`, `corte`, `baixo_contraste`, `combinada`, com o tipo sorteado de forma equilibrada entre as classes (15 NORMAL + 15 ALTERADA por tipo).
- Parâmetros:
  - ruído gaussiano com σ ∈ [10, 25] ou sal e pimenta com fração ∈ [0,02; 0,05], escolhidos ao acaso;
  - rotação com |ângulo| ∈ [10°, 30°], sinal aleatório, preenchimento 0 e mesmo tamanho;
  - corte que remove uma fração ∈ [0,20; 0,40] das bordas (distribuída aleatoriamente entre os 4 lados) e depois redimensiona para o tamanho original;
  - baixo contraste com `out = média + f·(img − média)` e f ∈ [0,3; 0,5];
  - combinada: 2 tipos distintos entre os 4 anteriores.
- `SEED = 42` (`np.random.default_rng`); parâmetros registrados por imagem.
- `degraded_metadata.csv` com as colunas `image, original_image, patient_id, label, class_name, degradation, params` (JSON). É gravado em `data/degraded/` e copiado para `results/degraded_metadata.csv` (versionado).
- Os limites de qualidade são calibrados nas imagens **originais da validação**, nunca no teste.

## Review Focus

1. **Saída fora de [0, 255]** após ruído ou contraste: a imagem é recortada para esse intervalo e mantém `uint8`.
2. **Reprodutibilidade:** a mesma semente gera os mesmos parâmetros e os mesmos pixels.
3. **Teste com menos de 75 imagens em uma classe:** `ValueError` com uma mensagem clara.
4. **Rotação e corte não reduzem contraste nem nitidez:** a checagem de qualidade não vai detectá-los. Isso precisa aparecer como limitação no notebook e no artigo, sem ser escondido.

---

### Task 1: `src/degrade.py`

**Files:**
- Create: `src/degrade.py`, `tests/test_degrade.py`

**Interfaces:**
- Produces:
  - `DEGRADATIONS: tuple[str, ...] = ("ruido", "rotacao", "corte", "baixo_contraste", "combinada")`
  - `noise(img, rng) -> tuple[np.ndarray, dict]`, `rotate(img, rng)`, `crop(img, rng)`, `low_contrast(img, rng)`, `combined(img, rng)`: cada uma devolve `(img_uint8_mesmo_shape, params)`; `params` sempre tem a chave `"tipo"`, mais os valores sorteados (`sigma`/`amount`, `angulo`, `fracao` + `lados`, `fator`, `etapas: list[dict]`)
  - `select_subset(test_df, n_per_class: int = 75, seed: int = SEED) -> pd.DataFrame`
  - `assign_degradations(subset: pd.DataFrame, seed: int = SEED) -> pd.Series`: 15 de cada tipo por classe
  - `build_degraded_set(subset: pd.DataFrame, out_dir: Path, seed: int = SEED) -> pd.DataFrame`: lê `subset.path` com `cv2.IMREAD_GRAYSCALE`, grava `{tipo}_{original_image}` em `out_dir` e devolve os metadados (também gravados em `out_dir/degraded_metadata.csv`)

- [ ] **Step 1: Testes que falham**

```python
# tests/test_degrade.py
import numpy as np
import pandas as pd
import pytest
from src.degrade import (DEGRADATIONS, noise, rotate, crop, low_contrast, combined,
                         select_subset, assign_degradations)

IMG = np.random.default_rng(1).integers(0, 256, (128, 128), dtype=np.uint8)

@pytest.mark.parametrize("fn", [noise, rotate, crop, low_contrast, combined])
def test_shape_dtype_and_type_key(fn):
    out, params = fn(IMG, np.random.default_rng(0))
    assert out.shape == IMG.shape and out.dtype == np.uint8 and "tipo" in params

@pytest.mark.parametrize("fn", [noise, rotate, crop, low_contrast, combined])
def test_reproducible(fn):
    a, pa = fn(IMG, np.random.default_rng(5)); b, pb = fn(IMG, np.random.default_rng(5))
    assert pa == pb and np.array_equal(a, b)

def test_param_ranges():
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert 10 <= abs(rotate(IMG, rng)[1]["angulo"]) <= 30
        assert 0.20 <= crop(IMG, rng)[1]["fracao"] <= 0.40
        assert 0.3 <= low_contrast(IMG, rng)[1]["fator"] <= 0.5
        steps = combined(IMG, rng)[1]["etapas"]
        assert len(steps) == 2 and steps[0]["tipo"] != steps[1]["tipo"]

def test_low_contrast_reduces_std():
    assert low_contrast(IMG, np.random.default_rng(0))[0].std() < IMG.std()

@pytest.fixture
def test_df():
    return pd.DataFrame({"image": [f"{i}.png" for i in range(300)],
                         "patient_id": range(300), "label": [0, 1] * 150})

def test_select_and_assign_balanced(test_df):
    sub = select_subset(test_df)
    assert len(sub) == 150 and sub.label.value_counts().tolist() == [75, 75]
    deg = assign_degradations(sub)
    table = pd.crosstab(deg, sub.label)
    assert set(table.index) == set(DEGRADATIONS) and (table.values == 15).all()

def test_select_raises_when_class_too_small(test_df):
    with pytest.raises(ValueError):
        select_subset(pd.concat([test_df[test_df.label == 0].head(80), test_df[test_df.label == 1].head(10)]))
```

- [ ] **Step 2:** `pytest tests/test_degrade.py -v` → FAIL
- [ ] **Step 3:** Implementar com `cv2.warpAffine` (rotação), `cv2.resize` (corte) e `np.clip` (ruído e contraste).
- [ ] **Step 4:** `pytest tests/test_degrade.py -v` → todos passam
- [ ] **Step 5:** `git commit -m "feat: geração do conjunto degradado"`

---

### Task 2: `src/quality.py`

**Files:**
- Create: `src/quality.py`, `tests/test_quality.py`

**Interfaces:**
- Produces:
  - `quality_features(img: np.ndarray) -> dict[str, float]`: `contraste` = desvio-padrão dos pixels; `nitidez` = variância de `cv2.Laplacian(img, cv2.CV_64F)`
  - `fit_quality_limits(features: pd.DataFrame, quantile: float = 0.01) -> dict[str, float]`: o quantil inferior de cada coluna
  - `is_inadequate(features: dict, limits: dict) -> bool`: `True` se algum indicador ficar abaixo do limite

- [ ] **Step 1: Testes que falham** — `test_flat_image_is_inadequate` (uma imagem constante dá contraste 0 → `True`, com limites `{contraste: 5, nitidez: 5}`); `test_random_image_is_adequate`; `test_fit_limits_quantile` (sobre um DataFrame com `contraste = range(100)`, o quantil 0,01 fica ≈ 0,99).
- [ ] **Step 2:** `pytest tests/test_quality.py -v` → FAIL
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** `pytest tests/test_quality.py -v` → 3 passed
- [ ] **Step 5:** `git commit -m "feat: checagem de qualidade de imagem"`

---

### Task 3: Notebook `04_degradado.ipynb`

**Files:**
- Create: `notebooks/04_degradado.ipynb`, `results/degraded_metadata.csv`, `results/quality_limits.json`, `results/metrics/{degraded_by_type,degraded_priority}.csv`, `results/figures/{degradado_exemplos,degradado_queda}.png`

**Interfaces:**
- Consumes: os splits + `attach_paths` (Etapa 1); `models/resnet50.keras`, `make_dataset` (Etapa 2); `binary_metrics`, `Thresholds.from_json`, `decide_many` (Etapa 3); Tasks 1 e 2
- Produces: `results/metrics/degraded_by_type.csv` com as colunas `degradation, versao (original|degradada)` + chaves de `binary_metrics`

Células:
- [ ] **Step 1:** Setup (o mesmo da Etapa 2), carregando o modelo do Drive e `th = Thresholds.from_json("results/thresholds.json")`.
- [ ] **Step 2:** `subset = select_subset(test)`; `meta = build_degraded_set(subset, DATA_DIR / "degraded")`; copiar para `results/degraded_metadata.csv`.
- [ ] **Step 3:** Figura `degradado_exemplos.png`: um par original × degradada para cada tipo.
- [ ] **Step 4:** Predizer as 150 originais e as 150 degradadas (`make_dataset`, `shuffle=False`, `preprocess_input` da ResNet-50).
- [ ] **Step 5:** `degraded_by_type.csv`: `binary_metrics` em `th.t_rotina`, por tipo e no total, para as duas versões. Figura `degradado_queda.png` com a queda de Recall, F1 e Balanced Accuracy por tipo.
- [ ] **Step 6:** Qualidade: `quality_features` nas originais da validação → `fit_quality_limits` → `results/quality_limits.json`. Aplicar `is_inadequate` às 150 degradadas e às 150 originais; mostrar a tabela da taxa de "inadequada" por tipo.
- [ ] **Step 7:** `degraded_priority.csv`: `decide_many(p, th, inadequate)` nas degradadas; tabela saída × tipo.
- [ ] **Step 8:** Markdown de conclusões: quais degradações mais prejudicam o modelo, se a checagem de qualidade pega ruído e contraste e a limitação para rotação e corte.
- [ ] **Step 9:** `pytest -v` → todos passam
- [ ] **Step 10:** `git add notebooks/04_degradado.ipynb results/ && git commit -m "feat: avaliação no conjunto degradado"`
