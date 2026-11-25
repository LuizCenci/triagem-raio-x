# Etapa 1 — Dados e divisão por paciente: Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Baixar uma amostra do NIH ChestX-ray14, gerar os rótulos binários NORMAL/ALTERADA e produzir divisões treino/validação/teste por paciente, verificadas e versionadas como CSV.

**Architecture:** Módulos puros em `src/` (configuração, metadados, divisão, download) testados com pytest sobre DataFrames sintéticos; um notebook `01_eda.ipynb`, executável no Colab, que usa esses módulos para a análise exploratória e grava `data/splits/{train,validation,test}.csv`. As etapas seguintes consomem apenas esses CSVs.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn (`GroupShuffleSplit`), kagglehub, matplotlib/seaborn, pytest. TensorFlow/Keras entra no `requirements.txt` agora, mas só é usado a partir da Etapa 2.

**Spec:** `docs/spec.md` (especificação da disciplina) + `CLAUDE.md` (proposta da equipe — Entrega 1).

## Visão geral das etapas

| Etapa | Plano | Entrega | Formato |
|---|---|---|---|
| **1. Dados e divisão** | este arquivo | metadados limpos, rótulo binário, splits por paciente, EDA | módulos + testes + notebook |
| 2. Treinamento | `2026-11-25-etapa2-treinamento.md` | baseline + ResNet-50 em 2 fases; predições de validação e teste | notebook |
| 3. Avaliação e decisão | `2026-11-25-etapa3-avaliacao.md` | métricas, limiares na validação, camada de decisão com 5 saídas | módulos + testes + notebook |
| 4. Conjunto degradado | `2026-11-25-etapa4-degradado.md` | 150 imagens degradadas (Opção B), checagem de qualidade, comparação original × degradado | módulos + testes + notebook |
| 5. Grad-CAM e exemplos | `2026-11-25-etapa5-gradcam.md` | 12+ mapas Grad-CAM, figuras de acertos e erros | notebook |
| 6. Artigo e apresentação | `2026-11-25-etapa6-artigo.md` | PDF SBC de 4 a 6 páginas; slides para 8 minutos | documentos |

## Global Constraints

- Framework: TensorFlow/Keras (spec §2.3: "TensorFlow/Keras ou Ultralytics"). PyTorch sai do `requirements.txt`.
- Precisa rodar no Google Colab gratuito (spec §2.3, §5).
- Semente fixa `SEED = 42` em toda operação aleatória.
- Divisão **por paciente**: ~70% treino / 10% validação / 20% teste, com `GroupShuffleSplit` e `Patient ID` como grupo (proposta §4).
- Rótulo: exatamente `No Finding` → NORMAL (0); qualquer outro conteúdo → ALTERADA (1) (proposta §3).
- O teste nunca é usado para escolher limiares nem hiperparâmetros (proposta §4; spec "Imagem de teste não entra no treino").
- Imagens não são versionadas; os CSVs de divisão (`data/splits/`) são versionados.
- Fonte dos dados: Kaggle `nih-chest-xrays/sample` (5.606 imagens, ~1,1 GB, `sample_labels.csv`). O código também precisa aceitar o `Data_Entry_2017.csv` da base completa.
- Faixas etárias para análise por subgrupo: `<18`, `18-39`, `40-59`, `60+`.

## Review Focus

1. **Idade em formatos diferentes:** a amostra usa `"058Y"`, `"011M"`, `"005D"`, e a base completa usa inteiros, com outliers como 414. → anos inteiros; M/D → 0; fora de [0, 120] → nulo, sem derrubar o pipeline. (Task 2)
2. **Rótulos combinados e espaços extras:** `"Effusion|No Finding"` não ocorre na base, mas `" No Finding "` ou `"Hernia|Mass"` precisam virar 0 e 1, respectivamente. (Task 2)
3. **Imagem listada no CSV mas ausente no disco:** a linha é descartada e a contagem aparece no notebook, sem `FileNotFoundError` na Etapa 2. (Task 2)
4. **Paciente com muitas imagens:** todas as imagens dele ficam em um único split; a verificação falha com uma mensagem que nomeia o par de splits. (Task 3)
5. **Reprodutibilidade:** a mesma semente gera os mesmos splits; uma semente diferente gera splits diferentes. (Task 3)

---

### Task 1: Ambiente, configuração e pytest

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`, `src/config.py`, `tests/__init__.py`, `tests/test_config.py`

**Interfaces:**
- Produces (`src/config.py`):
  - `SEED: int = 42`
  - `TEST_SIZE: float = 0.20`, `VAL_SIZE: float = 0.10` (frações do total de pacientes)
  - `CLASSES: tuple[str, str] = ("NORMAL", "ALTERADA")` (índice = rótulo)
  - `NORMAL_TAG: str = "No Finding"`
  - `AGE_BINS: tuple[int, ...] = (0, 18, 40, 60, 121)`, `AGE_LABELS: tuple[str, ...] = ("<18", "18-39", "40-59", "60+")`
  - `ROOT: Path` (raiz do repo), `DATA_DIR = ROOT / "data"`, `SPLITS_DIR = DATA_DIR / "splits"`, `RESULTS_DIR = ROOT / "results"`
  - `KAGGLE_DATASET: str = "nih-chest-xrays/sample"`

- [ ] **Step 1: Atualizar `requirements.txt`**

Remover `torch`, `torchvision` e `grad-cam`. Adicionar `tensorflow>=2.15`, `kagglehub>=0.3` e `pytest>=8.0`. Manter o restante.

- [ ] **Step 2: Criar `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 3: Escrever o teste que falha**

```python
# tests/test_config.py
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
```

- [ ] **Step 4: Rodar e confirmar a falha**

Run: `pytest tests/test_config.py -v`
Expected: FAIL com `ImportError: cannot import name 'config'`

- [ ] **Step 5: Implementar `src/config.py`** com os valores do bloco Interfaces. `ROOT = Path(__file__).resolve().parents[1]`.

- [ ] **Step 6: Rodar e confirmar que passa**

Run: `pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pytest.ini src/config.py tests/
git commit -m "chore: configuração base e pytest"
```

---

### Task 2: Metadados e rótulo binário

**Files:**
- Create: `src/metadata.py`, `tests/test_metadata.py`

**Interfaces:**
- Consumes: `config.NORMAL_TAG`, `config.CLASSES`, `config.AGE_BINS`, `config.AGE_LABELS`
- Produces (`src/metadata.py`):
  - `binary_label(finding_labels: str) -> int`: 0 se `finding_labels.strip() == "No Finding"`, senão 1
  - `parse_age(value) -> int | None`: aceita int/float ou string `"NNNY"`, `"NNNM"` ou `"NNND"`; M/D → 0; fora de [0, 120] ou valor inválido → `None`
  - `age_group(age: int | None) -> str | None`: usa `AGE_BINS`/`AGE_LABELS`, com intervalos fechados à esquerda (`right=False`)
  - `load_metadata(csv_path: str | Path) -> pd.DataFrame`, com as colunas, nesta ordem: `image`, `finding_labels`, `patient_id` (int), `age` (`Int64`, anulável), `age_group`, `sex`, `view`, `label` (int 0/1), `class_name`
  - `attach_paths(df: pd.DataFrame, images_root: str | Path) -> pd.DataFrame`: indexa `images_root.rglob("*.png")` pelo nome do arquivo, adiciona a coluna `path` (str) e descarta as linhas sem arquivo; não altera o `df` de entrada

Mapeamento de colunas do CSV original (é igual na amostra e na base completa): `Image Index`→`image`, `Finding Labels`→`finding_labels`, `Patient ID`→`patient_id`, `Patient Age`→`age`, `Patient Gender`→`sex`, `View Position`→`view`. As demais colunas são descartadas.

- [ ] **Step 1: Escrever os testes que falham**

```python
# tests/test_metadata.py
import pandas as pd
import pytest
from src.metadata import binary_label, parse_age, age_group, load_metadata, attach_paths

@pytest.mark.parametrize("raw,expected", [
    ("No Finding", 0), (" No Finding ", 0),
    ("Effusion", 1), ("Hernia|Mass", 1), ("Infiltration|Effusion|Atelectasis", 1),
])
def test_binary_label(raw, expected):
    assert binary_label(raw) == expected

@pytest.mark.parametrize("raw,expected", [
    ("058Y", 58), ("011M", 0), ("005D", 0), (47, 47), (47.0, 47),
    (414, None), ("abc", None), (None, None), (-1, None),
])
def test_parse_age(raw, expected):
    assert parse_age(raw) == expected

@pytest.mark.parametrize("age,expected", [
    (0, "<18"), (17, "<18"), (18, "18-39"), (39, "18-39"), (40, "40-59"), (60, "60+"), (120, "60+"), (None, None),
])
def test_age_group(age, expected):
    assert age_group(age) == expected

@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({
        "Image Index": ["a.png", "b.png", "c.png"],
        "Finding Labels": ["No Finding", "Mass|Nodule", "Effusion"],
        "Follow-up #": [0, 1, 0],
        "Patient ID": [1, 1, 2],
        "Patient Age": ["030Y", "031Y", "414Y"],
        "Patient Gender": ["M", "M", "F"],
        "View Position": ["PA", "AP", "PA"],
    })
    path = tmp_path / "labels.csv"
    df.to_csv(path, index=False)
    return path

def test_load_metadata_columns_and_values(csv_file):
    df = load_metadata(csv_file)
    assert list(df.columns) == ["image", "finding_labels", "patient_id", "age", "age_group",
                                "sex", "view", "label", "class_name"]
    assert df["label"].tolist() == [0, 1, 1]
    assert df["class_name"].tolist() == ["NORMAL", "ALTERADA", "ALTERADA"]
    assert df["age"].tolist()[:2] == [30, 31] and pd.isna(df["age"].iloc[2])
    assert df["age_group"].tolist()[:2] == ["18-39", "18-39"]

def test_attach_paths_drops_missing_and_finds_nested(csv_file, tmp_path):
    nested = tmp_path / "images_001" / "images"
    nested.mkdir(parents=True)
    (nested / "a.png").write_bytes(b"")
    (tmp_path / "c.png").write_bytes(b"")
    df = load_metadata(csv_file)
    out = attach_paths(df, tmp_path)
    assert sorted(out["image"]) == ["a.png", "c.png"]
    assert out.set_index("image").loc["a.png", "path"].endswith("a.png")
    assert len(df) == 3 and "path" not in df.columns
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `pytest tests/test_metadata.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'src.metadata'`

- [ ] **Step 3: Implementar `src/metadata.py`** com as assinaturas do bloco Interfaces. A idade usa a regex `^(\d+)([YMD])$` sobre o valor sem espaços; `age` vira `pd.array(..., dtype="Int64")`.

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `pytest tests/test_metadata.py -v`
Expected: todos passam (24 casos)

- [ ] **Step 5: Commit**

```bash
git add src/metadata.py tests/test_metadata.py
git commit -m "feat: carga de metadados e rótulo binário"
```

---

### Task 3: Divisão por paciente

**Files:**
- Create: `src/split.py`, `tests/test_split.py`

**Interfaces:**
- Consumes: `config.SEED`, `config.TEST_SIZE`, `config.VAL_SIZE`; um DataFrame com as colunas `patient_id` e `label` (saída de `load_metadata`)
- Produces (`src/split.py`):
  - `SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")`
  - `patient_split(df, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SEED) -> dict[str, pd.DataFrame]`: chaves em `SPLIT_NAMES`; índices reiniciados; chama `assert_no_patient_overlap` antes de retornar
  - `assert_no_patient_overlap(splits: dict[str, pd.DataFrame]) -> None`: lança `ValueError` com a mensagem `"Pacientes em comum entre {a} e {b}: {n}"`
  - `save_splits(splits, out_dir: str | Path) -> None`: grava `{name}.csv` sem índice e cria o diretório se preciso
  - `load_splits(out_dir: str | Path) -> dict[str, pd.DataFrame]`: lê os três arquivos e roda `assert_no_patient_overlap`

Algoritmo (fixado pela proposta, §4): primeiro `GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)` separa o teste; depois, no restante, um segundo `GroupShuffleSplit` com `test_size=val_size / (1 - test_size)` (= 0,125) e a mesma semente separa a validação. Ambos usam `groups=df["patient_id"]`.

- [ ] **Step 1: Escrever os testes que falham**

```python
# tests/test_split.py
import numpy as np
import pandas as pd
import pytest
from src.split import SPLIT_NAMES, patient_split, assert_no_patient_overlap, save_splits, load_splits

@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    patients = np.repeat(np.arange(1000), rng.integers(1, 6, size=1000))  # 1–5 imagens por paciente
    return pd.DataFrame({"image": [f"{i}.png" for i in range(len(patients))],
                         "patient_id": patients,
                         "label": rng.integers(0, 2, size=len(patients))})

def test_keys_and_full_coverage(df):
    s = patient_split(df)
    assert tuple(s) == SPLIT_NAMES
    assert sum(len(p) for p in s.values()) == len(df)
    assert set(pd.concat(s.values())["image"]) == set(df["image"])

def test_no_patient_in_two_splits(df):
    s = patient_split(df)
    ids = [set(p["patient_id"]) for p in s.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])

def test_patient_proportions(df):
    s = patient_split(df)
    n = df["patient_id"].nunique()
    frac = {k: v["patient_id"].nunique() / n for k, v in s.items()}
    assert frac["test"] == pytest.approx(0.20, abs=0.01)
    assert frac["validation"] == pytest.approx(0.10, abs=0.01)
    assert frac["train"] == pytest.approx(0.70, abs=0.01)

def test_same_seed_same_split_and_different_seed_differs(df):
    a, b, c = patient_split(df, seed=42), patient_split(df, seed=42), patient_split(df, seed=7)
    assert a["test"]["image"].tolist() == b["test"]["image"].tolist()
    assert set(a["test"]["patient_id"]) != set(c["test"]["patient_id"])

def test_overlap_raises_with_split_names():
    s = {"train": pd.DataFrame({"patient_id": [1, 2]}),
         "validation": pd.DataFrame({"patient_id": [3]}),
         "test": pd.DataFrame({"patient_id": [2]})}
    with pytest.raises(ValueError, match="train e test: 1"):
        assert_no_patient_overlap(s)

def test_save_and_load_roundtrip(df, tmp_path):
    s = patient_split(df)
    save_splits(s, tmp_path / "splits")
    loaded = load_splits(tmp_path / "splits")
    for name in SPLIT_NAMES:
        pd.testing.assert_frame_equal(loaded[name], s[name])
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `pytest tests/test_split.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'src.split'`

- [ ] **Step 3: Implementar `src/split.py`** conforme o bloco Interfaces e o algoritmo acima.

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `pytest tests/test_split.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/split.py tests/test_split.py
git commit -m "feat: divisão por paciente com verificação de vazamento"
```

---

### Task 4: Download, notebook de EDA e splits versionados

**Files:**
- Create: `src/download.py`, `tests/test_download.py`, `notebooks/01_eda.ipynb`, `data/splits/{train,validation,test}.csv` (gerados), `results/figures/eda_*.png` (gerados)
- Modify: `.gitignore` (liberar `data/splits/`), `data/README.md` (seção "Obtenção via Kaggle"), `README.md` (como executar a Etapa 1)

**Interfaces:**
- Consumes: `config.KAGGLE_DATASET`, `config.SPLITS_DIR`, `config.RESULTS_DIR`; `load_metadata`, `attach_paths` (Task 2); `patient_split`, `save_splits` (Task 3)
- Produces:
  - `src/download.py`: `download_dataset(dataset: str = KAGGLE_DATASET) -> Path`, um wrapper fino de `kagglehub.dataset_download`, sem teste (depende de rede); `find_metadata_csv(root: str | Path) -> Path`, que procura via `rglob`, nesta ordem, `sample_labels.csv` e depois `Data_Entry_2017.csv`, e lança `FileNotFoundError` se não achar nenhum
  - `data/splits/{train,validation,test}.csv` com as colunas de `load_metadata`, **sem** a coluna `path` (os caminhos dependem da máquina; a Etapa 2 chama `attach_paths` de novo)

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_download.py
import pytest
from src.download import find_metadata_csv

def test_finds_sample_csv_nested(tmp_path):
    (tmp_path / "sample").mkdir()
    (tmp_path / "sample" / "sample_labels.csv").write_text("x")
    assert find_metadata_csv(tmp_path).name == "sample_labels.csv"

def test_finds_full_dataset_csv(tmp_path):
    (tmp_path / "Data_Entry_2017.csv").write_text("x")
    assert find_metadata_csv(tmp_path).name == "Data_Entry_2017.csv"

def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_metadata_csv(tmp_path)
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `pytest tests/test_download.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'src.download'`

- [ ] **Step 3: Implementar `src/download.py`** conforme o bloco Interfaces.

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `pytest tests/test_download.py -v`
Expected: 3 passed

- [ ] **Step 5: Ajustar `.gitignore`**

Logo após `!data/README.md`, acrescentar `!data/splits/`. Confirmar com `git check-ignore -v data/splits/train.csv`, que não deve retornar nada.

- [ ] **Step 6: Criar `notebooks/01_eda.ipynb`** com as células, nesta ordem:
  1. **Setup Colab:** se `"google.colab" in sys.modules`, rodar `git clone https://github.com/LuizCenci/triagem-raio-x` + `pip install -r requirements.txt` e fazer `chdir` para o repo; caso contrário, `chdir` para a raiz do repo. Depois, `sys.path.insert(0, ROOT)`.
  2. **Download:** `root = download_dataset()`; `csv = find_metadata_csv(root)`.
  3. **Carga:** `df = load_metadata(csv)`; `df_img = attach_paths(df, root)`; imprimir `len(df)`, `len(df_img)` e o número de imagens ausentes.
  4. **Distribuições** (uma figura cada, salvas em `results/figures/`): `eda_classes.png` (contagem NORMAL × ALTERADA), `eda_view.png` (PA/AP por classe), `eda_sex.png` (sexo por classe), `eda_age.png` (faixa etária por classe), `eda_images_per_patient.png` (histograma). Também imprimir o número de pacientes e de idades nulas.
  5. **Divisão:** `splits = patient_split(df_img.drop(columns="path"))`; mostrar a tabela split × classe (contagem e %) e o número de pacientes por split.
  6. **Gravação:** `save_splits(splits, SPLITS_DIR)`; imprimir os caminhos gravados.
  7. **Conclusões** (markdown): proporção entre as classes, decisão sobre a ponderação para a Etapa 2 (usar `class_weight` se a classe minoritária ficar abaixo de 40%) e as limitações observadas.

- [ ] **Step 7: Executar o notebook**

Run (local, com o dataset acessível): `jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb`. Também é possível rodar no Colab e baixar o `.ipynb` e os CSVs.
Expected: execução sem erros; `data/splits/` com 3 CSVs que somam ≈5.606 linhas (na amostra); 5 PNGs em `results/figures/`; a célula 5 sem `ValueError`.

- [ ] **Step 8: Verificar os splits gravados de forma independente**

Run: `python -c "from src.split import load_splits; from src.config import SPLITS_DIR; s = load_splits(SPLITS_DIR); print({k: (len(v), v.patient_id.nunique(), round(v.label.mean(), 3)) for k, v in s.items()})"`
Expected: 3 entradas; nenhum erro de sobreposição; proporção de ALTERADA (`label.mean()`) parecida entre os splits (diferença ≤ 0,05).

- [ ] **Step 9: Atualizar a documentação**

Em `data/README.md`, adicionar a seção "Obtenção via Kaggle (recomendado)": o dataset `nih-chest-xrays/sample`, como `download_dataset()` resolve o caminho e a observação de que `data/splits/` é versionado. Em `README.md`, na seção "Configuração do ambiente", acrescentar `pytest` e a execução do `01_eda.ipynb`.

- [ ] **Step 10: Rodar a suíte completa**

Run: `pytest -v`
Expected: todos os testes das Tasks 1 a 4 passam.

- [ ] **Step 11: Commit**

```bash
git add src/download.py tests/test_download.py notebooks/01_eda.ipynb data/splits/ results/figures/ .gitignore data/README.md README.md
git commit -m "feat: eda e splits por paciente"
```
