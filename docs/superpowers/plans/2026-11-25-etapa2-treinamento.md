# Etapa 2 — Treinamento: Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treinar um baseline simples e a ResNet-50 (transfer learning em 2 fases) e gravar as predições `p(alterada)` de validação e teste para as Etapas 3 a 5.

**Architecture:** Um pipeline `tf.data` pequeno em `src/pipeline.py` (reaproveitado nas Etapas 4 e 5) e um notebook `02_treinamento.ipynb` para rodar no Colab com GPU. Todo o treino fica no notebook, sem testes unitários (escopo reduzido por prazo). O pré-processamento de cada modelo é feito no `tf.data`, fora do modelo, para que o `.keras` salvo não dependa de camadas `Lambda`.

**Tech Stack:** TensorFlow/Keras 2.15+, pandas, scikit-learn (`compute_class_weight`), matplotlib.

**Spec:** `docs/spec.md` §2.3 + `CLAUDE.md` §4 e §6.

## Global Constraints

- Entrada 224×224; PNG lido em escala de cinza e replicado em 3 canais.
- `SEED = 42` (`keras.utils.set_random_seed`).
- Hiperparâmetros e parada antecipada usam **só a validação**. O teste é apenas predito e gravado; nenhuma métrica de teste é olhada nesta etapa.
- Sem flip horizontal na augmentation (a anatomia do tórax não é simétrica).
- Modelos salvos em `models/` (no `.gitignore`); no Colab, copiar para o Google Drive.
- Nomes dos modelos usados em arquivos: `baseline`, `resnet50`.

## Review Focus

1. **BatchNorm durante o fine-tuning:** a base é chamada com `training=False`, senão as estatísticas do ImageNet são destruídas com lotes pequenos.
2. **Desbalanceamento:** `class_weight="balanced"` calculado **no treino**, nunca no conjunto inteiro.
3. **Sessão do Colab caindo:** cada fase salva um checkpoint (`ModelCheckpoint` com `monitor="val_auc"`), e o notebook retoma a partir do arquivo se ele existir.
4. **Ordem das predições:** o dataset de predição não pode ter `shuffle`; a coluna `p_alterada` precisa casar linha a linha com o CSV do split.

---

### Task 1: `src/pipeline.py`

**Files:**
- Create: `src/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `config.SEED`
- Produces:
  - `IMG_SIZE: int = 224`
  - `load_image(path: str) -> tf.Tensor`: lê o PNG com `tf.io.decode_png(channels=1)`, redimensiona para 224×224 e devolve `float32` (224, 224, 3) com valores em [0, 255]
  - `make_dataset(df: pd.DataFrame, preprocess: Callable, batch_size: int = 32, shuffle: bool = False, augment: keras.Sequential | None = None) -> tf.data.Dataset`: produz `(imagem, label)`; o embaralhamento usa `SEED`; a ordem é `map(load_image) → augment → preprocess → batch → prefetch`
  - `AUGMENT: keras.Sequential` com `RandomRotation(0.03)`, `RandomZoom(0.1)` e `RandomContrast(0.1)`, todas com `seed=SEED`

- [ ] **Step 1: Teste de fumaça** — `test_load_image_shape_and_range`: grava um PNG 50×40 em `tmp_path` com `tf.io.encode_png` e verifica que `load_image` devolve shape `(224, 224, 3)` e máximo ≤ 255. Também `test_make_dataset_preserves_order`: com 3 PNGs e `shuffle=False`, os labels do primeiro lote são iguais a `df.label.tolist()`.
- [ ] **Step 2:** `pytest tests/test_pipeline.py -v` → FAIL (módulo inexistente)
- [ ] **Step 3:** Implementar conforme o bloco Interfaces.
- [ ] **Step 4:** `pytest tests/test_pipeline.py -v` → 2 passed
- [ ] **Step 5:** `git commit -m "feat: pipeline tf.data"`

---

### Task 2: Notebook `02_treinamento.ipynb`

**Files:**
- Create: `notebooks/02_treinamento.ipynb`, `results/predictions/{baseline,resnet50}_{validation,test}.csv`, `results/training/history_{baseline,resnet50_fase1,resnet50_fase2}.csv`, `results/figures/curvas_treino.png`, `results/experiments.csv`

**Interfaces:**
- Consumes: `load_splits(SPLITS_DIR)`, `attach_paths` (Etapa 1); `make_dataset`, `AUGMENT` (Task 1)
- Produces:
  - `models/baseline.keras` e `models/resnet50.keras`; o modelo ResNet tem as camadas nomeadas `resnet50` (base aninhada), `gap`, `dropout` e `head` (usadas pelo Grad-CAM na Etapa 5)
  - `results/predictions/{modelo}_{split}.csv` com as colunas `image, patient_id, label, sex, age_group, view, p_alterada`
  - `results/experiments.csv`: uma linha por modelo, com `data, modelo, config (json), seed, val_auc, val_recall@0.5`

Células, nesta ordem:

- [ ] **Step 1: Setup** — o mesmo da célula 1 do `01_eda.ipynb`, mais a montagem do Drive (`/content/drive/MyDrive/triagem-raio-x/models`), `download_dataset()`, `attach_paths` em cada split e `set_random_seed(SEED)`.
- [ ] **Step 2: Pesos de classe** — `compute_class_weight("balanced", classes=[0, 1], y=train.label)` → dict; imprimir o resultado.
- [ ] **Step 3: Baseline** — uma CNN treinada do zero: `Rescaling(1/255)` → 3 × [`Conv2D(32·2^i, 3, relu)` → `MaxPool`] → `GlobalAveragePooling2D` → `Dense(1, sigmoid)`. Preprocess no dataset: identidade (o `Rescaling` fica dentro do modelo). `Adam(1e-3)`, perda `binary_crossentropy`, métricas `AUC(name="auc")`, `Recall(name="recall")` e `Precision(name="precision")`; até 15 épocas; `EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True)`.
- [ ] **Step 4: ResNet-50, fase 1 (base congelada)** — `inputs = Input((224, 224, 3))`; `base = ResNet50(include_top=False, weights="imagenet")` com `base.trainable = False`; `x = base(inputs, training=False)` → `GlobalAveragePooling2D(name="gap")` → `Dropout(0.3, name="dropout")` → `Dense(1, sigmoid, name="head")`. Preprocess no dataset: `keras.applications.resnet50.preprocess_input`. `Adam(1e-3)`, até 10 épocas, as mesmas métricas e o mesmo early stopping, mais `ModelCheckpoint`.
- [ ] **Step 5: ResNet-50, fase 2 (fine-tuning)** — descongelar as camadas da base a partir de `conv5_block1_1_conv` (inclusive), mantendo todas as `BatchNormalization` congeladas; recompilar com `Adam(1e-5)`; até 10 épocas, com o mesmo early stopping. Salvar `models/resnet50.keras`.
- [ ] **Step 6: Curvas** — `val_auc` e `loss` por época, para os 3 históricos, em `results/figures/curvas_treino.png`; gravar os históricos em CSV.
- [ ] **Step 7: Predições** — para cada modelo e para `validation` e `test`: `model.predict(make_dataset(df, preprocess, shuffle=False))` e gravar o CSV no formato do bloco Interfaces. Conferir que o número de linhas é igual ao do split.
- [ ] **Step 8: Registro** — acrescentar as linhas em `results/experiments.csv` (criar o arquivo com cabeçalho se não existir).
- [ ] **Step 9: Verificação**

Run: `python -c "import pandas as pd; [print(f, len(d:=pd.read_csv(f'results/predictions/{f}.csv')), d.p_alterada.between(0,1).all()) for f in ['baseline_validation','baseline_test','resnet50_validation','resnet50_test']]"`
Expected: 4 linhas com contagens iguais às dos splits e `True`.

- [ ] **Step 10: Commit** (o `.ipynb` com as saídas e `results/`)

```bash
git add notebooks/02_treinamento.ipynb results/
git commit -m "feat: treino do baseline e da resnet50"
```
