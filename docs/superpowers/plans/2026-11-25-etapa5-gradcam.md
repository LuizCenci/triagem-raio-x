# Etapa 5 — Grad-CAM e exemplos de acertos e erros: Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gerar os mapas Grad-CAM de 12 casos do teste (3 VP, 3 FN, 3 FP, 3 VN) e de 3 casos degradados, além das figuras de acertos e erros pedidas pela spec, com uma análise qualitativa de cada caso.

**Architecture:** Um único notebook `05_gradcam.ipynb`; o Grad-CAM é implementado nele com `tf.GradientTape` (escopo reduzido por prazo, sem módulo nem testes).

**Tech Stack:** TensorFlow/Keras, OpenCV, matplotlib.

**Spec:** `CLAUDE.md` §9; `docs/spec.md` §2.3 ("exemplos visuais de acertos e de erros").

## Global Constraints

- Mínimo exigido: 5 casos; meta da equipe: 12 do teste + 3 do conjunto degradado.
- Classificação do caso (VP, FN, FP, VN) em `th.t_rotina` (de `results/thresholds.json`).
- Seleção determinística: VP = maiores `p` entre as ALTERADAS acertadas; FN = menores `p` entre as ALTERADAS; FP = maiores `p` entre as NORMAIS; VN = menores `p` entre as NORMAIS. Os FN recebem atenção especial (proposta §9).
- Camada alvo: `conv5_block3_out`, dentro da base aninhada `resnet50`.
- O Grad-CAM é evidência qualitativa, não prova de raciocínio clínico (proposta §9).

## Review Focus

1. **Modelo aninhado:** `model.get_layer("conv5_block3_out")` não existe no nível de cima. É preciso montar o `grad_model` a partir de `model.get_layer("resnet50")` e aplicar `gap` e `head` manualmente.
2. **Mapa vazio** (gradientes todos ≤ 0 após o ReLU): normalizar com `max(heatmap.max(), 1e-8)` para não dividir por zero.
3. **Pré-processamento:** a imagem passa por `resnet50.preprocess_input` antes do `grad_model`, como no treino; a sobreposição é desenhada sobre a imagem original.

---

### Task 1: Notebook `05_gradcam.ipynb`

**Files:**
- Create: `notebooks/05_gradcam.ipynb`, `results/figures/{gradcam_teste,gradcam_degradado,acertos_erros}.png`, `results/gradcam_casos.csv`

**Interfaces:**
- Consumes: `models/resnet50.keras` com as camadas `resnet50`, `gap` e `head` (Etapa 2); `load_image` (Etapa 2); `results/predictions/resnet50_test.csv` (Etapa 2); `results/thresholds.json` (Etapa 3); `results/degraded_metadata.csv` + `data/degraded/` (Etapa 4)
- Produces: `results/gradcam_casos.csv` com as colunas `image, conjunto (teste|degradado), caso (VP|FN|FP|VN), label, p_alterada, foco (pulmao|fora|misto), observacao`

Células:
- [ ] **Step 1:** Setup (o mesmo da Etapa 2) e carga do modelo, das predições e dos limiares.
- [ ] **Step 2:** Função `grad_cam(model, img_224x224x3) -> np.ndarray (224, 224) em [0, 1]`:

```python
base = model.get_layer("resnet50")
grad_model = keras.Model(base.inputs, [base.get_layer("conv5_block3_out").output, base.output])
with tf.GradientTape() as tape:
    conv, feats = grad_model(preprocess_input(img[None]))
    tape.watch(conv)
    pred = model.get_layer("head")(model.get_layer("gap")(feats))[:, 0]
grads = tape.gradient(pred, conv)
weights = tf.reduce_mean(grads, axis=(1, 2))
cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights[0], axis=-1))
# normalizar com max(cam.max(), 1e-8) e redimensionar para 224×224 com cv2.resize
```

  Observação: como `base.output` é o próprio `conv5_block3_out`, o caminho até `pred` passa por `conv`. Conferir que `grads` não é `None`.
- [ ] **Step 3:** Sanidade: para um caso, conferir que `pred` calculado na função ≈ `p_alterada` do CSV (diferença < 1e-4).
- [ ] **Step 4:** Selecionar os 12 casos do teste pela regra das Global Constraints; montar a grade 4×3 (linhas = VP, FN, FP, VN), com a imagem original + sobreposição `cv2.COLORMAP_JET` (alpha 0,4) e um título com `p` e o rótulo → `gradcam_teste.png`.
- [ ] **Step 5:** 3 casos degradados (ALTERADAS com a maior queda de `p` entre a original e a degradada): pares original × degradada com o Grad-CAM → `gradcam_degradado.png`.
- [ ] **Step 6:** `acertos_erros.png`: uma grade sem Grad-CAM com 4 acertos e 4 erros do teste e 2 acertos e 2 erros do conjunto degradado, com `p`, rótulo e saída de `decide`.
- [ ] **Step 7:** Preencher `gradcam_casos.csv` com a inspeção visual de cada caso (`foco` e `observacao`: bordas, textos, marcadores, dispositivos?) e escrever um markdown resumindo os padrões, com ênfase nos FN.
- [ ] **Step 8:** `git add notebooks/05_gradcam.ipynb results/ && git commit -m "feat: grad-cam e exemplos visuais"`
