# Etapa 6 — Artigo curto e apresentação: Plano

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produzir o artigo em PDF (modelo SBC, 4 a 6 páginas) e os slides para 8 minutos, usando apenas resultados gerados nas Etapas 1 a 5.

**Architecture:** Fonte LaTeX em `report/artigo/` (modelo SBC, editável no Overleaf); figuras copiadas de `results/figures/`; tabelas montadas a partir de `results/metrics/*.csv`. Slides em `report/apresentacao/` (formato livre; exportar em PDF).

**Spec:** `docs/spec.md` §3, §4 e §6.

## Global Constraints

- Prazo: **26/11, até as 19h** (PDF); apresentação em aula em 26/11.
- Artigo: modelo SBC, PDF, **4 a 6 páginas incluindo figuras e referências**, em português.
- Seções: Introdução, Dados, Método, Resultados e discussão, Conclusão, Referências.
- Obrigatório no texto: dataset (nome, link, licença, nº de imagens, classes, recorte usado); imagens próprias e a opção adotada (B: conjunto degradado, como alternativa combinada com o professor para imagens médicas); métrica adequada + exemplos de acertos e erros; link do repositório e das imagens; **uma frase declarando o uso de IA generativa** (ferramenta e finalidade).
- Nenhum número no texto sem o arquivo de origem em `results/` (integridade, spec §4).
- Apresentação: 8 min + 2 min de perguntas; **todos os 6 integrantes falam**.
- O sistema é sempre apresentado como apoio à triagem, nunca como diagnóstico.

## Review Focus

1. **Excesso de páginas:** priorizar 4 figuras (curvas ROC/PR, matriz de confusão, queda no degradado, grade Grad-CAM) e 2 tabelas (métricas gerais + degradado por tipo); subgrupos em uma frase ou em uma tabela pequena.
2. **Licença do NIH não confirmada** (proposta §3): verificar na fonte oficial e citar o termo exato, ou declarar explicitamente que não há uma licença formal.
3. **Discrepância entre texto e resultados:** a última passada confere cada número contra os CSVs.

---

### Task 1: Esqueleto e insumos

- [ ] **Step 1:** Baixar o modelo SBC (LaTeX) para `report/artigo/` (`main.tex`, `sbc-template.sty`, `sbc.bst`) e criar `report/artigo/figuras/` e `refs.bib`.
- [ ] **Step 2:** Copiar para `report/artigo/figuras/`: `roc_pr.png`, `confusao_resnet50.png`, `degradado_queda.png`, `degradado_exemplos.png`, `gradcam_teste.png` e `acertos_erros.png`.
- [ ] **Step 3:** Montar as tabelas LaTeX a partir de `test_overall.csv`, `degraded_by_type.csv` e `test_priority.csv` (pode ser com `df.to_latex(float_format="%.3f")` em uma célula final do `03_avaliacao.ipynb`).
- [ ] **Step 4:** `refs.bib` com, no mínimo: Wang et al. 2017 (ChestX-ray8/14), He et al. 2016 (ResNet), Selvaraju et al. 2017 (Grad-CAM) e Deng et al. 2009 (ImageNet).
- [ ] **Step 5:** `git commit -m "docs: esqueleto do artigo"`

### Task 2: Redação (dividida entre os integrantes)

Fonte de cada seção:

| Seção | Fonte | ~Tamanho |
|---|---|---|
| Introdução | `CLAUDE.md` §1–§2 | 0,5 p |
| Dados | `CLAUDE.md` §3–§5; tabelas do `01_eda.ipynb`; `degraded_metadata.csv` | 1 p |
| Método | `CLAUDE.md` §6–§8; `02_treinamento.ipynb` (hiperparâmetros); `thresholds.json` | 1 p |
| Resultados e discussão | `results/metrics/*`; figuras; `gradcam_casos.csv`; limitações (`CLAUDE.md` §10) | 1,5–2 p |
| Conclusão | aprendizados + trabalhos futuros (dados externos, calibração, detecção de rotação e corte) | 0,3 p |

- [ ] **Step 1:** Escrever cada seção; ao fim de "Método", incluir a frase de declaração de IA generativa.
- [ ] **Step 2:** Compilar e conferir 4 ≤ páginas ≤ 6.
- [ ] **Step 3:** Revisão cruzada: cada número do texto conferido contra o CSV de origem; checar a lista de verificação da spec §6.
- [ ] **Step 4:** Gravar `report/artigo/artigo.pdf`; `git commit -m "docs: artigo final"`.

### Task 3: Apresentação

Roteiro (spec §3.2), com um bloco por integrante:

| Bloco | Tempo | Conteúdo |
|---|---|---|
| 1 | 1:00 | Problema e por que triagem, e não diagnóstico |
| 2 | 1:30 | NIH ChestX-ray14, recorte, divisão por paciente |
| 3 | 1:30 | Conjunto degradado (mostrar as imagens) |
| 4 | 1:30 | Modelo e camada de decisão |
| 5 | 1:30 | Resultados: métricas, prioridade, queda no degradado |
| 6 | 1:00 | Grad-CAM, acertos e erros, aprendizados |

- [ ] **Step 1:** Montar os slides (reaproveitando as figuras do artigo) em `report/apresentacao/`; exportar `apresentacao.pdf`.
- [ ] **Step 2:** Ensaio cronometrado: total ≤ 8:00.
- [ ] **Step 3:** Lista de perguntas prováveis (por que não usar acurácia? como evitar vazamento? o que o Grad-CAM mostrou nos FN? por que não há imagens próprias reais?), com a resposta de cada integrante.
- [ ] **Step 4:** `git commit -m "docs: slides da apresentação"`
