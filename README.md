# Triagem de Radiografias de Tórax

Trabalho de Visão Computacional — Bacharelado em Engenharia de Software, UTFPR Campus Dois Vizinhos.

Sistema que classifica radiografias frontais de tórax em **NORMAL** ou **ALTERADA** e estima a probabilidade de alteração, `p(alterada)`, para apoiar a **priorização da fila de análise**.

> **Aviso:** este é um protótipo acadêmico. O sistema não realiza diagnóstico e não substitui a avaliação de profissionais de saúde.

## Saídas do sistema

| Categoria | Significado |
|---|---|
| IMAGEM INADEQUADA | Qualidade insuficiente (rotação, corte, baixo contraste ou nitidez) |
| INDETERMINADO | Alta incerteza do modelo |
| PRIORIDADE ALTA | Alta probabilidade de alteração |
| PRIORIDADE MÉDIA | Probabilidade intermediária de alteração |
| ROTINA | Baixa probabilidade de alteração |

Os limiares são definidos no conjunto de validação, privilegiando o Recall da classe ALTERADA.

## Abordagem

- **Base:** NIH ChestX-ray14, com divisão por paciente (70/10/20).
- **Modelo:** ResNet-50 pré-treinada no ImageNet, com transfer learning e fine-tuning.
- **Comparação:** baseline simples e arquitetura leve (MobileNetV2 ou EfficientNet-B0).
- **Avaliação:** Recall, Precisão, F1-score, Balanced Accuracy, AUROC, PR-AUC e matriz de confusão.
- **Robustez:** conjunto de teste degradado com 150 imagens (ruído, rotação, corte, baixo contraste e combinações).
- **Explicabilidade:** Grad-CAM em casos de VP, FN, FP e VN.

## Estrutura do repositório

```
triagem-raio-x/
├── data/          # instruções para obter a base (imagens não versionadas)
├── notebooks/     # experimentos (01_eda.ipynb, 02_preprocessamento.ipynb, ...)
├── src/           # código reutilizável
├── results/       # métricas, matrizes de confusão e registros de experimentos
├── report/        # relatórios das entregas
├── README.md
└── requirements.txt
```

## Configuração do ambiente

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para obter e organizar os dados, siga as instruções em [data/README.md](data/README.md).

Testes e Etapa 1 (análise exploratória e divisão por paciente):

```bash
pytest
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

## Fluxo de trabalho

A branch `main` é protegida. Cada integrante trabalha em uma branch própria e abre um Pull Request, que só é integrado após aprovação de outro membro da equipe.

## Equipe

- Alana Luize Kunrath
- Lucas Henrique Bonissoni
- Luiz Augusto Cenci
- Matheus de Aquino da Silva
- Mauro Junior Cella Schuastz
- Murilo Ghedin Ghizzi
