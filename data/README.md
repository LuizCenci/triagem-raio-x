# Dados

As imagens não são versionadas no repositório devido ao tamanho da base (~42 GB).

## Base utilizada

**NIH ChestX-ray14** — 112.120 radiografias de tórax de 30.805 pacientes, disponibilizadas pelo National Institutes of Health.

- Fonte oficial: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Arquivo de metadados: `Data_Entry_2017.csv`

## Como obter

1. Baixe os arquivos `images_*.tar.gz` e o `Data_Entry_2017.csv` da fonte oficial.
2. Extraia as imagens em `data/raw/images/`.
3. Coloque o CSV de metadados em `data/raw/`.

## Estrutura esperada

```
data/
├── raw/          # imagens e metadados originais
├── processed/    # imagens redimensionadas e normalizadas
├── train/        # ~70% dos pacientes
├── validation/   # ~10% dos pacientes
├── test/         # ~20% dos pacientes (isolado até a avaliação final)
└── degraded/     # 150 imagens degradadas do conjunto de teste + degraded_metadata.csv
```

## Rótulos

- **NORMAL**: imagens com o rótulo `No Finding`.
- **ALTERADA**: imagens com ao menos um dos demais achados.

A divisão é feita **por paciente** (`Patient ID`), com semente fixa, garantindo ausência de interseção de pacientes entre os conjuntos.
