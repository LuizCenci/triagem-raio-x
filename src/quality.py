"""Indicadores simples de qualidade de imagem para a saída IMAGEM INADEQUADA."""
import cv2
import numpy as np
import pandas as pd


def quality_features(img: np.ndarray) -> dict[str, float]:
    """Contraste (desvio-padrão dos pixels) e nitidez (variância do Laplaciano)."""
    return {"contraste": float(img.std()),
            "nitidez": float(cv2.Laplacian(img, cv2.CV_64F).var())}


def fit_quality_limits(features: pd.DataFrame, quantile: float = 0.01) -> dict[str, float]:
    """Limite inferior de cada indicador = quantil `quantile` das imagens de referência."""
    return {col: float(features[col].quantile(quantile)) for col in ("contraste", "nitidez")}


def is_inadequate(features: dict, limits: dict) -> bool:
    return any(features[k] < limits[k] for k in limits)
