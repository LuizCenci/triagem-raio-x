"""Métricas de avaliação binária, busca de limiares e análise por subgrupo."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, p, threshold: float) -> dict[str, float]:
    """Calcula métricas binárias para positivo = ALTERADA (1) com corte p >= threshold."""
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(p, dtype=float)
    y_pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, y_pred, labels=[0, 1]).ravel())
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    has_two_classes = len(np.unique(y)) >= 2
    auroc = float(roc_auc_score(y, prob)) if has_two_classes else float("nan")
    pr_auc = float(average_precision_score(y, prob)) if has_two_classes else float("nan")

    return {
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "specificity": specificity,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": int(len(y)),
    }


def threshold_for_recall(y_true, p, target: float) -> float:
    """Maior limiar entre os valores únicos de p que atinge recall >= target."""
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(p, dtype=float)
    for t in np.sort(np.unique(prob))[::-1]:
        if recall_score(y, (prob >= t).astype(int), zero_division=0) >= target:
            return float(t)
    return float(np.min(prob))


def threshold_for_precision(y_true, p, target: float) -> float | None:
    """Menor limiar entre os valores únicos de p com precisão >= target e >= 1 positivo previsto."""
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(p, dtype=float)
    for t in np.sort(np.unique(prob)):
        y_pred = (prob >= t).astype(int)
        if y_pred.sum() > 0 and precision_score(y, y_pred, zero_division=0) >= target:
            return float(t)
    return None


def subgroup_metrics(df: pd.DataFrame, group_col: str, threshold: float) -> pd.DataFrame:
    """Calcula binary_metrics por subgrupo de `group_col`, ignorando valores nulos."""
    rows = []
    valid = df.dropna(subset=[group_col])
    for group, sub in valid.groupby(group_col, sort=True):
        metrics = binary_metrics(sub["label"], sub["p_alterada"], threshold)
        rows.append({"group": group, **metrics})
    return pd.DataFrame(rows)
