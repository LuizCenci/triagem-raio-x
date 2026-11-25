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
