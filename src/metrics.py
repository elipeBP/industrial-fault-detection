"""Calculo e comparacao das metricas de classificacao para cenarios desbalanceados."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def confusion_counts(y_true, y_pred):
    """Retorna TN, FP, FN, TP na ordem convencional do sklearn (labels 0/1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def build_metrics_table(y_true, y_proba, threshold=0.5):
    """Monta o quadro comparativo de metricas para um dado threshold.

    Acuracia e incluida de proposito para evidenciar a "armadilha perigosa":
    em um dataset com 99.5% de classe Operacional, um modelo que preve
    sempre "Operacional" atinge ~99.5% de acuracia sem detectar nenhuma
    falha real — por isso ela e reportada ao lado de metricas sensiveis ao
    desbalanceamento (Precisao, Recall, F-beta, AUC-ROC).
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)

    return {
        "threshold": threshold,
        **counts,
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": fbeta_score(y_true, y_pred, beta=1.0, zero_division=0),
        "f2_score": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "f0_5_score": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_proba),
    }


def compute_roc_curve(y_true, y_proba):
    """Retorna (fpr, tpr, thresholds) para plotar a curva AUC-ROC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr, thresholds


def accuracy_trap_baseline(y_true):
    """Acuracia de um modelo 'ingenuo' que sempre preve a classe majoritaria.

    Serve como evidencia numerica direta da armadilha da acuracia em dados
    desbalanceados: mostra o piso de acuracia obtido sem nenhuma capacidade
    preditiva real.
    """
    y_true = np.asarray(y_true)
    naive_pred = np.zeros_like(y_true)
    return accuracy_score(y_true, naive_pred)
