"""Ajuste dinamico do threshold de decisao com base em custo financeiro.

Premissas de custo (ajustaveis conforme a realidade da planta):
- Falso Negativo (falha real nao detectada): parada nao planejada, danos ao
  equipamento e possiveis riscos operacionais -> custo alto.
- Falso Positivo (alarme de falha que nao se confirma): inspecao preventiva
  desnecessaria -> custo baixo, mas nao nulo.
- Verdadeiro Positivo / Verdadeiro Negativo: tratados como custo zero neste
  modelo simplificado (o custo de manutencao de rotina de um TP e
  considerado investimento esperado, nao uma perda extra).
"""

import numpy as np
import pandas as pd

from .metrics import confusion_counts

COST_FALSE_NEGATIVE = 50_000.0  # R$ - parada nao planejada / falha nao detectada
COST_FALSE_POSITIVE = 500.0  # R$ - inspecao preventiva desnecessaria
DEFAULT_THRESHOLD = 0.5


def total_cost_at_threshold(y_true, y_proba, threshold, cost_fn=COST_FALSE_NEGATIVE, cost_fp=COST_FALSE_POSITIVE):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    cost = counts["FN"] * cost_fn + counts["FP"] * cost_fp
    return cost, counts


def sweep_thresholds(y_true, y_proba, thresholds=None, cost_fn=COST_FALSE_NEGATIVE, cost_fp=COST_FALSE_POSITIVE):
    """Varre um intervalo de thresholds e calcula o custo total em cada um."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for t in thresholds:
        cost, counts = total_cost_at_threshold(y_true, y_proba, t, cost_fn, cost_fp)
        rows.append({"threshold": t, "custo_total": cost, **counts})

    return pd.DataFrame(rows)


def find_optimal_threshold(sweep_df):
    """Retorna a linha (threshold, custo, contagens) de menor custo total."""
    return sweep_df.loc[sweep_df["custo_total"].idxmin()]


def savings_vs_default(sweep_df, default_threshold=DEFAULT_THRESHOLD):
    """Calcula a economia (R$) do threshold otimo em relacao ao threshold padrao."""
    idx_default = (sweep_df["threshold"] - default_threshold).abs().idxmin()
    custo_default = sweep_df.loc[idx_default, "custo_total"]

    optimal_row = find_optimal_threshold(sweep_df)
    economia = custo_default - optimal_row["custo_total"]

    return {
        "custo_threshold_padrao": custo_default,
        "threshold_padrao": sweep_df.loc[idx_default, "threshold"],
        "custo_threshold_otimo": optimal_row["custo_total"],
        "threshold_otimo": optimal_row["threshold"],
        "economia_reais": economia,
    }
