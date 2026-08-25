"""Gera dados sinteticos de sensores industriais com forte desbalanceamento de classes."""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "vibracao_mm_s",
    "temperatura_c",
    "pressao_bar",
    "rotacao_rpm",
    "corrente_a",
]
TARGET_COLUMN = "falha"


def generate_sensor_data(n_samples=50_000, failure_rate=0.005, random_state=42):
    """Cria um dataset sintetico de telemetria de sensores industriais.

    Classe 0 (Operacional): ~ (1 - failure_rate) das amostras, com sensores em
    faixa nominal. Classe 1 (Falha): ~ failure_rate das amostras, com
    distribuicoes deslocadas (mais vibracao/temperatura/corrente, menor
    pressao) para simular o comportamento real de um equipamento antes de
    falhar, mantendo sobreposicao realista entre as classes.
    """
    rng = np.random.default_rng(random_state)

    n_falha = max(1, int(round(n_samples * failure_rate)))
    n_normal = n_samples - n_falha

    normal = pd.DataFrame(
        {
            "vibracao_mm_s": rng.normal(2.5, 0.7, n_normal),
            "temperatura_c": rng.normal(60, 6, n_normal),
            "pressao_bar": rng.normal(100, 10, n_normal),
            "rotacao_rpm": rng.normal(1500, 60, n_normal),
            "corrente_a": rng.normal(10, 1.8, n_normal),
        }
    )
    normal[TARGET_COLUMN] = 0

    # Distribuicoes deslocadas mas com sobreposicao realista em relacao a classe
    # Operacional (nem todo evento de falha e um outlier obvio), para que o
    # modelo cometa erros reais e a curva ROC/analise de custo sejam informativas.
    falha = pd.DataFrame(
        {
            "vibracao_mm_s": rng.normal(4.3, 1.3, n_falha),
            "temperatura_c": rng.normal(72, 9, n_falha),
            "pressao_bar": rng.normal(85, 14, n_falha),
            "rotacao_rpm": rng.normal(1420, 110, n_falha),
            "corrente_a": rng.normal(13, 2.5, n_falha),
        }
    )
    falha[TARGET_COLUMN] = 1

    df = pd.concat([normal, falha], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Garante que nenhum sensor fisicamente impossivel apareca (ex: pressao negativa)
    df["pressao_bar"] = df["pressao_bar"].clip(lower=0)
    df["rotacao_rpm"] = df["rotacao_rpm"].clip(lower=0)
    df["corrente_a"] = df["corrente_a"].clip(lower=0)

    return df


if __name__ == "__main__":
    dataset = generate_sensor_data()
    print(dataset[TARGET_COLUMN].value_counts(normalize=True))
    print(dataset.head())
