"""Isolamento treino/teste e escalonamento sem vazamento de dados (data leakage).

Regra de ouro aplicada aqui: o conjunto de teste e separado ANTES de qualquer
ajuste de pre-processamento. O StandardScaler e ajustado (fit) somente com
estatisticas do conjunto de treino; o conjunto de teste apenas recebe a
transformacao (transform), nunca participa do fit. Isso evita que informacao
do teste "contamine" o pipeline e infle metricas de forma irrealista.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_train_test(df, target_column, test_size=0.2, random_state=42):
    """Isola o teste antes de qualquer etapa de pre-processamento.

    Usa estratificacao pela variavel alvo para preservar a proporcao de
    99.5%/0.5% tanto no treino quanto no teste.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def fit_scaler(X_train):
    """Ajusta o StandardScaler usando exclusivamente dados de treino."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_features(scaler, X):
    """Aplica um scaler ja ajustado (nunca re-ajusta) a qualquer conjunto."""
    return scaler.transform(X)
