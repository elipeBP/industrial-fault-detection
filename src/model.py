"""Treinamento do modelo de classificacao de falhas."""

from sklearn.ensemble import RandomForestClassifier


def train_model(X_train_scaled, y_train, random_state=42):
    """Treina um RandomForest com balanceamento de classe interno.

    class_weight='balanced' pondera o erro da classe minoritaria (falha)
    proporcionalmente a sua raridade, o que ajuda o modelo a nao ignorar a
    classe de interesse mesmo com 0.5% de prevalencia.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    return model
