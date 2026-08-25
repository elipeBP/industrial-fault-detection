"""Pipeline completo: geracao de dados -> split sem leakage -> treino -> metricas ->
analise financeira de threshold -> geracao do Relatorio Executivo em PDF.

Uso:
    python main.py
"""

import os

from src.data_generation import FEATURE_COLUMNS, TARGET_COLUMN, generate_sensor_data
from src.metrics import accuracy_trap_baseline, build_metrics_table, compute_roc_curve
from src.model import train_model
from src.preprocessing import fit_scaler, split_train_test, transform_features
from src.report import build_report
from src.threshold_analysis import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    savings_vs_default,
    sweep_thresholds,
)

AUTHOR = "Felipe Padilha — UniSENAI — Engenharia de Dados e MLOps"
N_SAMPLES = 50_000
FAILURE_RATE = 0.005
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "relatorio_executivo.pdf")


def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Dados sinteticos desbalanceados
    df = generate_sensor_data(n_samples=N_SAMPLES, failure_rate=FAILURE_RATE, random_state=RANDOM_STATE)
    print(f"Dataset gerado: {len(df):,} amostras | prevalencia de falha: "
          f"{df[TARGET_COLUMN].mean()*100:.3f}%")

    # 2. Isolamento treino/teste ANTES de qualquer pre-processamento
    X_train, X_test, y_train, y_test = split_train_test(df, TARGET_COLUMN, test_size=0.2,
                                                          random_state=RANDOM_STATE)

    # 3. Scaler ajustado somente no treino (previne data leakage)
    scaler = fit_scaler(X_train)
    X_train_scaled = transform_features(scaler, X_train)
    X_test_scaled = transform_features(scaler, X_test)

    # 4. Treinamento
    model = train_model(X_train_scaled, y_train, random_state=RANDOM_STATE)

    # 5. Metricas no conjunto de teste (nunca visto pelo modelo ou pelo scaler)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics_default = build_metrics_table(y_test, y_proba, threshold=0.5)
    naive_accuracy = accuracy_trap_baseline(y_test)
    fpr, tpr, _ = compute_roc_curve(y_test, y_proba)

    print("\n--- Metricas (threshold = 0.5) ---")
    for k, v in metrics_default.items():
        print(f"{k}: {v}")
    print(f"Acuracia do modelo ingenuo (sempre 'Operacional'): {naive_accuracy:.4f}")

    # 6. Ajuste dinamico do threshold com base em custo financeiro
    sweep_df = sweep_thresholds(y_test, y_proba)
    savings_info = savings_vs_default(sweep_df, default_threshold=0.5)

    print("\n--- Analise Financeira de Threshold ---")
    for k, v in savings_info.items():
        print(f"{k}: {v}")

    # 7. Relatorio Executivo em PDF
    dataset_info = {"n_samples": N_SAMPLES, "failure_rate": FAILURE_RATE}
    build_report(
        output_path=OUTPUT_PDF,
        author=AUTHOR,
        dataset_info=dataset_info,
        metrics_default=metrics_default,
        naive_accuracy=naive_accuracy,
        fpr=fpr,
        tpr=tpr,
        sweep_df=sweep_df,
        savings_info=savings_info,
        cost_fn=COST_FALSE_NEGATIVE,
        cost_fp=COST_FALSE_POSITIVE,
    )
    print(f"\nRelatorio Executivo gerado em: {OUTPUT_PDF}")


if __name__ == "__main__":
    run_pipeline()
