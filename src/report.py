"""Geracao do Relatorio Executivo em PDF a partir dos resultados do pipeline."""

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def _new_text_page(pdf, title_lines, paragraphs, author=None):
    """title_lines: string unica ou lista de linhas (para titulos longos)."""
    if isinstance(title_lines, str):
        title_lines = [title_lines]

    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    y_title = 0.94
    for line in title_lines:
        fig.text(0.08, y_title, line, fontsize=16, fontweight="bold")
        y_title -= 0.032

    if author:
        y_title -= 0.008
        fig.text(0.08, y_title, author, fontsize=10, color="dimgray")
        y_title -= 0.03

    y = min(0.85, y_title - 0.02)
    for block_title, block_text in paragraphs:
        fig.text(0.08, y, block_title, fontsize=12, fontweight="bold")
        y -= 0.035
        fig.text(0.08, y, block_text, fontsize=10, wrap=True, va="top",
                  bbox=dict(boxstyle="square,pad=0", facecolor="none", edgecolor="none"))
        # Estimativa simples de altura ocupada pelo texto (uma pagina A4, fonte 10)
        n_linhas = max(1, len(block_text) // 95 + block_text.count("\n") + 1)
        y -= 0.028 * n_linhas + 0.03

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def _page_arquitetura(pdf, author, dataset_info):
    paragrafo_contexto = (
        f"Dataset sintetico com {dataset_info['n_samples']:,} amostras, simulando sensores "
        f"industriais (vibracao, temperatura, pressao, rotacao, corrente). Prevalencia real de "
        f"falha: {dataset_info['failure_rate']*100:.2f}% das amostras — um cenario tipico de "
        f"desbalanceamento extremo de classes em manutencao preditiva."
    ).replace(",", ".")

    paragrafo_leakage = (
        "O conjunto de teste foi isolado com train_test_split (estratificado pela variavel "
        "alvo) ANTES de qualquer etapa de pre-processamento. O StandardScaler foi ajustado "
        "(fit) exclusivamente com estatisticas do conjunto de treino; o conjunto de teste "
        "recebeu apenas a transformacao (transform), sem participar do calculo de media/"
        "desvio-padrao. Essa ordem de operacoes impede Data Leakage: se o scaler fosse "
        "ajustado com o dataset completo, estatisticas do teste (que deveria representar dados "
        "nunca vistos) contaminariam o pre-processamento do treino, inflando artificialmente as "
        "metricas de validacao e escondendo problemas que apareceriam somente em produção."
    )

    paragrafo_modelo = (
        "Modelo: RandomForestClassifier com class_weight='balanced', que pondera o erro da "
        "classe minoritaria (falha) de forma inversamente proporcional a sua frequencia, "
        "evitando que o modelo simplesmente ignore a classe rara para maximizar acuracia."
    )

    _new_text_page(
        pdf,
        ["Relatorio Executivo", "Deteccao de Falhas em Sensores Industriais"],
        [
            ("1. Contexto e Dados", paragrafo_contexto),
            ("2. Arquitetura da Solucao e Prevencao de Data Leakage", paragrafo_leakage),
            ("3. Modelo", paragrafo_modelo),
        ],
        author=author,
    )


def _page_metricas(pdf, metrics_default, naive_accuracy):
    fig, (ax_table, ax_cm) = plt.subplots(1, 2, figsize=(11.69, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.suptitle("Quadro Comparativo de Metricas (threshold = 0.5)", fontsize=14, fontweight="bold")

    linhas = [
        ("Acuracia", f"{metrics_default['acuracia']:.4f}"),
        ("Acuracia (baseline ingenuo)", f"{naive_accuracy:.4f}"),
        ("Precisao", f"{metrics_default['precisao']:.4f}"),
        ("Recall", f"{metrics_default['recall']:.4f}"),
        ("F1-Score", f"{metrics_default['f1_score']:.4f}"),
        ("F2-Score (prioriza Recall)", f"{metrics_default['f2_score']:.4f}"),
        ("F0.5-Score (prioriza Precisao)", f"{metrics_default['f0_5_score']:.4f}"),
        ("AUC-ROC", f"{metrics_default['auc_roc']:.4f}"),
    ]
    ax_table.axis("off")
    table = ax_table.table(cellText=linhas, colLabels=["Metrica", "Valor"], loc="center",
                            cellLoc="left", colWidths=[0.7, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax_table.set_title(
        "Nota: acuracia alta (~99.5%) tambem e obtida por um modelo que\nnunca preve falha — por isso metricas sensiveis ao\ndesbalanceamento sao essenciais.",
        fontsize=8, loc="left", style="italic",
    )

    cm = np.array([[metrics_default["TN"], metrics_default["FP"]],
                   [metrics_default["FN"], metrics_default["TP"]]])
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1], labels=["Pred. Operacional", "Pred. Falha"])
    ax_cm.set_yticks([0, 1], labels=["Real Operacional", "Real Falha"])
    ax_cm.set_title("Matriz de Confusao")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center",
                       color="black", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def _page_roc(pdf, fpr, tpr, auc_value):
    fig, ax = plt.subplots(figsize=(8.27, 6))
    ax.plot(fpr, tpr, label=f"Modelo (AUC = {auc_value:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classificador aleatorio (AUC = 0.5)")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Recall / TPR)")
    ax.set_title("Curva AUC-ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_custo_threshold(pdf, sweep_df, savings_info, cost_fn, cost_fp):
    fig, (ax, ax_text) = plt.subplots(2, 1, figsize=(8.27, 9),
                                       gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(sweep_df["threshold"], sweep_df["custo_total"], linewidth=2, color="tab:red")

    ax.axvline(savings_info["threshold_otimo"], color="tab:green", linestyle="--",
               label=f"Threshold otimo = {savings_info['threshold_otimo']:.2f}")
    ax.axvline(savings_info["threshold_padrao"], color="tab:gray", linestyle=":",
               label=f"Threshold padrao = {savings_info['threshold_padrao']:.2f}")
    ax.scatter([savings_info["threshold_otimo"]], [savings_info["custo_threshold_otimo"]],
               color="tab:green", zorder=5)

    ax.set_xlabel("Threshold de Decisao")
    ax.set_ylabel("Custo Total Estimado (R$)")
    ax.set_title("Analise Financeira: Custo Total vs. Threshold de Decisao")
    ax.legend()
    ax.grid(alpha=0.3)

    texto = (
        f"Premissas de custo: FN (falha nao detectada) = R\\$ {cost_fn:,.2f} | "
        f"FP (alarme falso) = R\\$ {cost_fp:,.2f}\n"
        f"Custo no threshold padrao (0.5): R\\$ {savings_info['custo_threshold_padrao']:,.2f}\n"
        f"Custo no threshold otimo ({savings_info['threshold_otimo']:.2f}): "
        f"R\\$ {savings_info['custo_threshold_otimo']:,.2f}\n"
        f"Economia estimada: R\\$ {savings_info['economia_reais']:,.2f}"
    ).replace(",", "X").replace(".", ",").replace("X", ".")

    ax_text.axis("off")
    ax_text.text(0.0, 1.0, texto, fontsize=10, va="top", ha="left", transform=ax_text.transAxes)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_recomendacoes(pdf):
    texto_monitoramento = (
        "- Monitorar Data Drift: comparar periodicamente a distribuicao das features de entrada "
        "em producao (vibracao, temperatura, pressao, rotacao, corrente) contra a distribuicao "
        "de treino (ex.: teste de Kolmogorov-Smirnov ou PSI). Alertar se houver desvio "
        "significativo.\n"
        "- Monitorar Concept Drift: acompanhar Precisao/Recall/F2-Score em uma janela deslizante "
        "de predicoes confirmadas (feedback de manutencao real), nao apenas Acuracia.\n"
        "- Retreinamento programado: retreinar o modelo periodicamente (ex.: mensal ou por "
        "gatilho de drift) incorporando falhas reais confirmadas, mantendo o mesmo pipeline de "
        "isolamento treino/teste para evitar leakage em cada novo ciclo.\n"
        "- Revisao periodica do threshold: revalidar o threshold otimo sempre que os custos de "
        "FN/FP mudarem (ex.: nova politica de manutencao, mudanca no custo de parada), pois o "
        "ponto de menor custo depende diretamente dessas premissas financeiras.\n"
        "- Alertas de qualidade de dados: validar ranges fisicos dos sensores (ex.: pressao >= 0) "
        "antes da inferencia, para nao alimentar o modelo com leituras invalidas de hardware."
    )

    _new_text_page(
        pdf,
        ["Recomendacoes de Monitoramento MLOps", "em Producao"],
        [("Plano de Monitoramento", texto_monitoramento)],
    )


def build_report(output_path, author, dataset_info, metrics_default, naive_accuracy,
                  fpr, tpr, sweep_df, savings_info, cost_fn, cost_fp):
    with PdfPages(output_path) as pdf:
        _page_arquitetura(pdf, author, dataset_info)
        _page_metricas(pdf, metrics_default, naive_accuracy)
        _page_roc(pdf, fpr, tpr, metrics_default["auc_roc"])
        _page_custo_threshold(pdf, sweep_df, savings_info, cost_fn, cost_fp)
        _page_recomendacoes(pdf)
