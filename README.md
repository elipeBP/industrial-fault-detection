# Falhas em Sensores Industriais — MLOps

Dois trabalhos práticos da disciplina de Engenharia de Dados e MLOps (UniSENAI), ambos girando em torno de detecção de falha em equipamentos industriais a partir de dados de sensores — um focado no pipeline de treino/avaliação, o outro na predição em tempo real.

## Atividade 1 — Simulador de predição em tempo real (compressor)

Pasta: [`atividade1-compressor/`](atividade1-compressor/)

Dashboard em Streamlit que simula telemetria IoT de um compressor industrial (pressão, temperatura, vazão, corrente, horímetros) chegando com atrasos aleatórios, e roteia cada leitura dinamicamente para um de dois modelos `RandomForestClassifier` já treinados:

- **Modelo "motor rodando"** — usa os sensores brutos + médias/desvios móveis de 7 dias.
- **Modelo "motor parado"** — usa apenas sensores e horímetros brutos.

O roteamento é decidido pelo campo `status_operacao` do pacote de telemetria: `prever_ponto_dinamicamente()` compara o status contra a lista de status "em operação" e escolhe o modelo correspondente antes de gerar a predição.

```bash
cd atividade1-compressor
pip install streamlit pandas joblib scikit-learn
streamlit run Teste_Compressor.py
```

`treinarmodelo.py` é o script que gerou os dois arquivos `.joblib` incluídos na pasta. `respostas_teoricas.md` documenta as decisões de arquitetura (por que chaveamento dinâmico de modelo, quais features cada um espera).

## Atividade 2 — Avaliação de métricas em classificação desbalanceada

Pipeline reprodutível que gera dados sintéticos de sensores industriais fortemente desbalanceados (99,5% Operacional vs. 0,5% Falha), treina um classificador, compara métricas de avaliação e realiza o ajuste dinâmico do threshold de decisão com base em custo financeiro, gerando um Relatório Executivo em PDF.

### Estrutura do projeto

```
(raiz do repositório)
├── main.py                     # orquestra o pipeline completo
├── requirements.txt
├── src/
│   ├── data_generation.py      # geração dos dados sintéticos desbalanceados
│   ├── preprocessing.py        # split treino/teste + scaler sem data leakage
│   ├── model.py                # treinamento do RandomForestClassifier
│   ├── metrics.py               # acurácia, matriz de confusão, precisão, recall, F-beta, AUC-ROC
│   ├── threshold_analysis.py   # análise financeira e ajuste dinâmico do threshold
│   └── report.py               # geração do Relatório Executivo em PDF
└── outputs/
    └── relatorio_executivo.pdf # gerado ao executar main.py
```

### Como executar

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py
```

O relatório é gerado em `outputs/relatorio_executivo.pdf`.

### Arquitetura da solução e prevenção de Data Leakage

O conjunto de teste é isolado com `train_test_split` (estratificado pela
variável alvo) **antes** de qualquer etapa de pré-processamento
(`src/preprocessing.py`). O `StandardScaler` é ajustado (`fit`) exclusivamente
com estatísticas do conjunto de treino; o conjunto de teste recebe apenas a
transformação (`transform`), nunca participando do cálculo de média/desvio-padrão.
Isso evita que estatísticas do teste (que deveria representar dados nunca
vistos) contaminem o pré-processamento do treino — o que infla artificialmente
as métricas de validação e esconde problemas que só apareceriam em produção.

### Métricas avaliadas

Acurácia, Matriz de Confusão (TP/TN/FP/FN), Precisão, Recall, família F-beta
(F1, F2, F0.5) e AUC-ROC — comparadas explicitamente com a acurácia de um
modelo "ingênuo" que sempre prevê a classe majoritária, para evidenciar a
armadilha da acurácia em datasets desbalanceados.

### Ajuste dinâmico de threshold

`src/threshold_analysis.py` varre thresholds de decisão de 0.01 a 0.99,
calculando o custo total estimado (`FN × custo_FN + FP × custo_FP`) em cada
ponto. Premissas de custo (ajustáveis no topo do arquivo):

- **Falso Negativo** (falha real não detectada → parada não planejada): R$ 50.000,00
- **Falso Positivo** (alarme falso → inspeção preventiva desnecessária): R$ 500,00

O relatório destaca o threshold de menor custo total e a economia estimada em
relação ao threshold padrão (0,5).

### Recomendações de monitoramento em produção

Incluídas na última página do relatório: monitoramento de data drift e concept
drift, retreinamento programado, revisão periódica do threshold e validação de
qualidade de dados na entrada.
