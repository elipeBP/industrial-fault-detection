# Respostas — Trabalho III (Dados IoT e Treinamento)

## Questão 1: Arquitetura de Roteamento de Modelos

**Como o código decide qual modelo usar**

A função `prever_ponto_dinamicamente()` lê o campo `status_operacao` do pacote de
telemetria recebido e compara com a lista `status_motor_rodando = [2, 7, 8, 9, 10,
11, 12, 13, 14]`. Se o status estiver nessa lista, o ponto é roteado para
`modelo_rodando`; caso contrário (motor parado, status fora da lista), é roteado
para `modelo_parado`. Cada modelo então recebe um vetor de features distinto —
o modelo de motor rodando exige, além dos sensores brutos, as colunas derivadas
`_media_7dias`, `_std_7dias` e `_desvio_7dias`; o modelo de motor parado usa
apenas as variáveis brutas de sensores e horímetros.

**Por que chaveamento dinâmico é melhor que um modelo único global**

- **Regimes com distribuições estatísticas diferentes:** um compressor parado
  tem vazão, corrente e pressão de saída próximas de zero, enquanto um
  compressor rodando apresenta essas variáveis em faixas operacionais normais.
  Um único modelo teria que aprender uma distribuição multimodal, o que reduz a
  capacidade de discriminação em cada regime específico.
- **Relevância de features distinta por regime:** variáveis como
  `temp_descarga` ou `vazao_medida` são informativas com o motor rodando, mas
  ruído ou constantes com o motor parado. Modelos especializados evitam que
  features irrelevantes em um regime "contaminem" o aprendizado do outro.
- **Ciclo de vida e manutenção independentes:** os dois modelos podem ser
  retreinados, versionados e monitorados (data/model drift) separadamente, sem
  que uma mudança no comportamento de motor parado degrade a performance do
  modelo de motor rodando (e vice-versa).
- **Menor complexidade e melhor calibração:** modelos especialistas tendem a
  ter fronteiras de decisão mais simples e limiares de alerta mais bem
  calibrados para o contexto operacional, reduzindo falsos positivos/negativos
  em comparação com um modelo genérico que tenta cobrir todos os cenários.

Esse padrão é equivalente a uma abordagem de *mixture of experts*: um roteador
simples (baseado em regra de negócio) decide qual especialista consultar, em
vez de forçar um único modelo a generalizar sobre contextos operacionais muito
diferentes.

---

## Questão 2: Análise Crítica de Engenharia de Recursos (Feature Engineering)

### a) Problemas técnicos e operacionais da aproximação atual

No código, `_media_7dias = valor_instantâneo * 0.95`, `_std_7dias = valor_instantâneo
* 0.1` e `_desvio_7dias = valor_instantâneo - média_calculada`. Isso não é uma
agregação temporal real — é uma transformação determinística do dado do
instante atual. Os problemas são:

- **Ausência de histórico real:** a "média de 7 dias" não usa nenhuma leitura
  passada; é apenas 95% do valor atual. Se o sensor tiver uma falha pontual
  (pico ou queda), a média "acompanha" o erro instantaneamente, ao invés de
  suavizá-lo — o que é o oposto do propósito de uma média móvel.
- **Correlação artificial (data leakage estrutural):** como as três features
  derivadas são funções lineares exatas do valor bruto, o modelo, durante o
  treinamento, pode aprender a se apoiar nessa relação sintética e determinística
  em vez de padrões genuínos de degradação ao longo do tempo. Isso infla a
  acurácia percebida em treino/validação, mas não corresponde ao que ocorrerá
  quando (e se) a arquitetura de dados real começar a alimentar agregados
  verdadeiros — gerando **skew entre treino e produção** (train/serve skew).
- **Incapacidade de detectar tendências e anomalias reais:** o desvio nunca
  reflete uma anomalia de fato, pois é sempre ~5% do valor bruto — não importa
  se o sensor está estável ou oscilando violentamente nos últimos 7 dias, o
  "desvio" calculado será sempre proporcional e positivo, não capturando picos,
  quedas ou instabilidade real do equipamento.
- **Risco operacional direto:** falhas graduais (ex.: aumento progressivo de
  temperatura ao longo de dias, típico de desgaste mecânico) são exatamente o
  padrão que uma média/desvio de 7 dias deveria capturar. Como a aproximação
  ignora o passado, o sistema perde a capacidade de antecipar esse tipo de
  falha — indo contra o próprio objetivo do "Modelo de Motor Rodando: falhas
  iminentes em 7 dias".

### b) Como uma arquitetura de dados em nuvem ideal deveria calcular isso

- **Ingestão em stream:** os pacotes de telemetria (via MQTT/LoRaWAN gateway →
  IoT Hub/AWS IoT Core/Kafka) deveriam ser persistidos com timestamp e ID do
  equipamento em um banco de série temporal (ex.: TimescaleDB, InfluxDB,
  BigQuery particionado por tempo).
- **Processamento de janela deslizante:** um processador de stream (Spark
  Structured Streaming, Flink, ou jobs agendados via dbt/Airflow) deveria manter
  o estado de uma janela de 7 dias por equipamento, recalculando incrementalmente
  média, desvio-padrão e desvio a cada nova leitura — considerando watermarking
  para dados atrasados (comum em redes celular/LoRaWAN) e imputação para
  leituras ausentes.
- **Feature Store com paridade treino-produção:** os agregados deveriam ser
  publicados em um *feature store* (ex.: Feast, SageMaker Feature Store,
  Databricks Feature Store), garantindo que a mesma lógica de cálculo seja usada
  tanto para gerar o dataset de treino (point-in-time correct) quanto para
  servir a inferência em tempo real (online store de baixa latência, ex. Redis).
- **Desacoplamento inferência/features:** a aplicação de inferência (o
  Streamlit, no protótipo) apenas consultaria o vetor de features já
  pré-calculado no feature store, em vez de recalcular aproximações inline —
  eliminando duplicidade de lógica e o risco de divergência entre o que foi
  treinado e o que é servido.

---

## Questão 3: Simulação de Latência e Gargalos no Streamlit

### a) Fenômeno real simulado

A linha `tempo_de_espera = intervalo_base_segundos + random.uniform(0,
atraso_iot_max_segundos)` simula a **latência e o jitter de rede** típicos de
comunicação IoT via celular/LoRaWAN: os pacotes não chegam em intervalos
perfeitamente regulares. Há variação por congestionamento de rede, retransmissões,
tempo de fila no gateway, cobertura de sinal instável e, no caso específico do
LoRaWAN, restrições de duty cycle/airtime que atrasam o próximo envio. O
`intervalo_base_segundos` representa o período programado de telemetria e o
componente aleatório representa esse atraso variável e não determinístico.

### b) Problema do `while` + `time.sleep()` e alternativa no Streamlit

O laço `while st.session_state.running: ... time.sleep(tempo_de_espera)` bloqueia
a **thread principal de execução do script** do Streamlit. Como o Streamlit
funciona re-executando o script a cada interação, um laço bloqueante impede que
o servidor processe novos eventos (cliques em botões, WebSocket de UI, etc.)
enquanto está "dormindo", o que pode travar a responsividade da interface e
dificultar até mesmo parar a simulação clicando no botão.

A alternativa recomendada é usar **`st.fragment` com o parâmetro `run_every`**
(disponível a partir do Streamlit 1.33+): um fragmento decorado com
`@st.fragment(run_every="10s")` é re-executado automaticamente no intervalo
definido, sem laço bloqueante e sem travar o restante da página, permitindo que
o resto da interface (botões, outros elementos) continue responsivo entre as
atualizações. Uma alternativa equivalente, para versões mais antigas ou
projetos que preferem um componente de terceiros, é o `streamlit-autorefresh`
(`st_autorefresh(interval=...)`), que dispara reruns periódicos via timer no
navegador em vez de `time.sleep()` no lado do servidor.
