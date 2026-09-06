# ==============================================================================
# SIMULADOR DE PREVISÃO EM TEMPO REAL
# CÓDIGO AGORA INCLUI ATRASOS ALEATÓRIOS DO IoT PARA UMA SIMULAÇÃO MAIS REALISTA.
# ==============================================================================

# 1. IMPORTANDO BIBLIOTECAS
import streamlit as st
import pandas as pd
import joblib
import time
import random

# Definir as listas de variáveis e status, idênticas às do script de treinamento.
variaveis_continuas_sensores = [
    'pressao_saida', 'pressao_saida_max', 'pressao_saida_min', 'pressao_saida_med',
    'pressao_interna', 'pressao_interna_max', 'pressao_interna_min', 'pressao_interna_med',
    'temp_descarga', 'temp_descarga_max', 'temp_descarga_min', 'temp_descarga_med',
    'vazao_min', 'vazao_max', 'vazao_med', 'vazao_medida', 'vazao_max_medida', 'vazao_min_medida',
    'potencia_med', 'potencia_max',
    'corrente_max', 'corrente_med', 'corrente_min', 'corrente'
]
lista_horimetros = [
    'horas_total_operacao', 'horas_carga', 'horas_tubo_nylon', 'horas_termostatica',
    'horas_reaperto_eletrica', 'horas_limpeza', 'horas_manutencao_ftr_ar',
    'horas_manutencao_ftr_oleo', 'horas_manutencao_separador',
    'horas_manutencao_graxa_motor', 'horas_manutencao_oleo'
]
status_motor_rodando = [2, 7, 8, 9, 10, 11, 12, 13, 14]

# 2. CARREGAR OS MODELOS UMA ÚNICA VEZ
@st.cache_resource
def carregar_modelos():
    try:
        modelo_rodando = joblib.load('modelo_motor_rodando.joblib')
        modelo_parado = joblib.load('modelo_motor_parado.joblib')
        return modelo_rodando, modelo_parado
    except FileNotFoundError:
        st.error("Erro: Modelos não encontrados! Por favor, execute o script de treinamento para gerá-los.")
        return None, None

modelo_rodando, modelo_parado = carregar_modelos()
if not modelo_rodando or not modelo_parado:
    st.stop()


# 3. FUNÇÃO DE PREVISÃO QUE FAZ O CHAVEAMENTO
def prever_ponto_dinamicamente(novo_ponto_de_dados):
    """
    Recebe um único ponto de dados e decide qual modelo usar.
    """
    status = novo_ponto_de_dados.get('status_operacao')
    previsao_label = "Dados inválidos"
    
    if status in status_motor_rodando:
        # Pega as features que o modelo de motor rodando espera
        features_treinadas = modelo_rodando.feature_names_in_
        features_dict = {}

        # Itera sobre a lista de features que o modelo treinado espera
        for col in features_treinadas:
            # Verifica se é uma feature de engenharia ou um dado bruto
            if '_media_7dias' in col:
                base_col = col.replace('_media_7dias', '')
                features_dict[col] = [novo_ponto_de_dados.get(base_col, 0) * 0.95]
            elif '_std_7dias' in col:
                base_col = col.replace('_std_7dias', '')
                features_dict[col] = [novo_ponto_de_dados.get(base_col, 0) * 0.1]
            elif '_desvio_7dias' in col:
                base_col = col.replace('_desvio_7dias', '')
                media = novo_ponto_de_dados.get(base_col, 0) * 0.95
                features_dict[col] = [novo_ponto_de_dados.get(base_col, 0) - media]
            else:
                # Preenche com o valor do dado bruto ou 0 se não existir
                features_dict[col] = [novo_ponto_de_dados.get(col, 0)]
                
        features_rodando = pd.DataFrame(features_dict)
        previsao = modelo_rodando.predict(features_rodando)[0]
        
        # Ajusta a mensagem para refletir a previsão de 7 dias
        previsao_label = "Motor Rodando: " + ("Possível Falha em 7 Dias" if previsao == 1 else "Normal (sem previsão de falha em 7 dias)")
        
    else:
        # Pega as features que o modelo de motor parado espera
        features_treinadas = modelo_parado.feature_names_in_
        features_dict = {}

        for col in features_treinadas:
            features_dict[col] = [novo_ponto_de_dados.get(col, 0)]
            
        features_parado = pd.DataFrame(features_dict)
        previsao = modelo_parado.predict(features_parado)[0]
        
        # Ajusta a mensagem para refletir a previsão de 7 dias
        previsao_label = "Motor Parado: " + ("Possível Falha em 7 Dias" if previsao == 1 else "Normal (sem previsão de falha em 7 dias)")
        
    return previsao_label

# ==============================================================================
# Interface do Streamlit para Simulação em Tempo Real
# ==============================================================================
st.set_page_config(page_title="Monitoramento em Tempo Real", layout="wide")

st.title("Sistema de Predição de Falhas em 7 Dias")
st.markdown("---")
st.markdown("### ⚙️ Status do Compressor")

placeholder_status = st.empty()
placeholder_metrics = st.empty()

st.markdown("### 📋 Histórico Recente (últimas 5 leituras)")
placeholder_historico = st.empty()

# Parâmetros da simulação realística de IoT
intervalo_base_segundos = 10 # Simula o intervalo de 10 minutos entre os envios
atraso_iot_max_segundos = 5  # Simula um atraso aleatório de até 5 segundos

if 'running' not in st.session_state:
    st.session_state.running = False

if 'historico' not in st.session_state:
    st.session_state.historico = []

if st.button('Iniciar Simulação' if not st.session_state.running else 'Parar Simulação'):
    st.session_state.running = not st.session_state.running

# Exibe o histórico já armazenado na sessão (persiste mesmo com a simulação parada)
with placeholder_historico.container():
    if st.session_state.historico:
        st.dataframe(pd.DataFrame(st.session_state.historico), use_container_width=True)
    else:
        st.caption("Nenhuma leitura registrada ainda.")

if st.session_state.running:
    st.info("Simulação em execução... Processando um novo ponto de dado a cada 10 segundos, com atrasos aleatórios do IoT.")
    
    while st.session_state.running:
        # Simula o fluxo de dados em tempo real com valores aleatórios
        current_status = random.choice(status_motor_rodando + [0])
        novo_ponto = {
            'status_operacao': current_status,
            'contagem_falhas_criticas': 0, 
            'contagem_falhas_nao_criticas': 0,
            **{col: random.uniform(5, 20) for col in variaveis_continuas_sensores},
            **{col: random.uniform(100, 5000) for col in lista_horimetros}
        }
        
        label = prever_ponto_dinamicamente(novo_ponto)
        horario_leitura = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        with placeholder_status.container():
            st.markdown(f"**Última Leitura:** `{horario_leitura}`")
            st.markdown(f"**Status de Operação:** `{novo_ponto['status_operacao']}`")

        with placeholder_metrics.container():
            if "Falha" in label:
                st.error(f"**{label}**")
            else:
                st.success(f"**{label}**")

        # Registra a leitura no histórico da sessão, mantendo apenas as últimas 5
        st.session_state.historico.append({
            'Horário': horario_leitura,
            'Status de Operação': novo_ponto['status_operacao'],
            'Pressão de Saída': round(novo_ponto['pressao_saida'], 2),
            'Predição': "Possível Falha" if "Falha" in label else "Normal"
        })
        st.session_state.historico = st.session_state.historico[-5:]

        with placeholder_historico.container():
            st.dataframe(pd.DataFrame(st.session_state.historico), use_container_width=True)

        # Pausa para simular o intervalo de envio e o atraso do IoT
        tempo_de_espera = intervalo_base_segundos + random.uniform(0, atraso_iot_max_segundos)
        time.sleep(tempo_de_espera)