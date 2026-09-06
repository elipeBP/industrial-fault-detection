# ==============================================================================
# SCRIPT DE TREINAMENTO E GERAÇÃO DOS MODELOS (.JOBLIB)
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

print("Gerando modelos de IA...")

# 1. Definição das variáveis (iguais ao simulador)
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

# Features esperadas pelo modelo de motor rodando (incluindo as calculadas de 7 dias)
features_rodando = []
for var in variaveis_continuas_sensores + lista_horimetros:
    features_rodando.extend([var, f"{var}_media_7dias", f"{var}_std_7dias", f"{var}_desvio_7dias"])

# Features esperadas pelo modelo de motor parado
features_parado = variaveis_continuas_sensores + lista_horimetros

# 2. Criando dados sintéticos para simular o treinamento
np.random.seed(42)
n_amostras = 200

# Dados simulados Rodando
X_rodando = pd.DataFrame(np.random.uniform(5, 100, size=(n_amostras, len(features_rodando))), columns=features_rodando)
y_rodando = np.random.choice([0, 1], size=n_amostras, p=[0.85, 0.15]) # 15% de probabilidade de falha

# Dados simulados Parado
X_parado = pd.DataFrame(np.random.uniform(0, 50, size=(n_amostras, len(features_parado))), columns=features_parado)
y_parado = np.random.choice([0, 1], size=n_amostras, p=[0.95, 0.05]) # 5% de probabilidade de falha

# 3. Treinando os modelos
modelo_rodando = RandomForestClassifier(n_estimators=50, random_state=42)
modelo_rodando.fit(X_rodando, y_rodando)

modelo_parado = RandomForestClassifier(n_estimators=50, random_state=42)
modelo_parado.fit(X_parado, y_parado)

# 4. Salvando os arquivos .joblib
joblib.dump(modelo_rodando, 'modelo_motor_rodando.joblib')
joblib.dump(modelo_parado, 'modelo_motor_parado.joblib')

print(" Sucesso! Modelos gerados com sucesso na pasta:")
print("   - modelo_motor_rodando.joblib")
print("   - modelo_motor_parado.joblib")