import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Preparação dos Dados
df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Suas 18 features selecionadas
features = [
    'EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3', ] #'std_close5', 'std_open5', 'std_open7', 'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 'EMA_11', 'std_close9']
target = 'trend'

# Divisão Temporal (Hold-out)
train_start, train_end = '2024-01-01', '2024-03-30'
test_start, test_end = '2024-04-01', '2024-06-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train = treino[features]
y_train = treino[target]
X_test = validacao[features]
y_test = validacao[target]

# 2. Configuração do Espaço de Busca (Hiperparâmetros)
param_dist = {
    'n_estimators': [200, 300, 400],           # Focando ao redor de 200
    'max_depth': [4, 5, 6],                    # Focando ao redor de 5
    'learning_rate': [0.005, 0.01, 0.02],      # Reduzindo para aprendizado mais fino
    'subsample': [0.6, 0.7, 0.8],              # Focando ao redor de 0.7
    'colsample_bytree': [0.85, 0.9, 0.95],     # Focando ao redor de 0.9
    'gamma': [0.3, 0.5, 0.7, 1.0],             # Refinando a regularização
    'min_child_weight': [8, 10, 12, 15]        # Testando valores maiores para estabilidade
}

# 3. Execução do Random Search
# O Random Search sorteia combinações aleatórias e valida via Cross-Validation (CV)
xgb = XGBClassifier(eval_metric='logloss', random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=100,             # Se tiver tempo, mantenha alto
    scoring='f1_macro',     # TROCA CRÍTICA: Equilibra acertos de compra e venda
    cv=9,                   
    n_jobs=-1,
    random_state=42
)

print("Iniciando busca de hiperparâmetros...")
random_search.fit(X_train, y_train)

# 4. Avaliação do Melhor Modelo (Best Estimator)
best_model = random_search.best_estimator_

# Predições
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# --- SAÍDA DE RESULTADOS ---
print("\n" + "="*50)
print("MELHORES HIPERPARÂMETROS:")
print(random_search.best_params_)
print("="*50)

# Avaliação de Treino
print("\n[MÉTRICAS DE TREINO]")
print(f"Acurácia: {accuracy_score(y_train, y_pred_train):.4f}")
print("Matriz de Confusão (Treino):")
print(confusion_matrix(y_train, y_pred_train))

# Avaliação de Teste
print("\n[MÉTRICAS DE TESTE - OUT-OF-SAMPLE]")
print(f"Acurácia: {accuracy_score(y_test, y_pred_test):.4f}")
print("Matriz de Confusão (Teste):")
print(confusion_matrix(y_test, y_pred_test))
print("\nRelatório Final:")
print(classification_report(y_test, y_pred_test))