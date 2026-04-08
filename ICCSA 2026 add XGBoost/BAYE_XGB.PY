import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from bayes_opt import BayesianOptimization

# 1. Preparação dos Dados
df = pd.read_csv('dataset.csv', parse_dates=['datetime']).sort_values('datetime')

features = [
    'EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 
    'std_close3', 'std_open3', 'std_close5', 'std_open5', 'std_open7', 
    'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 'EMA_11', 'std_close9'
]
target = 'trend'

treino_df = df[(df['datetime'] >= '2024-01-01') & (df['datetime'] <= '2024-03-30')].copy()
teste_df = df[(df['datetime'] >= '2024-04-01') & (df['datetime'] <= '2024-06-30')].copy()

X_train, y_train = treino_df[features], treino_df[target]
X_test, y_test = teste_df[features], teste_df[target]

# 2. Função objetivo (fitness)
def xgb_evaluate(n_estimators, max_depth, learning_rate, subsample, gamma, min_child_weight):
    
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": max(0.001, learning_rate),
        "subsample": max(0.1, min(1.0, subsample)),
        "gamma": max(0, gamma),
        "min_child_weight": int(min_child_weight),
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": -1
    }
    
    try:
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)
    except:
        return 0.0

# 3. Espaço de busca
pbounds = {
    'n_estimators': (100, 1200),
    'max_depth': (3, 10),
    'learning_rate': (0.005, 0.1),
    'subsample': (0.4, 0.9),
    'gamma': (0, 10),
    'min_child_weight': (1, 40)
}

# 4. Otimizador Bayesiano
optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

print("Iniciando Otimização Bayesiana...")
optimizer.maximize(
    init_points=10,   # explora aleatoriamente
    n_iter=5000         # iterações bayesianas
)

# 5. Melhor resultado
best_params = optimizer.max['params']

# Ajustando tipos
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])

# 6. Modelo final
final_model = XGBClassifier(
    **best_params,
    random_state=42
)

final_model.fit(X_train, y_train)

# Avaliação
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

print("\n" + "="*50)
print(f"MELHORES PARÂMETROS: {best_params}")

print(f"\nACURÁCIA DE TREINO: {accuracy_score(y_train, y_train_pred):.4f}")
print("Matriz de Confusão (Treino):")
print(confusion_matrix(y_train, y_train_pred))

print(f"\nACURÁCIA DE TESTE: {accuracy_score(y_test, y_test_pred):.4f}")
print("Matriz de Confusão (Teste):")
print(confusion_matrix(y_test, y_test_pred))

print("\nRelatório Final (Teste):")
print(classification_report(y_test, y_test_pred))