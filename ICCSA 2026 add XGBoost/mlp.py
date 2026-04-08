import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, 
                             roc_curve, matthews_corrcoef, classification_report)

# --- Carregamento e Preparação ---
df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

features = [
    'EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 
    'std_close3', 'std_open3']
target = 'trend'  

train_start, train_end = '2024-01-01', '2024-03-30'
test_start, test_end = '2024-04-01', '2024-06-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train, y_train = treino[features], treino[target]
X_test, y_test = validacao[features], validacao[target]

# --- Configuração do Modelo com seus Parâmetros Encontrados ---
final_model = MLPClassifier(
    hidden_layer_sizes=(4, 6),  # Seus parâmetros
    activation='relu',          # Seus parâmetros
    solver='lbfgs',             # Seus parâmetros
    alpha=0.09804,              # Seus parâmetros
    learning_rate_init=0.02713, # Seus parâmetros
    random_state=42,            # Mantido para reprodutibilidade
    max_iter=1000               # Aumentado para garantir convergência do lbfgs
)

# Treinamento
final_model.fit(X_train, y_train)

# Predições
y_pred = final_model.predict(X_test)
y_probs = final_model.predict_proba(X_test)[:, 1]

# --- Cálculo das Métricas ---
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_probs)

# --- Exibição dos Resultados ---
print("="*40)
print(f"{'RESULTADOS MINI ÍNDICE (5MIN)':^40}")
print("="*40)
print(f"Acurácia: {acc:.4f}")
print(f"MCC:      {mcc:.4f}")
print(f"ROC AUC:  {auc_score:.4f}")
print("-"*40)
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# --- Plot da Curva ROC ---
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#2c3e50', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='#e74c3c', linestyle='--', label='Chute Aleatório (0.5)')
plt.fill_between(fpr, tpr, alpha=0.1, color='#2c3e50')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC - MLP (Mini Índice)')
plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

# --- (Assumindo que os dados X_train, y_train, X_test, y_test já estão carregados) ---

# Re-treinando o modelo com seus parâmetros confirmados
final_model = MLPClassifier(
    hidden_layer_sizes=(4, 6),
    activation='relu',
    solver='lbfgs',
    alpha=0.09804,
    learning_rate_init=0.02713,
    random_state=42,
    max_iter=1000
)
final_model.fit(X_train, y_train)

# 1. Obter as probabilidades em vez de apenas a classe
# y_probs[:, 0] é a prob de ser 0 (venda), y_probs[:, 1] é a prob de ser 1 (compra)
y_probs = final_model.predict_proba(X_test)

# 2. Definir o Limiar de Confiança (Threshold)
# Só operamos se a confiança for maior que 60%
confianca = 0.60


# Criar filtros
filtro_compra = y_probs[:, 1] >= confianca
filtro_venda = y_probs[:, 0] >= confianca
filtro_total = filtro_compra | filtro_venda

# 3. Aplicar o filtro nos dados de teste
y_test_filtrado = y_test[filtro_total]
y_pred_filtrado = np.where(y_probs[filtro_total, 1] >= confianca, 1, 0)

# --- Resultados ---
acc_original = accuracy_score(y_test, final_model.predict(X_test))
acc_filtrada = accuracy_score(y_test_filtrado, y_pred_filtrado)
mcc_filtrado = matthews_corrcoef(y_test_filtrado, y_pred_filtrado)

n_trades_antes = len(y_test)
n_trades_depois = len(y_test_filtrado)

print("="*45)
print(f"{'BACKTEST COM FILTRO DE CONFIANÇA (>60%)':^45}")
print("="*45)
print(f"Acurácia Original:    {acc_original:.4f} ({n_trades_antes} trades)")
print(f"Acurácia Filtrada:    {acc_filtrada:.4f} ({n_trades_depois} trades)")
print(f"MCC Filtrado:         {mcc_filtrado:.4f}")
print(f"Redução de Sinais:    {100 * (1 - n_trades_depois/n_trades_antes):.2f}%")
print("-" * 45)
print("Matriz de Confusão (Sinais Filtrados):")
print(confusion_matrix(y_test_filtrado, y_pred_filtrado))

# --- Plot de Sensibilidade ---
thresholds = np.linspace(0.5, 0.75, 10)
accs = []
trades = []

for t in thresholds:
    f = (y_probs[:, 1] >= t) | (y_probs[:, 0] >= t)
    if any(f):
        accs.append(accuracy_score(y_test[f], np.where(y_probs[f, 1] >= t, 1, 0)))
        trades.append(len(y_test[f]))
    else:
        accs.append(None)
        trades.append(0)

plt.figure(figsize=(10, 5))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(thresholds, accs, 'g-', marker='o', label='Acurácia')
ax2.bar(thresholds, trades, alpha=0.2, color='blue', width=0.02, label='Nº de Trades')
ax1.set_xlabel('Nível de Confiança (Threshold)')
ax1.set_ylabel('Acurácia', color='g')
ax2.set_ylabel('Quantidade de Trades', color='b')
plt.title('Efeito do Filtro de Confiança no Mini Índice')
plt.show()

import matplotlib.pyplot as plt

# Vamos calcular o MCC para vários níveis de confiança
thresholds = np.linspace(0.5, 0.70, 20)
mcc_values = []

for t in thresholds:
    f = (y_probs[:, 1] >= t) | (y_probs[:, 0] >= t)
    if any(f):
        m = matthews_corrcoef(y_test[f], np.where(y_probs[f, 1] >= t, 1, 0))
        mcc_values.append(m)
    else:
        mcc_values.append(0)

plt.figure(figsize=(8, 4))
plt.plot(thresholds, mcc_values, color='purple', lw=2, marker='s', markersize=4)
plt.axhline(y=0.28, color='r', linestyle='--', label='Seu MCC Atual (0.60)')
plt.fill_between(thresholds, mcc_values, alpha=0.2, color='purple')
plt.xlabel('Confiança Exigida (Threshold)')
plt.ylabel('Qualidade do Modelo (MCC)')
plt.title('Estabilidade da Vantagem Estatística (MCC)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()