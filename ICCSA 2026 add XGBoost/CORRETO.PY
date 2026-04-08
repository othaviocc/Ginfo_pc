import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, 
                             roc_curve, matthews_corrcoef, classification_report)

# --- 1. Carregamento e Preparação ---
df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3']
target = 'trend'  

train_start, train_end = '2024-01-01', '2024-03-30'
test_start, test_end = '2024-04-01', '2024-06-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train, y_train = treino[features], treino[target]
X_test, y_test = validacao[features], validacao[target]

# --- 2. Configuração do Modelo (Parâmetros Originais) ---
final_model = MLPClassifier(
    hidden_layer_sizes=(4, 6),
    activation='relu',
    solver='lbfgs',
    alpha=0.09804,
    learning_rate_init=0.02713,
    random_state=42,
    max_iter=1000
)

# Treinamento
final_model.fit(X_train, y_train)

# Predições
y_pred = final_model.predict(X_test)
y_probs = final_model.predict_proba(X_test)[:, 1] # Probabilidade da classe 1

# --- 3. Cálculo das Métricas ---
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_probs)

# --- 4. Exibição dos Resultados no Console ---
print("="*40)
print(f"{'RESULTADOS GLOBAIS MINI ÍNDICE':^40}")
print("="*40)
print(f"Acurácia: {acc:.4f}")
print(f"MCC:      {mcc:.4f}")
print(f"ROC AUC:  {auc_score:.4f}")
print("-" * 40)
print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nRelatório:")
print(classification_report(y_test, y_pred))

# --- 5. Plots ---
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Plot A: Curva ROC
# --- Plot da Curva ROC (Versão em Inglês) ---
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#2c3e50', lw=3, label=f'ROC AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], color='#e74c3c', linestyle='--', label='Random Guess (0.5)')
plt.fill_between(fpr, tpr, alpha=0.1, color='#2c3e50')

# Legendas e Eixos em Inglês
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# --- Matriz de Confusão Visual (Opcional - Inglês) ---
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title(f'Confusion Matrix\nMCC: {mcc:.4f}')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')


plt.tight_layout()
plt.show()


import math

# Dados da sua imagem (Tabela 8)
tn, fp = 2781, 1477
fn, tp = 1507, 2541

# Cálculo manual do MCC
def calc_mcc(tp, tn, fp, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator

mcc_real = calc_mcc(tp, tn, fp, fn)
acc_real = (tp + tn) / (tp + tn + fp + fn)

print(f"Resultados do Modelo da Imagem:")
print(f"Acurácia: {acc_real:.4f}")
print(f"MCC:      {mcc_real:.4f}")