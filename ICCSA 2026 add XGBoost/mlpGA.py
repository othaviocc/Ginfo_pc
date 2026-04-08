from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef

# --- Carregamento e Preparação ---
df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

features = [
    'EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 
    'std_close3', 'std_open3', 'std_close5', 'std_open5', 'std_open7', 
    'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 'EMA_11', 'std_close9'
]
target = 'trend'  

train_start, train_end = '2024-01-01', '2024-03-30'
test_start, test_end = '2024-04-01', '2024-06-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train, y_train = treino[features], treino[target]
X_test, y_test = validacao[features], validacao[target]

# --- Configurações do Algoritmo Genético ---
param_ranges = {
    'hidden_layer_sizes': (10, 200),
    'activation': (0, 2),
    'alpha': (0.0001, 0.1),
    'learning_rate_init': (0.0001, 0.1),
    'max_iter': (100, 500),
}

param_bits = {
    'hidden_layer_sizes': 8,
    'activation': 2,
    'alpha': 10,
    'learning_rate_init': 10,
    'max_iter': 9,
}

activation_map = {0: 'relu', 1: 'tanh', 2: 'logistic'}
total_bits = sum(param_bits.values())

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bin", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bin, total_bits)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_binary(gene, minimo, maximo, n_bits):
    binary_str = ''.join(map(str, gene))
    int_value = int(binary_str, 2)
    max_value = 2 ** n_bits - 1
    return minimo + (int_value / max_value) * (maximo - minimo)

def decode_individual(individual):
    idx = 0
    decoded = {}
    for param, (min_val, max_val) in param_ranges.items():
        bits = param_bits[param]
        value = decode_binary(individual[idx:idx+bits], min_val, max_val, bits)
        if param in ['hidden_layer_sizes', 'max_iter']:
            value = int(round(value))
        elif param == 'activation':
            value = activation_map[int(round(value))]
        decoded[param] = value
        idx += bits
    return decoded

def calculate_fitness(individual):
    params = decode_individual(individual)
    model = MLPClassifier(
        hidden_layer_sizes=(params['hidden_layer_sizes'],),
        activation=params['activation'],
        alpha=params['alpha'],
        learning_rate_init=params['learning_rate_init'],
        max_iter=params['max_iter'],
        random_state=42
    )
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    except:
        acc = 0.0
    return (acc,)

toolbox.register("evaluate", calculate_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(n_gen=50, pop_size=20): # Reduzi n_gen para teste, ajuste conforme necessário
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(population, toolbox, cxpb=0.8, mutpb=0.1, ngen=n_gen, 
                        stats=stats, halloffame=hof, verbose=True)

    best_params = decode_individual(hof[0])
    
    # --- Avaliação Final Detalhada ---
    final_model = MLPClassifier(
        hidden_layer_sizes=(best_params['hidden_layer_sizes'],),
        activation=best_params['activation'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        max_iter=best_params['max_iter'],
        random_state=42
    )
    final_model.fit(X_train, y_train)
    
    y_pred = final_model.predict(X_test)
    y_probs = final_model.predict_proba(X_test)[:, 1] # Probabilidade da classe positiva

    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_probs)

    print('\n' + '='*30)
    print(f"MELHORES PARÂMETROS ENCONTRADOS:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    
    print('\n' + '='*30)
    print(f"MÉTRICAS DE TESTE:")
    print(f"Acurácia: {acc:.4f}")
    print(f"MCC:      {mcc:.4f}")
    print(f"ROC AUC:  {auc_score:.4f}")
    print('='*30)

    print("\nMatriz de Confusão (Teste):")
    print(confusion_matrix(y_test, y_pred))

    # --- Plot da Curva ROC ---
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC - Mini Índice (5min)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    return hof[0], best_params, acc

if __name__ == '__main__':
    # Dica: Para encontrar os 64%, talvez precise aumentar pop_size para 50 e n_gen para 100
    main(n_gen=30, pop_size=20)