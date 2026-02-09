from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


df = pd.read_csv('normalizados_passo2.csv', parse_dates=['datetime'])

features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3', 'std_close5', 'std_open5', 'std_open7', 'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 'EMA_11', 'std_close9']

target = 'trend'  

train_start, train_end = '2024-01-01', '2024-03-30'
test_start, test_end = '2024-04-01', '2024-06-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train = treino[features]
y_train = treino[target]
X_test = validacao[features]
y_test = validacao[target]

# Escalonamento para MLP
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Bits para cada parâmetro
n_bits_hidden_layer1 = 5    # 1-32 neurônios
n_bits_hidden_layer2 = 5    # 0-32 neurônios (0 = sem segunda camada)
n_bits_activation = 2       # 0-3 ('identity', 'logistic', 'tanh', 'relu')
n_bits_solver = 2           # 0-2 ('lbfgs', 'sgd', 'adam')
n_bits_alpha = 8            # 0.0001 a 0.1
n_bits_lr_init = 8          # 0.0001 a 0.1

total_bits = (n_bits_hidden_layer1 + n_bits_hidden_layer2 + n_bits_activation +
              n_bits_solver + n_bits_alpha + n_bits_lr_init)

toolbox.register("attr_bin", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bin, total_bits)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def decode_binary(gene, minimo, maximo, n_bits):
    binary_str = ''.join(map(str, gene))
    int_value = int(binary_str, 2)
    max_value = 2 ** n_bits - 1
    return minimo + (int_value / max_value) * (maximo - minimo)

def evaluate(individual):
    idx = 0

    hidden1 = int(decode_binary(individual[idx:idx+n_bits_hidden_layer1], 1, 32, n_bits_hidden_layer1))
    idx += n_bits_hidden_layer1

    hidden2 = int(decode_binary(individual[idx:idx+n_bits_hidden_layer2], 0, 32, n_bits_hidden_layer2))
    idx += n_bits_hidden_layer2

    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs', 'sgd', 'adam']

    activation_idx = min(int(decode_binary(individual[idx:idx+n_bits_activation], 0, len(activations)-1, n_bits_activation)), len(activations)-1)
    idx += n_bits_activation

    solver_idx = min(int(decode_binary(individual[idx:idx+n_bits_solver], 0, len(solvers)-1, n_bits_solver)), len(solvers)-1)
    idx += n_bits_solver

    alpha = decode_binary(individual[idx:idx+n_bits_alpha], 0.0001, 0.1, n_bits_alpha)
    idx += n_bits_alpha

    lr_init = decode_binary(individual[idx:idx+n_bits_lr_init], 0.0001, 0.1, n_bits_lr_init)

    hidden_layers = (hidden1,) if hidden2 == 0 else (hidden1, hidden2)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activations[activation_idx],
        solver=solvers[solver_idx],
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=2000,  # aumenta iterações para melhorar convergência
        random_state=42
    )

    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
    except:
        acc = 0  # penaliza combinações inválidas

    return acc,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(n_gen=1000, pop_size=30):
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(population, toolbox, cxpb=0.8, mutpb=0.05, ngen=n_gen,
                        stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    idx = 0

    hidden1 = int(decode_binary(best_ind[idx:idx+n_bits_hidden_layer1], 1, 200, n_bits_hidden_layer1))
    idx += n_bits_hidden_layer1
    hidden2 = int(decode_binary(best_ind[idx:idx+n_bits_hidden_layer2], 0, 200, n_bits_hidden_layer2))
    idx += n_bits_hidden_layer2

    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs', 'sgd', 'adam']

    activation_idx = min(int(decode_binary(best_ind[idx:idx+n_bits_activation], 0, len(activations)-1, n_bits_activation)), len(activations)-1)
    idx += n_bits_activation

    solver_idx = min(int(decode_binary(best_ind[idx:idx+n_bits_solver], 0, len(solvers)-1, n_bits_solver)), len(solvers)-1)
    idx += n_bits_solver

    alpha = decode_binary(best_ind[idx:idx+n_bits_alpha], 0.0001, 0.1, n_bits_alpha)
    idx += n_bits_alpha
    lr_init = decode_binary(best_ind[idx:idx+n_bits_lr_init], 0.0001, 0.1, n_bits_lr_init)

    hidden_layers = (hidden1,) if hidden2 == 0 else (hidden1, hidden2)

    print(f'\nMelhor acurácia: {best_ind.fitness.values[0]:.4f}')
    print('Melhores parâmetros encontrados:')
    print(f'  hidden_layer_sizes = {hidden_layers}')
    print(f'  activation         = {activations[activation_idx]}')
    print(f'  solver             = {solvers[solver_idx]}')
    print(f'  alpha              = {alpha:.5f}')
    print(f'  learning_rate_init = {lr_init:.5f}')

    # Treina modelo final
    model_final = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activations[activation_idx],
        solver=solvers[solver_idx],
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=2000,
        random_state=42
    )
    model_final.fit(X_train, y_train)

    # Predições
    y_train_pred = model_final.predict(X_train)
    y_test_pred = model_final.predict(X_test)

    print("\nMatriz de confusão - Dados de TREINAMENTO:")
    print(confusion_matrix(y_train, y_train_pred))
    print("\nMatriz de confusão - Dados de TESTE:")
    print(confusion_matrix(y_test, y_test_pred))

    return best_ind

best = main()
