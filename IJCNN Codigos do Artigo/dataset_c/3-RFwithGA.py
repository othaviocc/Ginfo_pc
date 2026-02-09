import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

random.seed(42)
np.random.seed(42)

df = pd.read_csv('normalizados_passo2.csv', parse_dates=['datetime'])

features = ['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3', 'std_close5', 'std_open5', 'std_open7', 'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 'EMA_11', 'std_close9']

'''
['EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 'std_close3', 'std_open3', 'std_close5', 'std_open5', 'std_open7', 'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 'EMA_11', 'std_close9']
'''

target = 'trend'

train_start, train_end = '2024-01-01', '2024-03-30'
test_start, test_end = '2024-04-01', '2024-06-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

X_train = treino[features]
y_train = treino[target]
X_test = validacao[features]
y_test = validacao[target]


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

n_bits_estimators = 8      # Para n_estimators no intervalo [10, 200]
n_bits_max_depth = 5       # Para max_depth no intervalo [1, 30]
n_bits_max_features = 3    # Para max_features no intervalo [1, len(features)]
n_bits_min_samples_leaf = 4 # Para min_samples_leaf no intervalo [1, 20]
n_bits_min_samples_split = 4 # Para min_samples_split no intervalo [2, 20]
total_bits = n_bits_estimators + n_bits_max_depth + n_bits_max_features + n_bits_min_samples_leaf + n_bits_min_samples_split

toolbox.register("attr_bin", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bin, total_bits)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decode_individual(individual):
    """Decodifica um indivíduo binário para um dicionário de hiperparâmetros do RandomForest."""
    def decode_gene(gene, minimo, maximo):
        n_bits = len(gene)
        int_value = int(''.join(map(str, gene)), 2)
        max_binary_value = 2**n_bits - 1
        if max_binary_value == 0: return minimo
        scaled_value = minimo + (int_value / max_binary_value) * (maximo - minimo)
        return int(round(scaled_value))

    params = {}
    idx = 0
    params['n_estimators'] = decode_gene(individual[idx:idx+n_bits_estimators], 10, 200)
    idx += n_bits_estimators
    params['max_depth'] = decode_gene(individual[idx:idx+n_bits_max_depth], 1, 30)
    idx += n_bits_max_depth
    params['max_features'] = decode_gene(individual[idx:idx+n_bits_max_features], 1, len(features))
    idx += n_bits_max_features
    params['min_samples_leaf'] = decode_gene(individual[idx:idx+n_bits_min_samples_leaf], 1, 20)
    idx += n_bits_min_samples_leaf
    params['min_samples_split'] = decode_gene(individual[idx:idx+n_bits_min_samples_split], 2, 20)
    return params

def evaluate(individual):
    params = decode_individual(individual)
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main(n_gen=1000, pop_size=20):
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("max", np.max)

    print("Iniciando a otimização com Algoritmo Genético...")
    
    algorithms.eaSimple(population,
                        toolbox,
                        cxpb=0.8,
                        mutpb=0.05,
                        ngen=n_gen,
                        stats=stats,
                        halloffame=hof,
                        verbose=True)

    best_individual = hof[0]
    best_params = decode_individual(best_individual)

    print(f'\nMelhor acurácia encontrada no conjunto de teste (Fitness): {best_individual.fitness.values[0]:.4f}')
    print(f'Melhores parâmetros encontrados (otimizados para o conjunto de teste):')
    for key, value in best_params.items():
        print(f'  {key:<20} = {value}')

    print("\n--- Avaliando os melhores parâmetros com Validação Cruzada no Treino ---")
    cv_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f'Scores dos 9 folds: {[f"{score:.4f}" for score in cv_scores]}')
    print(f'Acurácia média (CV com 9 folds) nos dados de TREINO: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}')

    print("\n--- Performance Final no Conjunto de Teste (Hold-Out) ---")
    model_final = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    model_final.fit(X_train, y_train)
    y_test_pred = model_final.predict(X_test)
    final_test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Acurácia FINAL no conjunto de TESTE: {final_test_accuracy:.4f}')
    cm_test = confusion_matrix(y_test, y_test_pred)
    print("\nMatriz de confusão - Dados de TESTE:")
    print(cm_test)

    return best_individual, best_params

if __name__ == '__main__':
    best_ind, best_params = main()