import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

# 1. Preparação dos Dados (Com a sua lógica de datas)
df = pd.read_csv('dataset.csv', parse_dates=['datetime']).sort_values('datetime')

features = [
    'EMA_3', 'SMA_3', 'ADXR', 'Bollinger_Norm', 'EMA_5', 'SMA_5', 
    'std_close3', 'std_open3', 'std_close5', 'std_open5', 'std_open7', 
    'SMA_7', 'std_close7', 'SMA_9', 'std_open9', 'std_close11', 'EMA_11', 'std_close9'
]
target = 'trend'

treino = df[(df['datetime'] >= '2024-01-01') & (df['datetime'] <= '2024-03-30')].copy()
teste = df[(df['datetime'] >= '2024-04-01') & (df['datetime'] <= '2024-06-30')].copy()

X_train, y_train = treino[features], treino[target]
X_test, y_test = teste[features], teste[target]

# 2. Configuração do DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Definição dos Genes (Hiperparâmetros)
toolbox.register("n_estimators", random.randint, 100, 1000)
toolbox.register("max_depth", random.randint, 3, 9)
toolbox.register("learning_rate", random.uniform, 0.005, 0.05)
toolbox.register("subsample", random.uniform, 0.6, 0.9)
toolbox.register("gamma", random.uniform, 0.1, 5.0)
toolbox.register("min_child_weight", random.randint, 1, 30)

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.n_estimators, toolbox.max_depth, toolbox.learning_rate, 
                  toolbox.subsample, toolbox.gamma, toolbox.min_child_weight), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3. Função de Fitness com TRAVAS (Clipping)
def evaluate(individual):
    # Proteção contra valores fora de limite gerados pela mutação
    n_est = max(50, int(individual[0]))
    m_depth = max(2, int(individual[1]))
    l_rate = max(0.001, min(0.1, individual[2]))
    sub = max(0.4, min(1.0, individual[3])) # TRAVA AQUI: Não passa de 1.0
    gam = max(0, individual[4])
    min_child = max(0, int(individual[5]))
    
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        model = XGBClassifier(
            n_estimators=n_est,
            max_depth=m_depth,
            learning_rate=l_rate,
            subsample=sub,
            gamma=gam,
            min_child_weight=min_child,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        scores.append(accuracy_score(y_train.iloc[val_idx], preds))
    
    return np.mean(scores),

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 4. Evolução
def run_evolution():
    pop = toolbox.population(n=50) # Aumentado para 50 para buscar os 64%
    ngen = 30                      # Aumentado para 30 gerações
    hof = tools.HallOfFame(1)      # Garante que o melhor nunca seja perdido
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    
    print("Iniciando Evolução (Versão Estável com Clipping)...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=ngen, 
                        stats=stats, halloffame=hof, verbose=True)
    
    return hof[0]

if __name__ == "__main__":
    best_ind = run_evolution()
    
    # 5. Treinamento Final com o Campeão (Com as travas aplicadas)
    n_est, m_depth, l_rate, sub, gam, min_child = best_ind
    
    final_model = XGBClassifier(
        n_estimators=max(50, int(n_est)),
        max_depth=max(2, int(m_depth)),
        learning_rate=max(0.001, min(0.1, l_rate)),
        subsample=max(0.4, min(1.0, sub)),
        gamma=max(0, gam),
        min_child_weight=max(0, int(min_child)),
        random_state=42
    )
    
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    
    print("\n" + "="*50)
    print(f"MELHORES PARÂMETROS: {best_ind}")
    print(f"ACURÁCIA FINAL (TESTE): {accuracy_score(y_test, y_pred):.4f}")
    print("="*50)
    print(classification_report(y_test, y_pred))