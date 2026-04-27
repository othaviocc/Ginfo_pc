import numpy as np
import pandas as pd
import random
import warnings
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from deap import base, creator, tools, algorithms

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


class Fuzzifier:
    def __init__(self, mf_type='triangular', n_partitions=3):
        self.mf_type = mf_type
        self.n_partitions = n_partitions
        self.params = {}

    def fit(self, X):
        for col in X.columns:
            min_val, max_val = X[col].min(), X[col].max()
            if min_val == max_val: max_val += 1e-9 
            if 'gmfmfs' in self.mf_type:
                quantiles = np.linspace(0, 1, self.n_partitions + 1)
                self.params[col] = np.unique(np.quantile(X[col].dropna(), quantiles))
            else:
                self.params[col] = np.linspace(min_val, max_val, self.n_partitions)

    def transform(self, X):
        X_fuzzy = pd.DataFrame(index=X.index)
        for col in X.columns:
            params = self.params.get(col)
            if 'gmfmfs' in self.mf_type:
                for i in range(len(params) - 1):
                    p_at, p_px = params[i], params[i+1]
                    col_name = f"{col}_{self.mf_type}_{i}"
                    mf_lin = np.where((X[col] >= p_at) & (X[col] <= p_px), (X[col]-p_at)/(p_px-p_at+1e-9), 0)
                    if self.mf_type == 'gmfmfs_nonlinear':
                        mf_fuz = np.where(mf_lin <= 0.5, 2*mf_lin**2, 1-2*(1-mf_lin)**2)
                    else: mf_fuz = mf_lin
                    X_fuzzy[col_name] = mf_fuz
            else:
                centers = params
                w = centers[1] - centers[0] if len(centers) > 1 else 1.0
                for i, c in enumerate(centers):
                    col_name = f"{col}_{self.mf_type}_{i}"
                    if self.mf_type == 'triangular':
                        mf = np.maximum(0, 1 - np.abs(X[col] - c) / w)
                    elif self.mf_type == 'gaussian':
                        mf = np.exp(-0.5 * ((X[col] - c) / (w/1.5))**2)
                    elif self.mf_type == 'trapezoidal':
                        mf = np.clip(1 - (np.maximum(0, np.abs(X[col] - c) - w*0.2) / (w*0.8)), 0, 1)
                    X_fuzzy[col_name] = mf
        return X_fuzzy


class FuzzyDecisionTree:
    def __init__(self, max_depth=8, min_samples_split=2, min_samples_leaf=1):
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion='entropy', 
            random_state=SEED
        )
        
    def fit(self, X_f, y): 
        self.tree.fit(X_f, y)
        
    def predict_proba(self, X_f): 
        return self.tree.predict_proba(X_f)
    
    def predict_proba_soft(self, X_f):
        t = self.tree.tree_
        n_s = X_f.shape[0]
        res = np.zeros((n_s, self.tree.n_classes_))
        q = [(0, np.ones(n_s))]
        X_a = X_f.values
        while q:
            node, weights = q.pop(0)
            if np.all(weights == 0): continue
            if t.children_left[node] == t.children_right[node]:
                v = t.value[node][0]
                res += weights[:, np.newaxis] * (v / v.sum())
            else:
                feat = t.feature[node]
                mu = np.clip(X_a[:, feat], 0, 1)
                q.append((t.children_right[node], weights * mu))
                q.append((t.children_left[node], weights * (1-mu)))
        return res


class FuzzyRandomForest:
    def __init__(self, n_estimators=100, mf_type='gaussian', n_partitions=5, voting='MWLFUS', 
                 max_depth=8, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.voting = voting
        self.mf_type = mf_type
        self.n_partitions = n_partitions
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        
        self.fuzzifier = Fuzzifier(mf_type=mf_type, n_partitions=n_partitions)
        self.trees, self.features_per_tree, self.tree_weights = [], [], []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fuzzifier.fit(X)
        X_f = self.fuzzifier.transform(X)
        n_s, n_f = X_f.shape
        
        # Ajuste dinâmico do max_features
        if self.max_features == 'sqrt': m_f = max(1, int(np.sqrt(n_f)))
        elif self.max_features == 'log2': m_f = max(1, int(np.log2(n_f)))
        elif isinstance(self.max_features, float): m_f = max(1, int(self.max_features * n_f))
        else: m_f = n_f
            
        for _ in range(self.n_estimators):
            idx = np.random.choice(n_s, size=n_s, replace=True)
            oob = np.array(list(set(range(n_s)) - set(idx)))
            feats = np.random.choice(n_f, size=m_f, replace=False)
            self.features_per_tree.append(feats)
            
            fdt = FuzzyDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            fdt.fit(X_f.iloc[idx, feats], y.iloc[idx])
            
            if len(oob) > 0:
                preds_o = fdt.tree.predict(X_f.iloc[oob, feats])
                acc = accuracy_score(y.iloc[oob], preds_o)
                self.tree_weights.append(acc)
                if self.voting == 'MWLFUS':
                    fdt.error_tree = (NearestNeighbors(n_neighbors=5).fit(X_f.iloc[oob, feats]), (preds_o != y.iloc[oob]).astype(int).values)
            else: self.tree_weights.append(1.0)
            self.trees.append(fdt)

    def predict(self, X):
        X_f = self.fuzzifier.transform(X)
        votes = np.zeros((X.shape[0], len(self.classes_)))
        for i, tree in enumerate(self.trees):
            X_sub = X_f.iloc[:, self.features_per_tree[i]]
            p = tree.predict_proba_soft(X_sub) if self.voting == 'SOFT_ROUTING' else tree.predict_proba(X_sub)
            if self.voting == 'SMI': w = 1.0
            elif self.voting == 'MWLT': w = self.tree_weights[i]
            elif self.voting == 'MWLFUS':
                knn, err = tree.error_tree
                _, ids = knn.kneighbors(X_sub)
                w = (1.0 - err[ids].mean(axis=1))[:, np.newaxis]
            else: w = 1.0
            votes += p * w
        return self.classes_[np.argmax(votes, axis=1)]

if __name__ == "__main__":
    print("\n[1] Carregando dados...")
    df = pd.read_csv('dataset.csv', parse_dates=['datetime'])
    features = ['SMA_3', 'EMA_3', 'SMA_5', 'EMA_5', 'std_close3', 'std_open3', 'ADXR', 'Bollinger_Norm']
    
    X_train = df[(df['datetime'] < '2024-04-01')][features]
    y_train = df[(df['datetime'] < '2024-04-01')]['trend']
    X_test = df[(df['datetime'] >= '2024-04-01')][features]
    y_test = df[(df['datetime'] >= '2024-04-01')]['trend']

    BOUNDS = {
        'n_estimators': (50, 300),       
        'max_depth': (3, 30),            
        'min_samples_split': (2, 20),    
        'min_samples_leaf': (1, 15),     
        'max_features': (0.1, 1.0)       
    }

    print("[2] Configurando Algoritmo Genético...")
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_n_est", random.randint, BOUNDS['n_estimators'][0], BOUNDS['n_estimators'][1])
    toolbox.register("attr_depth", random.randint, BOUNDS['max_depth'][0], BOUNDS['max_depth'][1])
    toolbox.register("attr_split", random.randint, BOUNDS['min_samples_split'][0], BOUNDS['min_samples_split'][1])
    toolbox.register("attr_leaf", random.randint, BOUNDS['min_samples_leaf'][0], BOUNDS['min_samples_leaf'][1])
    toolbox.register("attr_feat", random.uniform, BOUNDS['max_features'][0], BOUNDS['max_features'][1])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_n_est, toolbox.attr_depth, toolbox.attr_split,
                      toolbox.attr_leaf, toolbox.attr_feat), n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_model(individual):
        n_est = int(individual[0])
        depth = int(individual[1])
        split = int(individual[2])
        leaf  = int(individual[3])
        feat  = float(individual[4])

        model = FuzzyRandomForest(
            n_estimators=n_est, 
            mf_type='gaussian', 
            n_partitions=5,     
            voting='MWLFUS',    
            max_depth=depth, 
            min_samples_split=split, 
            min_samples_leaf=leaf, 
            max_features=feat
        )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds),

    toolbox.register("evaluate", eval_model)
    toolbox.register("mate", tools.cxTwoPoint) # Crossover: TwoPoint
    toolbox.register("select", tools.selTournament, tournsize=3) # Seleção: Tournament

    def mutate_bounds(individual, indpb):
        if random.random() < indpb: individual[0] = random.randint(BOUNDS['n_estimators'][0], BOUNDS['n_estimators'][1])
        if random.random() < indpb: individual[1] = random.randint(BOUNDS['max_depth'][0], BOUNDS['max_depth'][1])
        if random.random() < indpb: individual[2] = random.randint(BOUNDS['min_samples_split'][0], BOUNDS['min_samples_split'][1])
        if random.random() < indpb: individual[3] = random.randint(BOUNDS['min_samples_leaf'][0], BOUNDS['min_samples_leaf'][1])
        if random.random() < indpb: individual[4] = random.uniform(BOUNDS['max_features'][0], BOUNDS['max_features'][1])
        return individual,

    toolbox.register("mutate", mutate_bounds, indpb=0.2)

    POP_SIZE = 10    # (Altere para 20 no teste final, mas custará muito processamento)
    NGEN = 10        # (Altere para 3000 no teste final, como na imagem)
    CXPB = 0.8       # cxpb: 0.8
    MUTPB = 0.05     # mutpb: 0.05

    print(f"\n[3] Iniciando Evolução... População: {POP_SIZE} | Gerações: {NGEN}")
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1) 
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, 
                                       ngen=NGEN, stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    print("\n" + "="*70)
    print("OTIMIZAÇÃO CONCLUÍDA! MELHORES HIPERPARÂMETROS:")
    print("="*70)
    print(f" - n_estimators: {int(best_ind[0])}")
    print(f" - max_depth: {int(best_ind[1])}")
    print(f" - min_samples_split: {int(best_ind[2])}")
    print(f" - min_samples_leaf: {int(best_ind[3])}")
    print(f" - max_features: {float(best_ind[4]):.4f} ({int(best_ind[4]*100)}% das colunas)")
    
    print(f"\nAcurácia do Indivíduo Campeão: {best_ind.fitness.values[0]:.4f}")