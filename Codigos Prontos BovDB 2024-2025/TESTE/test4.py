import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from skrebate import ReliefF
import numpy as np

df = pd.read_csv('data_training.csv', parse_dates=['datetime'])

train_start, train_end = '2024-01-01', '2024-03-30'
test_start, test_end = '2024-04-01', '2024-06-30'

treino = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
validacao = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

for sub_df in [treino, validacao]:
    sub_df['year'] = sub_df['datetime'].dt.year
    sub_df['month'] = sub_df['datetime'].dt.month
    sub_df['day'] = sub_df['datetime'].dt.day    
    sub_df['hour'] = sub_df['datetime'].dt.hour
    sub_df['minute'] = sub_df['datetime'].dt.minute
    sub_df.drop(columns=['datetime','date','close','open','low','high','volume','average','amount_stock','id_ticker','business'], inplace=True)

def remove_non_numeric(df):
    return df.select_dtypes(include=[np.number])

X_train = treino.drop(columns=['trend'])
y_train = treino['trend']
X_train = remove_non_numeric(X_train)

X_valid = validacao.drop(columns=['trend'])
y_valid = validacao['trend']
X_valid = remove_non_numeric(X_valid)

scaler = MinMaxScaler()
X_trains = scaler.fit_transform(X_train)






#================


info_gain = mutual_info_classif(X_trains, y_train, random_state= 42)
info_gain_series = pd.Series(info_gain, index=X_train.columns)
info_gain_sorted = info_gain_series.sort_values(ascending=False)

print("Top Information Gain:")
print(info_gain_sorted.head(10))

relief = ReliefF(n_neighbors=100, n_features_to_select=X_train.shape[1])
relief.fit(X_trains, y_train)
relief_scores = relief.feature_importances_
relief_series = pd.Series(relief_scores, index=X_train.columns)
relief_sorted = relief_series.sort_values(ascending=False)

print("Top ReliefF:")
print(relief_sorted.head(10))
