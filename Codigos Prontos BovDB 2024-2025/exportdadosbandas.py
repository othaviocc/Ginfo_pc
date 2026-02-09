import sqlite3
import pandas as pd
import plotly.graph_objects as go

class DataProcessor:
    def __init__(self, db_path, query):
        self.db_path = db_path
        self.query = query
        self.df = None

    def load_data(self):
        # Conectar ao banco de dados e carregar os dados
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql_query(self.query, conn)
        conn.close()

    def process_data(self):
        # Criar uma nova coluna de datetime combinando data e hora
        self.df['datetime'] = pd.to_datetime(self.df['date'] + ' ' + self.df['time'], format='%Y-%m-%d %H:%M:%S')
        # Configurar datetime como índice
        self.df.set_index('datetime', inplace=True)
        # Filtrar os dados para incluir apenas a partir das 09:00:00
        self.df = self.df[self.df.index.time >= pd.to_datetime('09:00:00').time()]
        return self.df

    def identify_5_min_candles(self):
        # Reamostragem para manter os candles de 5 minutos
        df_5min = self.df.resample('5T').agg({
            'open': 'first',
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }).dropna()
        return df_5min

    def calculate_bollinger_bands(self, df, period=7, std_fac=0.7929549):
        df['SMA'] = df['close'].rolling(period).mean()    # Média Móvel Simples --> Middle Band
        df['STD'] = df['close'].rolling(period).std()     # Desvio Padrão
        df['Upper Band'] = df['SMA'] + (df['STD'] * std_fac)   # Upper Band
        df['Lower Band'] = df['SMA'] - (df['STD'] * std_fac)   # Lower Band

        return df

    def export_to_csv(self, df, file_path):
        columns = ['Upper Band', 'Lower Band', 'close', 'open', 'high', 'low', 'volume']
        df_to_export = df[columns]
        df_to_export.to_csv(file_path, index_label='datetime')

db_path = r'C:\\Users\\othav\\BovDB.v2\\Database_define.db'
query =   """
    SELECT id_ticker, date, time, open, close, high, low, average, volume, business, amount_stock
FROM price5
WHERE 
    (id_ticker = 58413 AND date BETWEEN '2024-01-01' AND '2024-01-31')
    OR 
    (id_ticker = 2952 AND date BETWEEN '2024-02-01' AND '2024-03-31')
    OR
    (id_ticker = 2963 AND date BETWEEN '2024-04-01' AND '2024-04-30')
    OR 
    (id_ticker = 2978 AND date BETWEEN '2024-05-01' AND '2024-06-30');
"""

processor = DataProcessor(db_path, query)
processor.load_data()
df = processor.process_data()

df_5min = processor.identify_5_min_candles()
df_5min = processor.calculate_bollinger_bands(df_5min)

# Exportar os dados para um arquivo CSV
output_file_path = r'C:\\Users\\othav\\BovDB.v2\\candles_export.csv'
processor.export_to_csv(df_5min, output_file_path)

print(f"Arquivo exportado com sucesso para: {output_file_path}")
