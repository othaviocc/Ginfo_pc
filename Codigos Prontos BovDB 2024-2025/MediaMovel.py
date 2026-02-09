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
    def identify_60_min_candles(self):
        # Reamostragem para manter os candles de 60 minutos
        df_60min = self.df.resample('60T').agg({
            'open': 'first',
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }).dropna()
        return df_60min 
                   
    def calculate_bollinger_bands(self, df, period=7, std_fac=0.7929549):
        df['SMA'] = df['close'].rolling(period).mean()    #Media Movel Simples --> Middle Band
        df['STD'] = df['close'].rolling(period).std()     #Desvio  Padrão
        df['Upper Band'] = df['SMA'] + (df['STD'] * std_fac)   #Upper Band
        df['Lower Band'] = df['SMA'] - (df['STD'] * std_fac)   #Lower Band
        return df
    
    def sma(self,df):
        df['SMA3'] = df['close'].rolling(3).mean() 
        df['SMA5'] = df['close'].rolling(5).mean() 
        df['SMA7'] = df['close'].rolling(7).mean() 
        df['SMA9'] = df['close'].rolling(9).mean() 
        df['SMA11'] = df['close'].rolling(11).mean() 
        return df
    
# Função para plotar
def plot_sma(df):
    fig = go.Figure()

    # Plotar os candles com Medias Moveis
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA3'],
        line=dict(color='orange', width=1),
        name='SMA_3'
    ))
    '''fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA5'],
        line=dict(color='purple', width=1),
        name='SMA_5'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA7'],
        line=dict(color='gray', width=1),
        name='SMA_7'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA9'],
        line=dict(color='yellow', width=1),
        name='SMA_9'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA11'],
        line=dict(color='blue', width=1),
        name='SMA_11'
    ))'''

    # Configurar layout do gráfico
    fig.update_layout(
        title='Candlestick com Medias Moveis',
        xaxis_title='Data',
        yaxis_title='Preço',
        xaxis_rangeslider_visible=False
    )

    fig.show()

# Exemplo de uso
db_path = r'C:\\Users\\othav\\BovDB\\Database_define.db' 
query =  """
    SELECT id_ticker, date, time, open, close, high, low, average, volume, business, amount_stock
    FROM price5
    WHERE id_ticker = 3193 AND date BETWEEN '2024-06-26' AND '2024-06-27'
    """

processor = DataProcessor(db_path, query)
processor.load_data()
df = processor.process_data()

df_5min = processor.identify_5_min_candles()
df_5min = processor.sma(df_5min)

df_5min = df_5min[df_5min.index.date == pd.to_datetime('2024-06-27').date()]

plot_sma(df_5min)


