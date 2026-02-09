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

    def calculate_bollinger_bands(self, df, period=20, std_fac=2):
        df['SMA'] = df['close'].rolling(period).mean()    #Media Movel Simples --> Middle Band
        df['STD'] = df['close'].rolling(period).std()     #Desvio  Padrão
        df['Upper Band'] = df['SMA'] + (df['STD'] * std_fac)   #Upper Band
        df['Lower Band'] = df['SMA'] - (df['STD'] * std_fac)   #Lower Band
        return df

# Função para plotar candles com Bandas de Bollinger
def plot_candlestick_with_bollinger(df):
    fig = go.Figure()

    # Plotar os candles
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles'
    ))

    # Adicionar SMA, Banda Superior e Banda Inferior
    '''fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA'],
        line=dict(color='blue', width=1),
        name='SMA'
    ))'''
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Upper Band'],
        line=dict(color='green', width=1),
        name='Upper Band'
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Lower Band'],
        line=dict(color='red', width=1),
        name='Lower Band'
    ))

    '''min_lower_band = df['Lower Band'].min()
    min_lower_index = df['Lower Band'].idxmin()
    fig.add_trace(go.Scatter(
        x=[min_lower_index],  # Ponto do mínimo da Lower Band
        y=[min_lower_band],
        marker=dict(color='purple', size=10),
        name='Min Lower Band',
    ))

    max_upper_band = df['Upper Band'].max()
    max_upper_index = df['Upper Band'].idxmax()
    fig.add_trace(go.Scatter(
        x=[max_upper_index],  # Ponto do máximo da Upper Band
        y=[max_upper_band],
        marker=dict(color='orange', size=10),
        name='Max Upper Band',
    ))
'''
    # Configurar layout do gráfico
    fig.update_layout(
        title='Candlestick com Bandas de Bollinger',
        xaxis_title='Data',
        yaxis_title='Preço',
        xaxis_rangeslider_visible=False
    )

    fig.show()

# Exemplo de uso
db_path = r'C:\\Users\\othav\\BovDB\\Database_define.db' 
query = """
    SELECT id_ticker, date, time, open, close, high, low, average, volume, business, amount_stock
    FROM price5
    WHERE id_ticker = 2963 AND date BETWEEN '2024-04-15' AND '2024-04-16';
    """
# Criar uma instância do DataProcessor
processor = DataProcessor(db_path, query)
processor.load_data()
df = processor.process_data()

# Identificar candles de 5 minutos e calcular Bandas de Bollinger
df_5min = processor.identify_5_min_candles()
df_5min = processor.calculate_bollinger_bands(df_5min)

# Filtrar apenas o segundo dia para o plot
df_5min = df_5min[df_5min.index.date == pd.to_datetime('2024-04-16').date()]

# Plotar o gráfico com Bandas de Bollinger
plot_candlestick_with_bollinger(df_5min)
