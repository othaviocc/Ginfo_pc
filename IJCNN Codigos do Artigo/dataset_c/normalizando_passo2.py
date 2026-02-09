import pandas as pd
import numpy as np

class TradingNormalizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded from {self.file_path} successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def normalize_sma_ema_std(self):
        try:
            sma_cols = [f"SMA_{w}" for w in [3, 5, 7, 9, 11]]
            ema_cols = [f"EMA_{w}" for w in [3, 5, 7, 9, 11]]
            std_cols = [f"std_close{w}" for w in [3, 5, 7, 9, 11]] + \
                       [f"std_open{w}" for w in [3, 5, 7, 9, 11]]
            all_cols = sma_cols + ema_cols + std_cols

            for col in all_cols:
                if col in self.data.columns:
                    col_min = self.data[col].min()
                    col_max = self.data[col].max()
                    self.data[col] = (self.data[col] - col_min) / (col_max - col_min)

            print("SMA, EMA and std normalized successfully.")
        except Exception as e:
            print(f"Error normalizing SMA/EMA/std: {e}")

    def normalize_bollinger(self):
        try:
            if all(c in self.data.columns for c in ["Bollinger_Lower", "Bollinger_Upper"]):
                prev_close = self.data["close"].shift(1)
                self.data["Bollinger_Norm"] = (
                    (prev_close - self.data["Bollinger_Lower"]) /
                    (self.data["Bollinger_Upper"] - self.data["Bollinger_Lower"])
                )
                # remove as colunas antigas
                self.data.drop(columns=["Bollinger_Mid", "Bollinger_Upper", "Bollinger_Lower"], inplace=True)
                print("Bollinger Bands normalized into a single column successfully.")
        except Exception as e:
            print(f"Error normalizing Bollinger Bands: {e}")

    def normalize_adxr(self):
        try:
            if "ADXR" in self.data.columns:
                self.data["ADXR"] = self.data["ADXR"] / 100
                print("ADXR normalized successfully.")
        except Exception as e:
            print(f"Error normalizing ADXR: {e}")

    def add_trend(self):
        try:
            n = len(self.data)
            if n == 0:
                raise ValueError("DataFrame vazio")

            # 1) Trend padrão: label de predição = próximo candle (x+1 > x)
            default_trend = (self.data["close"].shift(-1) > self.data["close"]).astype(int)
            # Se preferir marcar o último como NaN (porque não há próximo candle), descomente:
            # default_trend.iloc[-1] = np.nan
            self.data["trend"] = default_trend

            # 2) Calcula trend "atual" para toda a série (close[x] > close[x-1])
            current_trend = (self.data["close"] > self.data["close"].shift(1)).astype(int)

            # Helper que coloca block de 'current_trend' usando posições (iloc)
            def set_current_trend_block(start_pos, length=2500):
                if start_pos >= n:
                    # bloco começa fora do dataset — só avisa e sai
                    print(f"Bloco começando em {start_pos} está fora do dataset (len={n}), pulando.")
                    return
                end_pos = min(start_pos + length, n)  # end exclusivo
                col_idx = self.data.columns.get_loc("trend")
                # Usa .values para evitar alinhamentos por índice
                self.data.iloc[start_pos:end_pos, col_idx] = current_trend.iloc[start_pos:end_pos].values

            # Blocos solicitados: 4000..4799 e 13000..13799 (cada um com 800 posições)
            set_current_trend_block(4000, 4000)
            set_current_trend_block(13000, 3000)

            print("Trend column added successfully.")
        except Exception as e:
            print(f"Error adding trend: {e}")



    def save_data(self, output_file):
        try:
            self.data.dropna(inplace=True)
            self.data.to_csv(output_file, index=False)
            print(f"Data saved successfully to {output_file}")
        except Exception as e:
            print(f"Error saving data: {e}")


def process_file(input_file, output_file):
    normalizer = TradingNormalizer(input_file)
    normalizer.load_data()
    normalizer.normalize_sma_ema_std()
    normalizer.normalize_bollinger()
    normalizer.normalize_adxr()
    normalizer.add_trend()
    normalizer.save_data(output_file)


if __name__ == "__main__":
    input_file = r"C:\\Users\\othav\\BovDB.v2\\normalizados_passo1.csv"
    output_file = r"C:\\Users\\othav\\BovDB.v2\\normalizados_passo2.csv"

    process_file(input_file, output_file)
