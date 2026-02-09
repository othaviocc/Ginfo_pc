import pandas as pd

def merge_and_process_data(features_file, labels_file, output_file):
    features_df = pd.read_csv(features_file, parse_dates=['datetime'])
    labels_df = pd.read_csv(labels_file, parse_dates=['datetime'])
    
    merged_df = pd.merge(features_df, labels_df, on=['datetime', 'id_ticker'], how='inner')
    
    merged_df['trend'] = merged_df['trend']  
    
    merged_df.dropna(inplace=True)
    
    merged_df.dropna(inplace=True)
    
    merged_df.to_csv(output_file, index=False)
    print(f"Arquivo salvo com sucesso: {output_file}")

features_file = "dados_indicadores.csv"  
labels_file = "output.csv"  
output_file = "data_training.csv"

merge_and_process_data(features_file, labels_file, output_file)