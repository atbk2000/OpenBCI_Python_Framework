import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Script para plotar o sinal do openBCI")
parser.add_argument('arquivo', type=str, help='Nome do arquivo a ser lido')
args = parser.parse_args()

data = pd.read_csv(args.arquivo)

# Encontrar o índice da primeira alteração no 'marker'
first_change_index = data['marker'].ne(data['marker'].iloc[0]).idxmax()

# Cortar os dados até esse ponto
data_cortada = data.iloc[first_change_index:]

# Verificar o resultado
print(data_cortada.head())

# Se quiser salvar o novo dataset cortado:
data_cortada.to_csv(args.arquivo, index=False)
