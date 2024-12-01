import pandas as pd
import numpy as np

# Caminho para o arquivo CSV original
input_file_path = "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\A_angelo_converted.csv"

# Caminho para o novo arquivo CSV
output_file_path = "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\A_angelo_positive.csv"

# Ler o arquivo CSV original
df = pd.read_csv(input_file_path)

# Transformar todos os valores negativos em positivos
df = df.abs()

# Salvar os dados transformados em um novo arquivo CSV
df.to_csv(input_file_path, index=False)

print(f"Dados transformados e salvos em {input_file_path}")