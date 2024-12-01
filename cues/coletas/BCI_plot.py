import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Script para plotar o sinal do openBCI")
parser.add_argument('arquivo', type=str, help='Nome do arquivo a ser lido')
args = parser.parse_args()



df = pd.read_csv(args.arquivo)

df['marker_normalizado'] = (df['marker'] - df['marker'].min()) / (df['marker'].max() - df['marker'].min())

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(df['time_board'], df['Fp1'], label='Fp1', color='blue')
ax1.plot(df['time_board'], df['C3'], label='C3', color='green')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude dos Sinais', color='black')
ax1.legend(loc='upper left')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()

ax2.step(df['time_board'], df['marker_normalizado'], label='Marker', color='red', where='post')

ax2.set_ylabel('Marker', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax2.legend(loc='upper right')

plt.title('Sinais Fp1, C3 e Marker (com eixo secundário)')
plt.tight_layout()
plt.show()
