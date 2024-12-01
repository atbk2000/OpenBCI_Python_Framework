import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
import argparse

parser = argparse.ArgumentParser(description="Script para plotar o sinal do openBCI")
parser.add_argument('arquivo', type=str, help='Nome do arquivo a ser lido')
args = parser.parse_args()

df = pd.read_csv(args.arquivo)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
    
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def apply_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def apply_notch_filter(data, fs, f0=60.0, Q=30.0):
    b, a = iirnotch(f0, Q, fs)
    y = filtfilt(b, a, data)
    return y

fs = 250
low_cutoff = 100
high_cutoff = 20
order = 5

df['Fp1_highpass'] = apply_highpass_filter(df['Fp1'], high_cutoff, fs, order)
df['Fp1_lowpass'] = apply_lowpass_filter(df['Fp1'], low_cutoff, fs, order)
df['Fp1_filtered'] = apply_notch_filter(df['Fp1_lowpass'], fs)

df['C3_highpass'] = apply_highpass_filter(df['C3'], high_cutoff, fs, order)
df['C3_lowpass'] = apply_lowpass_filter(df['C3'], low_cutoff, fs, order)
df['C3_filtered'] = apply_notch_filter(df['C3_lowpass'], fs)

df['marker_normalizado'] = (df['marker'] - df['marker'].min()) / (df['marker'].max() - df['marker'].min())

def plot(df, arg1, arg2):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(df['time_board'], df[arg1], label='Fp1', color='blue')
    ax1.plot(df['time_board'], df[arg2], label='C3', color='green')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude dos Sinais', color='black')
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()

    ax2.step(df['time_board'], df['marker_normalizado'], label='Marker', color='red', where='post')

    ax2.set_ylabel('Marker (Sinal Quadrado)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax2.legend(loc='upper right')

    plt.title('Sinais Fp1, C3 e Marker (com eixo secundário)')
    plt.tight_layout()
    plt.show()
    
plot(df, 'Fp1', 'C3')

# signal_fp1 = df['Fp1_filtered'].values
# signal_c3 = df['C3_filtered'].values
# sampling_rate = 250  # Vou assumir uma taxa de amostragem de 1000 Hz para calcular a FFT

# # Calculando a FFT dos sinais
# fft_fp1 = np.fft.fft(signal_fp1)
# fft_c3 = np.fft.fft(signal_c3)
# freqs = np.fft.fftfreq(len(signal_fp1), 1/sampling_rate)

# # Plotar o espectro de frequência dos dois sinais
# plt.figure(figsize=(12,6))
# plt.subplot(2, 1, 1)
# plt.plot(freqs[:len(freqs)//2], np.abs(fft_fp1[:len(fft_fp1)//2]), label='Fp1', color='blue')
# plt.title('Espectro de Frequência do Sinal Fp1')
# plt.xlabel('Frequência (Hz)')
# plt.ylabel('Amplitude')
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(freqs[:len(freqs)//2], np.abs(fft_c3[:len(fft_c3)//2]), label='C3', color='green')
# plt.title('Espectro de Frequência do Sinal C3')
# plt.xlabel('Frequência (Hz)')
# plt.ylabel('Amplitude')
# plt.grid(True)

# plt.tight_layout()
# plt.show()