import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Função para carregar os dados
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Função para aplicar o filtro Butterworth (banda passante)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Função para filtrar o sinal
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

# Função de retificação do sinal (valor absoluto)
def rectify_signal(signal):
    return np.abs(signal)

# Função para média móvel
def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Função de normalização
def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)

# Funções para extração das características
def compute_mav(signal):
    return np.mean(np.abs(signal))

def compute_zc(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings)

def compute_ssc(signal):
    ssc_count = 0
    for i in range(1, len(signal)-1):
        if np.sign(signal[i] - signal[i-1]) != np.sign(signal[i+1] - signal[i]):
            ssc_count += 1
    return ssc_count

def compute_wl(signal):
    return np.sum(np.abs(np.diff(signal)))

def extract_features(signal, fs):
    mav = compute_mav(signal)
    zc = compute_zc(signal)
    ssc = compute_ssc(signal)
    wl = compute_wl(signal)
    energy = np.sum(signal**2) / len(signal)
    return [mav, zc, ssc, wl, energy]

# Função para processar e extrair características dos dados
def process_and_extract_features(data, fs, lowcut, highcut, window_size):
    signal_fp1 = bandpass_filter(data['Fp1'], lowcut, highcut, fs)
    signal_c3 = bandpass_filter(data['C3'], lowcut, highcut, fs)

    # Retificação
    signal_fp1_rectified = rectify_signal(signal_fp1)
    signal_c3_rectified = rectify_signal(signal_c3)

    # Média Móvel
    signal_fp1_smooth = moving_average(signal_fp1_rectified, window_size)
    signal_c3_smooth = moving_average(signal_c3_rectified, window_size)

    # Normalização
    signal_fp1_normalized = normalize_signal(signal_fp1_smooth)
    signal_c3_normalized = normalize_signal(signal_c3_smooth)

    features = []
    labels = []

    # Extrair características para cada janela de tempo
    for i in range(0, len(data), fs):
        window_fp1 = signal_fp1_normalized[i:i+fs]
        window_c3 = signal_c3_normalized[i:i+fs]

        if len(window_fp1) == fs and len(window_c3) == fs:
            combined_features = extract_features(np.concatenate([window_fp1, window_c3]), fs)
            features.append(combined_features)
            label = data['marker'].iloc[i]
            labels.append(label)

    return np.array(features), np.array(labels)

# Função para dividir os dados em treino e teste
def split_data(features, labels, test_size=0.2):
    return train_test_split(features, labels, test_size=test_size, random_state=42)

# Função para treinar o modelo
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Função para avaliar o modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Caminho dos arquivos CSV
file_path_antonio = 'cues/coletas/A_antonio.csv'
file_path_angelo = 'cues/coletas/A_angelo.csv'

# Definir parâmetros
fs = 250  # Frequência de amostragem
lowcut = 20.0  # Frequência de corte inferior
highcut = 120.0  # Frequência de corte superior (ajustado para 120 Hz)
window_size = 50  # Tamanho da janela para média móvel

# Carregar dados
data_antonio = load_data(file_path_antonio)
data_angelo = load_data(file_path_angelo)

# Processar e extrair características para ambos os datasets
features_antonio, labels_antonio = process_and_extract_features(data_antonio, fs, lowcut, highcut, window_size)
features_angelo, labels_angelo = process_and_extract_features(data_angelo, fs, lowcut, highcut, window_size)

# Combinação dos dados
features_combined = np.vstack([features_antonio, features_angelo])
labels_combined = np.concatenate([labels_antonio, labels_angelo])

# Dividir os dados combinados em treino e teste
X_train_combined, X_test_combined, y_train_combined, y_test_combined = split_data(features_combined, labels_combined)

# Treinar o modelo
model_combined = train_model(X_train_combined, y_train_combined)

# Avaliar o modelo nos dados de treino e teste combinados
print("Avaliação no conjunto de treino:")
evaluate_model(model_combined, X_train_combined, y_train_combined)
print("\nAvaliação no conjunto de teste:")
evaluate_model(model_combined, X_test_combined, y_test_combined)

# Avaliação com um novo conjunto de dados
file_path_mateus = 'cues/coletas/A_murilo.csv'  # Substitua pelo caminho do seu novo arquivo
data_mateus = load_data(file_path_mateus)

# Processar e extrair características para o novo conjunto de dados
features_mateus, labels_mateus = process_and_extract_features(data_mateus, fs, lowcut, highcut, window_size)

# Avaliar o modelo no novo conjunto de dados
print("\nAvaliação no novo conjunto de dados:")
evaluate_model(model_combined, features_mateus, labels_mateus)
