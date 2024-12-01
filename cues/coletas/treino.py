import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

# Função para carregar e processar arquivos EMG
def carregar_dados(arquivos):
    dados = []
    for arquivo in arquivos:
        df = pd.read_csv(arquivo)
        df['arquivo'] = arquivo
        dados.append(df)
    return pd.concat(dados, ignore_index=True)

# Função para preparar dados para CSP com janelas fixas
def preparar_dados_para_csp(df, janela=500):
    # Selecionar apenas as colunas dos sensores e o marcador
    df = df[['Fp1', 'C3', 'marker']]
    X = []
    y = []
    
    # Percorrer cada segmento com base no marcador e extrair dados
    for marker in df['marker'].unique():
        segmento = df[df['marker'] == marker]
        dados_sensores = segmento[['Fp1', 'C3']].values.T
        
        # Dividir em janelas fixas
        num_janelas = dados_sensores.shape[1] // janela
        for i in range(num_janelas):
            janela_dados = dados_sensores[:, i * janela:(i + 1) * janela]
            if janela_dados.shape[1] == janela:
                X.append(janela_dados)
                y.append(marker)
    
    return np.array(X), np.array(y)

# Lista de arquivos de dados
arquivos = ['coleta1_italo.csv', 'coleta2_italo.csv', 'coleta3_italo.csv', 'coleta4_italo.csv', 'coleta5_italo.csv']

# Carregar os dados
df = carregar_dados(arquivos)

# Dividir os arquivos entre treino e teste
treino_arquivos = arquivos[:4]
teste_arquivo = arquivos[4]

df_treino = df[df['arquivo'].isin(treino_arquivos)]
df_teste = df[df['arquivo'] == teste_arquivo]

# Preparar os dados para CSP com janelas fixas de tamanho 500
X_train, y_train = preparar_dados_para_csp(df_treino, janela=500)
X_test, y_test = preparar_dados_para_csp(df_teste, janela=500)

# Inicializar o CSP e ajustar aos dados de treinamento
csp = CSP(n_components=2, reg=None, log=True)
X_train_csp = csp.fit_transform(X_train, y_train)
X_test_csp = csp.transform(X_test)

# Treinamento do modelo LDA
modelo_lda = LinearDiscriminantAnalysis()
modelo_lda.fit(X_train_csp, y_train)

# Avaliação do modelo
y_pred = modelo_lda.predict(X_test_csp)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
