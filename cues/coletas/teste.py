import os
import pandas as pd

# Diretório onde os arquivos CSV estão localizados
input_directory = "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas"

# Diretório onde os arquivos modificados serão salvos
output_directory_fp1_c3_time = "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\train"
output_directory_time_marker = "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\train"

# Criar os diretórios de saída se não existirem
os.makedirs(output_directory_fp1_c3_time, exist_ok=True)
os.makedirs(output_directory_time_marker, exist_ok=True)

# Função para processar os arquivos CSV
def process_csv_files(input_dir, output_dir_fp1_c3_time, output_dir_time_marker):
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv") and "_" in filename:
            input_file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(input_file_path)

            # Selecionar as colunas Fp1, C3 e time_board
            columns_fp1_c3_time = ['Fp1', 'C3', 'time_board']
            if all(col in df.columns for col in columns_fp1_c3_time):
                df_fp1_c3_time = df[columns_fp1_c3_time]
                output_file_path_fp1_c3_time = os.path.join(output_dir_fp1_c3_time, filename.replace(".csv", "_data.csv"))
                df_fp1_c3_time.to_csv(output_file_path_fp1_c3_time, index=False)

            # Selecionar as colunas time_board e marker
            columns_time_marker = ['time_board', 'marker']
            if all(col in df.columns for col in columns_time_marker):
                df_time_marker = df[columns_time_marker]
                output_file_path_time_marker = os.path.join(output_dir_time_marker, filename.replace(".csv", "_events.csv"))
                df_time_marker.to_csv(output_file_path_time_marker, index=False)

# Processar os arquivos CSV
process_csv_files(input_directory, output_directory_fp1_c3_time, output_directory_time_marker)

print("Processamento concluído!")