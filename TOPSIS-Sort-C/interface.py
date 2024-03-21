import streamlit as stlit
from topsis_sort_c import *
import pandas as pd
import numpy as np


#titulos da pagina
stlit.title('Algoritmo Topsis Sort-C')
stlit.write('This is a simple interface for Streamlit')

# Criando barra lateral
sidebar = stlit.sidebar
# Adicionando opções pra barra lateral
expander = sidebar.expander("Options", expanded=False)
with expander:
    sidebar.write("Option 1")
    # Adicionando botão de upload de arquivo CSV
    uploaded_file1 = sidebar.file_uploader("Arquivo CSV para Matriz")
    uploaded_file2 = sidebar.file_uploader("Arquivo CSV para os Perfis")
    uploaded_file3 = sidebar.file_uploader("Arquivo CSV para os Pesos")

# Verificando se os arquivos foram carregados
if uploaded_file1 is not None and uploaded_file2 is not None and uploaded_file3 is not None:
    # Lendo os arquivos CSV e convertendo em dataframes
    df_matriz = pd.read_csv(uploaded_file1, header=None)
    df_perfis = pd.read_csv(uploaded_file2, header=None)
    df_pesos = pd.read_csv(uploaded_file3, header=None)
    

    matriz = df_matriz.values.tolist()
    perfis = [
    [10, 10, 9],  # Bom
    [8, 7, 9],  # Médio
    [7, 6, 6]  # Ruim
]
    pesos = np.array([9, 8, 8])

    topsis(matriz, perfis, pesos)

