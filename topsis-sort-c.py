# import csv # Para a leitura dos dados
# Importa a biblioteca NumPy, que é utilizada para realizar operações matriciais eficientes em Python.
import numpy as np

# STEP 2
def calcular_perfis_centrais(matriz_decisao):
    matriz_decisao = np.array(matriz_decisao)

    # Calculando a média das critérios para obter os perfis centrais 
    perfis_centrais = np.mean(matriz_decisao, axis=0)
    # Ao calcular a média ao longo das colunas, nos será fornecida uma representação das características médias de cada critério em relação aos perfis de investimento.

    #retorna a matriz de perfis centrais
    return perfis_centrais


# STEP 3
def determinar_dominio_dos_criterios(matriz_decisao):
    matriz_decisao = np.array(matriz_decisao)

    # Determinando os maiores valores possíveis para cada critério (solução ideal)
    maiores_valores = np.max(matriz_decisao, axis=0)
    # Os maiores valores representam as soluções ideais em cada critério, ou seja, o melhor valor que uma alternativa poderia alcançar em cada critério.

    # Determinando os menores valores possíveis para cada critério (solução não ideal)
    menores_valores = np.min(matriz_decisao, axis=0)
    # Os menores valores representam as soluções não ideais em cada critério, ou seja, o pior valor que uma alternativa poderia alcançar em cada critério.

    # Criando a matriz de domínio com os maiores e menores valores de cada critério
    matriz_dominio = np.vstack((maiores_valores, menores_valores))
    # A matriz de domínio é composta pelos maiores e menores valores de cada critério, onde a primeira linha representa os maiores valores (solução ideal) e a segunda linha representa os menores valores (solução não ideal).

    return matriz_dominio

# STEP 4 
def criar_matriz_decisao_completa(matriz_decisao, dominio_dos_criterios, perfis_centrais):
    # Concatenando a matriz de decisão com a matriz de domínio
    matriz_decisao_completa = np.concatenate((matriz_decisao, dominio_dos_criterios, perfis_centrais.reshape(1, -1)), axis=0)

    # A concatenação dessas duas matrizes resulta na matriz de decisão completa, onde cada elemento da matriz contém tanto o valor real do critério quanto seus limites de variação(mínimo e máximo). Esta matriz é usada posteriormente no cálculo das distâncias entre as alternativas e as soluções ideal e anti-ideal no método TOPSIS.
    return matriz_decisao_completa

# STEP 5
# STEP 5.1: Normalizar a Matriz de Decisão Completa
# Isso é necessário porque os critérios podem ter unidades diferentes, escalas diferentes ou variações de valores muito distintas, o que pode distorcer a análise se não forem tratados adequadamente.A normalização coloca todos os critérios em uma escala comum para que possam ser comparáveis de forma justa.
def normalizar_matriz_decisao(matriz_decisao_completa, metodo_normalizacao='max'):
    # Definindo o método de normalização
    if metodo_normalizacao == 'max':
        # Normalização pelo máximo
        matriz_normalizada = matriz_decisao_completa / np.max(matriz_decisao_completa, axis=1, keepdims=True)
    elif metodo_normalizacao == 'interval':
        # Normalização por intervalo
        minimo = matriz_decisao_completa[-1]
        maximo = matriz_decisao_completa[-2]
        matriz_normalizada = (matriz_decisao_completa - minimo) / (maximo - minimo)
    else:
        raise ValueError("Método de normalização inválido. Use 'max' ou 'interval'.")

    #  Retorna matriz de decisão completa normalizada.
    return matriz_normalizada

# STEP 5.2: Calcular a Matriz de Decisão Ponderada e Normalizada
def calcular_matriz_ponderada_normalizada(matriz_normalizada, pesos):
    # Multiplicando cada valor normalizado pelo peso correspondente do critério
    # Utiliza-se o slicing para ignorar as duas últimas linhas da matriz, que representam os limites de variação (máximo e mínimo) e não os valores reais.
    # O slicing [:, None] é usado para transformar o array de pesos em uma matriz de uma única coluna, de modo que possa ser multiplicado diretamente com a matriz normalizada.
    matriz_ponderada_normalizada = matriz_normalizada[:-2] * pesos[:, None]

    # Retorna a matriz de decisão ponderada e normalizada
    return matriz_ponderada_normalizada

# STEP 6: Determine as soluções ideais e anti-ideais
def determinar_solucoes_ideais(matriz_ponderada_normalizada):
    # Determinando as soluções ideais
    solucao_ideal = np.max(matriz_ponderada_normalizada, axis=1)
    # A solução ideal é definida como o perfil de investimento que possui o melhor desempenho em todos os critérios. Para encontrá-la, identificamos o maior valor para cada critério entre todas as alternativas. Esses valores representam o perfil de investimento que maximiza os benefícios em cada critério.

    # Determinando as soluções anti-ideais
    solucao_anti_ideal = np.min(matriz_ponderada_normalizada, axis=1)
    # A solução anti-ideal é o oposto da solução ideal, representando o perfil de investimento que possui o pior desempenho em todos os critérios. Para encontrá-la, identificamos o menor valor para cada critério entre todas as alternativas. Esses valores representam o perfil de investimento que minimiza os benefícios em cada critério.

    return solucao_ideal, solucao_anti_ideal

# STEP 7: Calcular as distâncias euclidianas de cada alternativa e perfil para as soluções ideais e anti-ideais
def calcular_distancias_euclidianas(matriz_ponderada_normalizada, solucao_ideal, solucao_anti_ideal):
    """
    Calcula as distâncias euclidianas de cada alternativa e perfil para as soluções ideais e anti-ideais.

    Retorna:
    - distancias_ideal: As distâncias euclidianas de cada alternativa e perfil para a solução ideal.
    - distancias_anti_ideal: As distâncias euclidianas de cada alternativa e perfil para a solução anti-ideal.
    """

    # Calcular as distâncias euclidianas para a solução ideal
    distancias_ideal = np.sqrt(np.sum((matriz_ponderada_normalizada - solucao_ideal) ** 2, axis=1))
    # A distância euclidiana de cada alternativa para a solução ideal é calculada pela raiz quadrada da soma dos quadrados das diferenças entre os valores da alternativa e os valores ideais em cada critério.

    # Calcular as distâncias euclidianas para a solução anti-ideal
    distancias_anti_ideal = np.sqrt(np.sum((matriz_ponderada_normalizada - solucao_anti_ideal) ** 2, axis=1))
    # A distância euclidiana de cada alternativa para a solução anti-ideal é calculada da mesma forma, mas usando os valores anti-ideais em vez dos valores ideais.

    return distancias_ideal, distancias_anti_ideal


# STEP 8: Calcular o coeficiente de proximidade de cada alternativa para a solução ideal
def calcular_coeficiente_proximidade(distancias_ideal, distancias_anti_ideal):
    """
    Calcula o coeficiente de proximidade de cada alternativa para a solução ideal.

    Parâmetros:
    - distancias_ideal: As distâncias euclidianas de cada alternativa para a solução ideal.
    - distancias_anti_ideal: As distâncias euclidianas de cada alternativa para a solução anti-ideal.

    Retorna:
    - coeficiente_proximidade: O coeficiente de proximidade de cada alternativa para a solução ideal.
    """

    # Calcular o coeficiente de proximidade para cada alternativa
    coeficiente_proximidade = distancias_anti_ideal / (distancias_ideal + distancias_anti_ideal)
    # O coeficiente de proximidade é calculado como a razão entre a distância para a solução anti-ideal e a soma das distâncias para a solução ideal e a solução anti-ideal.

    return coeficiente_proximidade

# STEP 9: Classificar as alternativas fazendo comparações entre seus coeficientes de proximidade e os dos perfis
def classificar_alternativas(closeness_coefficients, closeness_coefficients_perfil):
    """
    Classifica as alternativas fazendo comparações entre seus coeficientes de proximidade e os dos perfis.

    Parâmetros:
    - closeness_coefficients: Os coeficientes de proximidade das alternativas em relação à solução ideal.
    - closeness_coefficients_perfil: Os coeficientes de proximidade dos perfis.

    Retorna:
    - classificacao: Um array de classificação das alternativas, indicando sua posição relativa em relação aos perfis.
    """

    # Comparar os coeficientes de proximidade das alternativas com os dos perfis
    # Se o coeficiente de proximidade da alternativa for maior ou igual ao do perfil, a alternativa é considerada adequada para o perfil.
    classificacao = np.where(closeness_coefficients >= closeness_coefficients_perfil, 'Adequado', 'Inadequado')

    return classificacao

# Função principal que executa todas as etapas do algoritmo TOPSIS
def topsis(matriz_decisao, pesos, metodo_normalizacao='max'):
    # STEP 2
    perfis_centrais = calcular_perfis_centrais(matriz_decisao)
    print("Perfis centrais:", perfis_centrais)

    # STEP 3
    dominio_dos_criterios = determinar_dominio_dos_criterios(matriz_decisao)
    print("Domínio dos critérios:", dominio_dos_criterios)
    
    # STEP 4
    matriz_decisao_completa = criar_matriz_decisao_completa(matriz_decisao, dominio_dos_criterios, perfis_centrais)
    print("Matriz de decisão completa:", matriz_decisao_completa)

    # STEP 5
    matriz_normalizada = normalizar_matriz_decisao(matriz_decisao_completa, metodo_normalizacao)
    print("Matriz normalizada:", matriz_normalizada)

    # STEP 5.2
    matriz_ponderada_normalizada = calcular_matriz_ponderada_normalizada(matriz_normalizada, pesos)
    print("Matriz ponderada normalizada:", matriz_ponderada_normalizada)

    # STEP 6
    solucao_ideal, solucao_anti_ideal = determinar_solucoes_ideais(matriz_ponderada_normalizada)
    print("Solução ideal:", solucao_ideal)
    print("Solução anti-ideal:", solucao_anti_ideal)

    # STEP 7
    distancias_ideal, distancias_anti_ideal = calcular_distancias_euclidianas(matriz_ponderada_normalizada, solucao_ideal, solucao_anti_ideal)
    print("Distâncias ideais:", distancias_ideal)
    print("Distâncias anti-ideais:", distancias_anti_ideal)

    # STEP 8
    coeficiente_proximidade = calcular_coeficiente_proximidade(distancias_ideal, distancias_anti_ideal)
    print("Coeficiente de proximidade:", coeficiente_proximidade)

    # STEP 9
    classificacao = classificar_alternativas(coeficiente_proximidade, perfis_centrais)

    print("Classificação:", classificacao)
    return classificacao

###############################################################################

# Exemplo de uso
# STEP 1
# Matriz de decisão, onde cada linha representa um perfil de investimento (Conservador, Moderado, Agressivo), e cada coluna representa um critério (Retorno, Risco, Liquidez, Crescimento). Os valores dentro da matriz são as características não normalizadas de cada perfil de investimento.
perfis_investimento = [[8, 3, 7, 5],
                       [7, 5, 6, 6],
                       [9, 7, 4, 8]]


# Base de dados esperada em csv
# data_csv = []
# with open('./data.csv', 'r') as arquivo_csv: # Abrir o arquivo CSV e ler os valores relacionados aos critérios de cada alternativa
#     csv_reader = csv.reader(arquivo_csv)
#     for linha in csv_reader:
#         data_linha = [float(valor) for valor in linha]
#         data_csv.append(data_linha)

# matriz de decisão incial
# matriz_numpy = np.array(data_csv) # Transformar o csv em uma matriz NumPy para facilitar manipulação de dados

# -Pseudocódigo
# Receber número de alternativas
# Receber número de critérios
# Receber o vetor de pesos
# Receber o método de normalização
# Chamar o TOPSIS

####

# # Exemplo
# decision_matrix = np.array([[5, 2, 10],
#                             [3, 8, 5],
#                             [9, 4, 1]])

# # Matriz de perfis de fronteira
# boundary_profiles = np.array([[7, 5, 3]])

# # Vetor de pesos
# weights = np.array([0.3, 0.5, 0.2])

# # Método de normalização
# normalization_method = "max"

# # Execução do algoritmo
# assignments = topsis_sort_c(decision_matrix, boundary_profiles, weights, normalization_method)

# # Impressão dos resultados
# print(assignments)

# # Saída - [1, 2, 1]