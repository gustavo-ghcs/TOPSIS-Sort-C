import numpy as np
import pandas as pd 

def normalize(matrix, method='max'):
    # Normaliza a matriz de acordo com o método especificado.
    if method == 'max':
        return matrix / matrix.max(axis=0)
    elif method == 'interval':
        max_values = matrix.max(axis=0)
        min_values = matrix.min(axis=0)
        return (matrix - min_values) / (max_values - min_values)
    else:
        raise ValueError("Método de normalização inválido. Use 'max' ou 'interval'.")

def euclidean_distance(v1, v2):
    # Calcula a distância euclidiana entre dois vetores.
    return np.sqrt(np.sum((v1 - v2)**2))

def topsis_sort_c(decision_matrix, boundary_profiles, weights, normalization_method='max'):
    # Cria uma matriz de decisão completa contendo a matriz original, seus perfis característicos e valores máximo e mínimo de cada coluna.
    domain_matrix = np.vstack([decision_matrix.max(axis=0), decision_matrix.min(axis=0)])
    complete_matrix = np.vstack([decision_matrix, boundary_profiles, domain_matrix])

    # Aplica o método de normalização especificado à matriz completa, e então multiplica cada valor pelos seus pesos correspondentes.
    normalized_matrix = normalize(complete_matrix, method=normalization_method)
    weighted_matrix = normalized_matrix * weights

    # Encontra as soluções ideais positivas e negativas da matriz de decisão, obtidas considerando-se os melhores e piores valores para cada critério.
    ideal_solution = weighted_matrix.max(axis=0)
    anti_ideal_solution = weighted_matrix.min(axis=0)

    # Calcula as distâncias de cada alternativa em relação às soluções ideais e negativas, e então calcula os coeficientes de proximidade para cada alternativa.
    distances_to_ideal = [euclidean_distance(weighted_matrix[i], ideal_solution) for i in range(decision_matrix.shape[0])]
    distances_to_anti_ideal = [euclidean_distance(weighted_matrix[i], anti_ideal_solution) for i in range(decision_matrix.shape[0])]
    distances_to_profiles = [euclidean_distance(weighted_matrix[i], weighted_matrix[decision_matrix.shape[0] + k]) 
                             for i in range(decision_matrix.shape[0]) for k in range(boundary_profiles.shape[0])]

    closeness_coefficients = [d_ai / (d_ai + d_plus) for d_ai, d_plus in zip(distances_to_anti_ideal, distances_to_ideal)]
    profile_closeness = [d_pk / (d_pk + d_minus) for d_pk, d_minus in zip(*[iter(distances_to_profiles)]*2)]

    # Classifica cada alternativa com base na sua proximidade em relação aos perfis característicos definidos, considerando as distâncias calculadas anteriormente.
    assignments = []  
    for i in range(decision_matrix.shape[0]):
        profile_diffs = [abs(closeness_coefficients[i] - profile_closeness[k]) 
                         for k in range(0, len(profile_closeness), decision_matrix.shape[0])]
        assignments.append(profile_diffs.index(min(profile_diffs)) + 1) 

    return assignments