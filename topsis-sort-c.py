import numpy as np
import pandas as pd 

def normalize(matrix, method='max'):
    """Normalizes the decision matrix.

    Args:
        matrix (numpy.ndarray): The decision matrix.
        method (str, optional): Normalization method ('max' or 'interval'). 
                                Defaults to 'max'.

    Returns:
        numpy.ndarray: The normalized decision matrix.
    """

    if method == 'max':
        return matrix / matrix.max(axis=0)
    elif method == 'interval':
        max_values = matrix.max(axis=0)
        min_values = matrix.min(axis=0)
        return (matrix - min_values) / (max_values - min_values)
    else:
        raise ValueError("Invalid normalization method.")

def euclidean_distance(v1, v2):
    """Calculates the Euclidean distance between two vectors.

    Args:
        v1 (numpy.ndarray): Vector 1.
        v2 (numpy.ndarray): Vector 2.

    Returns:
        float: The Euclidean distance.
    """
    return np.sqrt(np.sum((v1 - v2)**2))

def topsis_sort_c(decision_matrix, boundary_profiles, weights, normalization_method='max'):
    """Implements the TOPSIS-Sort-C algorithm.

    Args:
        decision_matrix (numpy.ndarray): The decision matrix (alternatives x criteria).
        boundary_profiles (numpy.ndarray):  Matrix of characteristic profiles.
        weights (numpy.ndarray):  Array of criteria weights.
        normalization_method (str, optional): Normalization method. Defaults to 'max'.

    Returns:
        list: A list of class assignments for each alternative.
    """

    # Create the complete decision matrix
    domain_matrix = np.vstack([decision_matrix.max(axis=0), decision_matrix.min(axis=0)])
    complete_matrix = np.vstack([decision_matrix, boundary_profiles, domain_matrix])

    # Normalize, apply weights
    normalized_matrix = normalize(complete_matrix, method=normalization_method)
    weighted_matrix = normalized_matrix * weights

    # Calculate ideal and anti-ideal solutions
    ideal_solution = weighted_matrix.max(axis=0)
    anti_ideal_solution = weighted_matrix.min(axis=0)

    # Calculate distances and closeness coefficients
    distances_to_ideal = [euclidean_distance(weighted_matrix[i], ideal_solution) for i in range(decision_matrix.shape[0])]
    distances_to_anti_ideal = [euclidean_distance(weighted_matrix[i], anti_ideal_solution) for i in range(decision_matrix.shape[0])]
    distances_to_profiles = [euclidean_distance(weighted_matrix[i], weighted_matrix[decision_matrix.shape[0] + k]) 
                             for i in range(decision_matrix.shape[0]) for k in range(boundary_profiles.shape[0])]

    closeness_coefficients = [d_ai / (d_ai + d_plus) for d_ai, d_plus in zip(distances_to_anti_ideal, distances_to_ideal)]
    profile_closeness = [d_pk / (d_pk + d_minus) for d_pk, d_minus in zip(*[iter(distances_to_profiles)]*2)]

    # Assign alternatives to classes 
    assignments = []  
    for i in range(decision_matrix.shape[0]):
        profile_diffs = [abs(closeness_coefficients[i] - profile_closeness[k]) 
                         for k in range(0, len(profile_closeness), decision_matrix.shape[0])]
        assignments.append(profile_diffs.index(min(profile_diffs)) + 1) 

    return assignments


# Sample decision matrix
decision_matrix = np.array([[5, 2, 10],
                            [3, 8, 5],
                            [9, 4, 1]]) 
 
# Characteristic profiles
boundary_profiles = np.array([[7, 5, 3]]) 