import numpy as np
import torch


def compute_lagged_covariance_helper_matrix(
    coefficient_matrix, unlagged_covariance_matrix, lag
):
    """
    Compute the covariance matrix of the lagged variables
    :param coefficient_matrix: numpy array, matrix of coefficients of the VAR model
    :param unlagged_covariance_matrix: numpy array, covariance matrix of the unlagged variables X_t,Y_t
    :param lag: Integer, lag of the lagged variables, e.g. for X_{t-3}, Y_{t} it is 3.
    :return: numpy array, covariance matrix of the lagged variables, for a 2x2 coefficient matrix this reads [[V(X_t),Cov(X_{t-1},Y_t)],[Cov(X_t,Y_{t-1}),V(Y_{t-1})]]
    not symmetric because Cov(X_{t-3}, Y_{t}) != Cov(X_{t}, Y_{t-3})
    """
    coefficient_matrix_to_the_power_of_lag = np.linalg.matrix_power(
        coefficient_matrix, lag
    )
    n = len(coefficient_matrix)
    return np.dot(coefficient_matrix_to_the_power_of_lag, unlagged_covariance_matrix)


def conditional_covariance(cov_matrix, index_x1, indices_x2):
    """Calculate the conditional covariance matrix of X_1 given X_2
    :param cov_matrix: numpy array, Covariance matrix of X_1 and X_2
    :param index_x1: Integer, Index of X_1 in the covariance matrix
    :param indices_x2: List, Indices of X_2 in the covariance matrix
    :return: Conditional covariance matrix of X_1 given X_2
    """

    # Extract covariance matrices for X_1 and X_2
    cov_x1 = cov_matrix[[index_x1]][:, [index_x1]]
    cov_x2 = cov_matrix[indices_x2][:, indices_x2]

    # Extract cross-covariance matrix between X_1 and X_2
    cov_x1_x2 = cov_matrix[[index_x1]][:, indices_x2]
    cov_x2_x1 = cov_matrix[indices_x2][:, [index_x1]]

    # Calculate conditional covariance matrix using the corrected formula
    conditional_cov = cov_x1 - np.dot(cov_x1_x2, np.linalg.inv(cov_x2)).dot(cov_x2_x1)
    return conditional_cov
