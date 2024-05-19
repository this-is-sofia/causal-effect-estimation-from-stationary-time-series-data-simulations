import random
from typing import List
from causy.sample_generator import (
    TimeseriesSampleGenerator as CausyTimeseriesSampleGenerator,
)
import torch
import numpy as np


class TimeseriesSampleGenerator(CausyTimeseriesSampleGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._variables = sorted(self._variables)

    def generate_coefficient_matrix(self):
        """
        generate the coefficient matrix for the sample generator graph
        :return:
        """

        matrix: List[List[float]] = [
            [0 for _ in self._variables] for _ in self._variables
        ]

        # map the initial values to numbers from 0 to n
        values_map = self.matrix_position_mapping()
        for i, k in enumerate(self._variables):
            values_map[k] = i

        for edge in self._edges:
            matrix[values_map[edge.to_node.node]][
                values_map[edge.from_node.node]
            ] = edge.value

        return matrix

    def matrix_position_mapping(self):
        values_map = {}
        for i, k in enumerate(self._variables):
            values_map[k] = i
        return values_map

    def vectorize_identity_block(self, n):
        # Create an empty tensor
        matrix = torch.zeros(n, n)

        # Fill the upper left block with an identity matrix
        matrix[: n // self._longest_lag, : n // self._longest_lag] = torch.eye(
            n // self._longest_lag
        )

        # Flatten the matrix
        vectorized_matrix = matrix.view(-1)

        return vectorized_matrix

    def compute_covariance_matrix(self) -> torch.Tensor:
        """
        Get covariance matrix for a given coefficient matrix assuming mutually independent standard normal noise terms
        :param coefficient_matrix:
        :return: For a 2x2 coefficient matrix this reads [[V(X_t),Cov(X_t,Y_t)],[Cov(X_t,Y_t),V(X_t)]]

        coefficient_matrix=[[a,0],[b,a]], i.e.
        coefficient_matrix[0][0] is the coefficient of X_t-1 in the equation for X_t, here a
        coefficient_matrix[0][1] is the coefficient of Y_t-1 in the equation for X_t (here: no edge, that means zero)
        coefficient_matrix[1][1] is the coefficient of Y_t-1 in the equation for Y_t, here a
        coefficient_matrix[1][0] is the coefficient of X_t-1 in the equation for Y_t, here b

        If the top left entry ([0][0]) in our coefficient matrix corresponds to X_{t-1} -> X_t, then the the top left
        entry in the covariance matrix is the variance of X_t, i.e. V(X_t).

        If the top right entry ([0][1]) in our coefficient matrix corresponds to X_{t-1} -> Y_t, then the the top left
        entry in the covariance matrix is the variance of X_t, i.e. V(X_t).
        """
        coefficient_matrix = self.generate_coefficient_matrix()
        kronecker_product = np.kron(coefficient_matrix, coefficient_matrix)
        n = len(coefficient_matrix)
        identity_matrix = np.identity(n**2)
        cov_matrix_noise_terms_vectorized = np.eye(n).flatten()
        vectorized_covariance_matrix = np.dot(
            np.linalg.pinv(identity_matrix - kronecker_product),
            cov_matrix_noise_terms_vectorized,
        )
        covariance_matrix = vectorized_covariance_matrix.reshape(n, n)
        return covariance_matrix
