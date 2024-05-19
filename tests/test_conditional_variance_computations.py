import unittest
import numpy as np

from analytic_conditional_variances import (
    compute_lagged_covariance_helper_matrix,
    conditional_covariance,
)
from models import MODEL_DAG_TWO_VARIABLES, MODEL_DAG_THREE_VARIABLES

def extract_variances_and_covariance_matrix(model):
    variables = list(model._variables)
    covariance_matrix = model.compute_covariance_matrix()

    variances = {}
    covariances = {}

    for h_i, h_var in enumerate(covariance_matrix):
        for v_i, v_var in enumerate(h_var):
            if h_i == v_i:
                variances[variables[h_i]] = v_var.item()
            else:
                if variables[h_i] not in covariances:
                    covariances[variables[h_i]] = {}
                covariances[variables[h_i]][variables[v_i]] = v_var.item()
    return variances, covariances


class TestConditionalVarianceComputations(unittest.TestCase):
    def test_compute_covariance_matrix(self):
        """
        Test the computation of the covariance matrix against ground truth
        :return:
        """
        model = MODEL_DAG_TWO_VARIABLES
        self.assertAlmostEqual(
            model.compute_covariance_matrix()[0][0],
            np.array([[5.26, 124.65], [124.65, 6602.43]])[0][0],
            1,
        )
        self.assertAlmostEqual(
            model.compute_covariance_matrix()[0][1],
            np.array([[5.26, 124.65], [124.65, 6602.43]])[0][1],
            1,
        )
        self.assertAlmostEqual(
            model.compute_covariance_matrix()[1][0],
            np.array([[5.26, 124.65], [124.65, 6602.43]])[1][0],
            1,
        )
        self.assertAlmostEqual(
            model.compute_covariance_matrix()[1][1],
            np.array([[5.26, 124.65], [124.65, 6602.43]])[1][1],
            0,
        )

        self.assertAlmostEqual(
            model.compute_covariance_matrix()[0][0],
            (1 / (1 - (0.9**2))),
            2,
        )

    def test_compute_lagged_covariance_helper_matrix(self):
        model = MODEL_DAG_TWO_VARIABLES
        variances, covariances = extract_variances_and_covariance_matrix(model)
        self.assertAlmostEqual(
            0.9 * variances["X"],
            compute_lagged_covariance_helper_matrix(
                model.generate_coefficient_matrix(),
                model.compute_covariance_matrix(),
                1,
            )[0][0],
            1,
        )
        self.assertAlmostEqual(
            (5 * variances["X"] + 0.9 * covariances["X"]["Y"])
            - (
                compute_lagged_covariance_helper_matrix(
                    model.generate_coefficient_matrix(),
                    model.compute_covariance_matrix(),
                    1,
                )[1][0]
            ),
            0,
            1,
        )

        self.assertAlmostEqual(
            (5 * covariances["X"]["Y"] + 0.9 * variances["Y"])
            - (
                compute_lagged_covariance_helper_matrix(
                    model.generate_coefficient_matrix(),
                    model.compute_covariance_matrix(),
                    1,
                )[1][1]
            ),
            0,
            1,
        )

        self.assertAlmostEqual(
            (0.9 * covariances["X"]["Y"])
            - (
                compute_lagged_covariance_helper_matrix(
                    model.generate_coefficient_matrix(),
                    model.compute_covariance_matrix(),
                    1,
                )[0][1]
            ),
            0,
            1,
        )

    def test_compute_lagged_covariance_helper_matrix_2(self):
        model = MODEL_DAG_THREE_VARIABLES
        variances, covariances = extract_variances_and_covariance_matrix(model)

        self.assertAlmostEqual(
            (0.9 * variances["X"])
            - (
                compute_lagged_covariance_helper_matrix(
                    model.generate_coefficient_matrix(),
                    model.compute_covariance_matrix(),
                    1,
                )[0][0]
            ),
            0,
            1,
        )

        self.assertAlmostEqual(
            (0.9 * covariances["X"]["Y"])
            - (
                compute_lagged_covariance_helper_matrix(
                    model.generate_coefficient_matrix(),
                    model.compute_covariance_matrix(),
                    1,
                )[0][1]
            ),
            0,
            1,
        )

        self.assertAlmostEqual(
            (0.9 * covariances["X"]["Z"])
            - (
                compute_lagged_covariance_helper_matrix(
                    model.generate_coefficient_matrix(),
                    model.compute_covariance_matrix(),
                    1,
                )[0][2]
            ),
            0,
            1,
        )

        self.assertAlmostEqual(
            (5 * covariances["X"]["Y"] + 0.9 * variances["Y"])
            - (
                compute_lagged_covariance_helper_matrix(
                    model.generate_coefficient_matrix(),
                    model.compute_covariance_matrix(),
                    1,
                )[1][1]
            ),
            0,
            1,
        )

        self.assertGreater(
            0.1,
            abs(
                (5 * variances["X"] + 0.9 * covariances["X"]["Y"])
                - (
                    compute_lagged_covariance_helper_matrix(
                        model.generate_coefficient_matrix(),
                        model.compute_covariance_matrix(),
                        1,
                    )[1][0]
                )
            ),
        )

        self.assertGreater(
            0.1,
            abs(
                (7 * covariances["Y"]["X"] + 0.9 * covariances["Z"]["X"])
                - (
                    compute_lagged_covariance_helper_matrix(
                        model.generate_coefficient_matrix(),
                        model.compute_covariance_matrix(),
                        1,
                    )[2][0]
                )
            ),
        )

        self.assertGreater(
            0.1,
            abs(
                (7 * variances["Y"] + 0.9 * covariances["Z"]["Y"])
                - (
                    compute_lagged_covariance_helper_matrix(
                        model.generate_coefficient_matrix(),
                        model.compute_covariance_matrix(),
                        1,
                    )[2][1]
                )
            ),
        )

    def test_conditional_covariance(self):
        model = MODEL_DAG_TWO_VARIABLES
        covariance_matrix = model.compute_covariance_matrix()
        print(covariance_matrix)
        print(conditional_covariance(covariance_matrix, 0, [1])[0][0])
        self.assertAlmostEqual(
            conditional_covariance(covariance_matrix, 0, [1])[0][0],
            (
                covariance_matrix[0][0]
                - (
                    covariance_matrix[0][1]
                    * covariance_matrix[1][0]
                    / covariance_matrix[1][1]
                )
            ),
            1,
        )

    def test_conditional_covariance_two_dimensional(self):
        self.assertAlmostEqual(
            conditional_covariance(np.array([[1, 0], [0, 1]]), 0, [1])[0][0], 1, 2
        )
        self.assertAlmostEqual(
            conditional_covariance(np.array([[5, 0], [0, 1]]), 0, [1])[0][0], 5, 2
        )
        self.assertAlmostEqual(
            conditional_covariance(np.array([[1, 0], [0, 5]]), 0, [1])[0][0], 1, 2
        )
        self.assertAlmostEqual(
            conditional_covariance(np.array([[1, 0], [0, 1]]), 1, [0])[0][0], 1, 2
        )
        self.assertAlmostEqual(
            conditional_covariance(np.array([[5, 0], [0, 1]]), 1, [0])[0][0], 1, 2
        )
        self.assertAlmostEqual(
            conditional_covariance(np.array([[1, 0], [0, 5]]), 1, [0])[0][0], 5, 2
        )

    def test_conditional_covariance_three_dimensional(self):
        Sigma = np.array([[2, 1, 2], [1, 1, 0], [2, 0, 1]])
        index = 0
        indices = [1, 2]
        self.assertAlmostEqual(
            conditional_covariance(Sigma, index, indices)[0][0], -3, 1
        )

    def test_conditional_covariance_three_dimensional_2(self):
        Sigma = np.array([[5, 1, 2], [1, 1, 0], [2, 0, 1]])
        index = 0
        indices = [1, 2]
        self.assertAlmostEqual(
            conditional_covariance(Sigma, index, indices)[0][0], 0, 1
        )

    def test_conditional_covariance_three_dimensional_3(self):
        Sigma = np.array([[5, 1, 2], [1, 1, 2], [2, 0, 1]])
        index = 0
        indices = [1, 2]
        self.assertAlmostEqual(
            conditional_covariance(Sigma, index, indices)[0][0], 4, 1
        )

    def test_conditional_covariance_three_dimensional_4(self):
        Sigma = np.array([[5, 1, 2], [1, 1, 0], [2, 3, 1]])
        index = 0
        indices = [1, 2]
        self.assertAlmostEqual(
            conditional_covariance(Sigma, index, indices)[0][0], 6, 1
        )

    def test_conditional_covariance_three_dimensional_5(self):
        Sigma = np.array([[5, 1, 2], [1, 1, 3], [2, 0, 1]])
        index = 0
        indices = [1, 2]
        self.assertAlmostEqual(
            conditional_covariance(Sigma, index, indices)[0][0], 6, 1
        )

    def test_conditional_covariance_four_dimensional(self):
        Sigma = np.array([[5, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        index = 0
        indices = [1, 2, 3]
        self.assertAlmostEqual(
            conditional_covariance(Sigma, index, indices)[0][0], 5, 1
        )

    def test_conditional_covariance_four_dimensional_2(self):
        Sigma = np.array([[5, 0, 0, 0], [0, 1, 2, 2], [0, 2, 1, 2], [0, 2, 2, 1]])
        index = 0
        indices = [1, 2, 3]
        self.assertAlmostEqual(
            conditional_covariance(Sigma, index, indices)[0][0], 5, 1
        )

    def test_conditional_covariance_four_dimensional_4(self):
        Sigma = np.array([[5, 1, 1, 1], [1, 1, 0, 2], [1, 0, 1, 0], [1, 0, 0, 1]])
        index = 0
        indices = [1, 2, 3]
        self.assertAlmostEqual(
            conditional_covariance(Sigma, index, indices)[0][0], 2, 1
        )

    def test_conditional_covariance_four_dimensional_4(self):
        Sigma = np.array([[5, 1, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, -2, 1]])
        index = 0
        indices = [1, 2, 3]
        self.assertAlmostEqual(
            conditional_covariance(Sigma, index, indices)[0][0], 0, 1
        )


if __name__ == "__main__":
    unittest.main()
