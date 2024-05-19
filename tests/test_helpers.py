from experiments.relation_variance_quotient import retrieve_variables_for_regression_as_nodes, generate_diagonal_matrix_with_variances, \
    compute_lagged_covariance_matrix, conditional_standard_deviation_quotient
from analytic_conditional_variances import (
    conditional_covariance,
)
import unittest
from models import MODEL_DAG_TWO_VARIABLES
import numpy as np
import math


class TestHelpers(unittest.TestCase):
    def test_retrieve_variables_for_regression_as_nodes(self):
        model = MODEL_DAG_TWO_VARIABLES
        start_node = "X-t2"
        end_node = "Y-t3"
        adjustment_set = ["Y-t2"]
        (
            start_node_obj,
            end_node_obj,
            adjustment_set_nodes,
        ) = retrieve_variables_for_regression_as_nodes(
            model, start_node, end_node, adjustment_set
        )
        self.assertEqual(start_node_obj.id, "X-t2")
        self.assertEqual(end_node_obj.id, "Y-t3")
        self.assertEqual(adjustment_set_nodes[0].id, "Y-t2")

    def test_retrieve_variables_for_regression_as_nodes_2(self):
        model = MODEL_DAG_TWO_VARIABLES
        start_node = "X-t2"
        end_node = "Y-t3"
        adjustment_set = ["Y-t2", "X-t1"]
        (
            start_node_obj,
            end_node_obj,
            adjustment_set_nodes,
        ) = retrieve_variables_for_regression_as_nodes(
            model, start_node, end_node, adjustment_set
        )
        self.assertEqual(start_node_obj.id, "X-t2")
        self.assertEqual(end_node_obj.id, "Y-t3")
        self.assertEqual(adjustment_set_nodes[0].id, "Y-t2")
        self.assertEqual(adjustment_set_nodes[1].id, "X-t1")

    def test_generate_diagonal_matrix_with_variances(self):
        model = MODEL_DAG_TWO_VARIABLES
        start_node = "X-t2"
        end_node = "Y-t3"
        adjustment_set = ["Y-t2"]
        (
            start_node_obj,
            end_node_obj,
            adjustment_set_nodes,
        ) = retrieve_variables_for_regression_as_nodes(
            model, start_node, end_node, adjustment_set
        )
        covariance_matrix = model.compute_covariance_matrix()
        diagonal_matrix = generate_diagonal_matrix_with_variances(
            [end_node_obj] + [start_node_obj] + adjustment_set_nodes,
            model,
            covariance_matrix,
        )
        self.assertAlmostEqual(diagonal_matrix[0][0], 6602.43, 0)
        self.assertAlmostEqual(diagonal_matrix[1][1], 5.26, 1)
        self.assertAlmostEqual(diagonal_matrix[2][2], 6602.43, 0)

    def test_generate_diagonal_matrix_with_variances(self):
        model = MODEL_DAG_TWO_VARIABLES
        start_node = "X-t2"
        end_node = "Y-t3"
        adjustment_set = ["Y-t2", "X-t1"]
        (
            start_node_obj,
            end_node_obj,
            adjustment_set_nodes,
        ) = retrieve_variables_for_regression_as_nodes(
            model, start_node, end_node, adjustment_set
        )
        covariance_matrix = model.compute_covariance_matrix()
        diagonal_matrix = generate_diagonal_matrix_with_variances(
            [end_node_obj] + [start_node_obj] + adjustment_set_nodes,
            model,
            covariance_matrix,
        )
        self.assertAlmostEqual(diagonal_matrix[0][0], 6602.43, 0)
        self.assertAlmostEqual(diagonal_matrix[1][1], 5.26, 1)
        self.assertAlmostEqual(diagonal_matrix[2][2], 6602.43, 0)
        self.assertAlmostEqual(diagonal_matrix[3][3], 5.26, 1)

    def test_compute_lagged_covariance_matrix(self):
        model = MODEL_DAG_TWO_VARIABLES
        start_node = "X-t2"
        end_node = "Y-t3"
        adjustment_set = ["Y-t2"]
        lagged_covariance_matrix = compute_lagged_covariance_matrix(
            start_node, end_node, adjustment_set, model
        )
        # variance of end node
        self.assertAlmostEqual(lagged_covariance_matrix[0][0], 6602.43, 0)
        # covariance between end node and start node
        self.assertAlmostEqual(
            lagged_covariance_matrix[1][0], (5 * 5.26 + 0.9 * 124.65), 1
        )
        self.assertAlmostEqual(
            lagged_covariance_matrix[0][1], (5 * 5.26 + 0.9 * 124.65), 1
        )
        # variance of start node
        self.assertAlmostEqual(lagged_covariance_matrix[1][1], 5.26, 1)
        # covariance between start node and first node in adjustment set
        self.assertAlmostEqual(lagged_covariance_matrix[1][2], (124.65), 1)
        self.assertAlmostEqual(lagged_covariance_matrix[2][1], (124.65), 1)
        # variance of first node in adjustment set
        self.assertAlmostEqual(lagged_covariance_matrix[2][2], 6602.43, 0)
        # covariance between end node and first node in adjustment set
        self.assertAlmostEqual(
            lagged_covariance_matrix[0][2], (0.9 * 6602.43 + 5 * 124.65), 0
        )
        self.assertAlmostEqual(
            lagged_covariance_matrix[2][0], (0.9 * 6602.43 + 5 * 124.65), 0
        )
        # test that matrix is symmetric
        self.assertTrue(
            np.allclose(lagged_covariance_matrix, lagged_covariance_matrix.T)
        )
        # test that matrix is positive definite
        self.assertTrue(np.all(np.linalg.eigvals(lagged_covariance_matrix) > 0))

    def test_compute_lagged_covariance_matrix_2(self):
        model = MODEL_DAG_TWO_VARIABLES
        start_node = "X-t2"
        end_node = "Y-t3"
        adjustment_set = ["Y-t2", "X-t1"]
        lagged_covariance_matrix = compute_lagged_covariance_matrix(
            start_node, end_node, adjustment_set, model
        )
        # variance of end node
        self.assertAlmostEqual(lagged_covariance_matrix[0][0], 6602.43, 0)
        # covariance between end node and start node
        self.assertAlmostEqual(
            lagged_covariance_matrix[1][0], (5 * 5.26 + 0.9 * 124.65), 1
        )
        self.assertAlmostEqual(
            lagged_covariance_matrix[0][1], (5 * 5.26 + 0.9 * 124.65), 1
        )
        # variance of start node
        self.assertAlmostEqual(lagged_covariance_matrix[1][1], 5.26, 1)
        # covariance between start node and first node in adjustment set
        self.assertAlmostEqual(
            lagged_covariance_matrix[2][0], (5 * 124.65 + 0.9 * 6602.43), 0
        )
        self.assertAlmostEqual(
            lagged_covariance_matrix[0][2], (5 * 124.65 + 0.9 * 6602.43), 0
        )
        # variance of first node in adjustment set
        self.assertAlmostEqual(lagged_covariance_matrix[2][2], 6602.43, 0)
        # covariance between start node and first node in adjustment set
        self.assertAlmostEqual(lagged_covariance_matrix[2][1], (124.65), 0)
        self.assertAlmostEqual(lagged_covariance_matrix[1][2], (124.65), 0)
        # variance of second node in adjustment set
        self.assertAlmostEqual(lagged_covariance_matrix[3][3], 5.26, 1)
        # covariance between first node in adjustment set and second node in adjustment set
        self.assertAlmostEqual(
            lagged_covariance_matrix[2][3], (0.9 * 124.65 + 5 * 5.26), 0
        )
        self.assertAlmostEqual(
            lagged_covariance_matrix[3][2], (0.9 * 124.65 + 5 * 5.26), 0
        )
        # test that matrix is symmetric
        self.assertTrue(
            np.allclose(lagged_covariance_matrix, lagged_covariance_matrix.T)
        )
        # test that matrix is positive definite
        self.assertTrue(np.all(np.linalg.eigvals(lagged_covariance_matrix) > 0))

    def test_conditional_standard_deviation_quotient(self):
        model = MODEL_DAG_TWO_VARIABLES
        lagged_covariance_matrix = compute_lagged_covariance_matrix(
            "X-t2", "Y-t3", ["Y-t2"], model
        )
        conditional_variance_numerator = conditional_covariance(
            lagged_covariance_matrix, 0, [1, 2]
        )
        conditional_variance_denominator = conditional_covariance(
            lagged_covariance_matrix, 1, [2]
        )
        conditional_variance_quotient = conditional_standard_deviation_quotient(
            lagged_covariance_matrix
        )
        self.assertAlmostEqual(
            conditional_variance_quotient,
            math.sqrt(
                conditional_variance_numerator[0][0]
                / conditional_variance_denominator[0][0]
            ),
            2,
        )
        self.assertAlmostEqual(
            conditional_variance_quotient,
            math.sqrt(conditional_variance_numerator[0][0])
            / math.sqrt(conditional_variance_denominator[0][0]),
            2,
        )

    def test_conditional_standard_deviation_quotient_2(self):
        model = MODEL_DAG_TWO_VARIABLES
        lagged_covariance_matrix = compute_lagged_covariance_matrix(
            "X-t2", "Y-t3", ["Y-t2", "X-t1"], model
        )
        conditional_variance_numerator = conditional_covariance(
            lagged_covariance_matrix, 0, [1, 2, 3]
        )
        conditional_variance_denominator = conditional_covariance(
            lagged_covariance_matrix, 1, [2, 3]
        )
        conditional_variance_quotient = conditional_standard_deviation_quotient(
            lagged_covariance_matrix
        )
        self.assertAlmostEqual(
            conditional_variance_quotient,
            math.sqrt(
                conditional_variance_numerator[0][0]
                / conditional_variance_denominator[0][0]
            ),
            2,
        )
        self.assertAlmostEqual(
            conditional_variance_quotient,
            math.sqrt(conditional_variance_numerator[0][0])
            / math.sqrt(conditional_variance_denominator[0][0]),
            2,
        )
