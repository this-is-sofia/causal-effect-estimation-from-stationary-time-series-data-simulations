import copy
import unittest
import numpy as np
import torch
import random

from models import MODEL_DAG_TWO_VARIABLES
from rooted_mean_squared_errors import (
    calculate_mean_variance_and_mean_squared_error,
    calculate_estimated_effect_and_squared_error,
    durbin_watson_test,
)


class TestConditionalVarianceComputations(unittest.TestCase):
    SEED = 42

    def setUp(self):
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.SEED)
        random.seed(0)

    def test_estimated_effect_in_calculate_estimated_effect_and_squared_error(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        (
            squared_error,
            estimated_total_causal_effect,
        ) = calculate_estimated_effect_and_squared_error(
            1000, 12.15, "X-t2", "Y-t5", ["Y-t2"], model
        )
        self.assertAlmostEqual(estimated_total_causal_effect, 12.15, 0)

        (
            squared_error,
            estimated_total_causal_effect,
        ) = calculate_estimated_effect_and_squared_error(
            1000, 5, "X-t2", "Y-t3", ["Y-t2"], model
        )
        self.assertAlmostEqual(estimated_total_causal_effect, 5, 1)

    def test_squared_error_in_calculate_estimated_effect_and_squared_error(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)
        model._initial_distribution_fn = lambda x: torch.tensor(0, dtype=torch.float32)
        model.random_fn = lambda: 0

        sample_size = 10
        true_total_causal_effect = 5
        start_node = "X-t2"
        end_node = "Y-t3"
        adjustment_set = ["Y-t2"]

        squared_error, estimated_effect = calculate_estimated_effect_and_squared_error(
            sample_size,
            true_total_causal_effect,
            start_node,
            end_node,
            adjustment_set,
            model,
        )

        self.assertEqual(estimated_effect, 0)
        self.assertEqual(squared_error, 25)

    def test_squared_error_in_calculate_estimated_effect_and_squared_error_2(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        sample_size = 100
        # set wrong true_total_causal_effect to test squared error
        true_total_causal_effect = 4
        start_node = "X-t2"
        end_node = "Y-t3"
        adjustment_set = ["Y-t2"]

        squared_error, estimated_effect = calculate_estimated_effect_and_squared_error(
            sample_size,
            true_total_causal_effect,
            start_node,
            end_node,
            adjustment_set,
            model,
        )

        self.assertAlmostEqual(estimated_effect, 5, 1)
        self.assertAlmostEqual(squared_error, 1, 1)

    def test_squared_error_in_calculate_estimated_effect_and_squared_error_3(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        sample_size = 100
        # set wrong true_total_causal_effect to test squared error
        true_total_causal_effect = 3
        start_node = "X-t2"
        end_node = "Y-t3"
        adjustment_set = ["Y-t2"]

        squared_error, estimated_effect = calculate_estimated_effect_and_squared_error(
            sample_size,
            true_total_causal_effect,
            start_node,
            end_node,
            adjustment_set,
            model,
        )

        self.assertAlmostEqual(estimated_effect, 5, 1)
        self.assertAlmostEqual(squared_error, 4, 1)

    def test_mean_in_calculate_mean_variance_and_rooted_mean_squared_error(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        mse, mean, variance = calculate_mean_variance_and_mean_squared_error(
            500, 12.15, "X-t2", "Y-t5", ["Y-t2"], 100, model
        )
        self.assertAlmostEqual(mean, 12.15, 0)

    def test_mean_in_calculate_mean_variance_and_rooted_mean_squared_error_2(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        mse, mean, variance = calculate_mean_variance_and_mean_squared_error(
            500, 5, "X-t2", "Y-t3", ["Y-t2"], 100, model
        )
        self.assertAlmostEqual(mean, 5, 0)

    def test_mse_in_calculate_mean_variance_and_rooted_mean_squared_error(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        # set wrong true_total_causal_effect to test mean squared error
        mse, mean, variance = calculate_mean_variance_and_mean_squared_error(
            500, 4, "X-t2", "Y-t3", ["Y-t2"], 100, model
        )
        self.assertAlmostEqual(mse, 1, 1)

    def test_mse_in_calculate_mean_variance_and_rooted_mean_squared_error_2(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        # set wrong true_total_causal_effect to test mean squared error
        mse, mean, variance = calculate_mean_variance_and_mean_squared_error(
            500, 3, "X-t2", "Y-t3", ["Y-t2"], 100, model
        )
        self.assertAlmostEqual(mse, 4, 1)

    def test_variance_in_calculate_mean_variance_and_rooted_mean_squared_error(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        mse, mean, variance = calculate_mean_variance_and_mean_squared_error(
            500, 5, "X-t2", "Y-t3", ["Y-t2"], 100, model
        )
        self.assertAlmostEqual(mse, variance, 1)

    def test_variance_in_calculate_mean_variance_and_rooted_mean_squared_error_2(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        mse, mean, variance = calculate_mean_variance_and_mean_squared_error(
            500, 5, "X-t2", "Y-t3", ["Y-t1"], 100, model
        )

        self.assertGreater(abs(mse - variance), 10)

    def test_variance_in_calculate_mean_variance_and_rooted_mean_squared_error_3(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)
        model._initial_distribution_fn = lambda x: torch.tensor(0, dtype=torch.float32)
        model.random_fn = lambda: 0

        mse, mean, variance = calculate_mean_variance_and_mean_squared_error(
            500, 5, "X-t2", "Y-t3", ["Y-t2"], 100, model
        )

        self.assertEqual(variance, 0, 0)

    def test_durbin_watson_test_null(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        dw = durbin_watson_test(1000, 12.15, "X-t2", "Y-t3", ["Y-t2"], model)
        self.assertAlmostEqual(dw, 0, 1)

    def test_durbin_watson_test_null2(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        dw = durbin_watson_test(1000, 12.15, "X-t2", "Y-t4", ["Y-t2"], model)
        self.assertAlmostEqual(dw, 0, 1)

    def test_durbin_watson_test_null2(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        dw = durbin_watson_test(1000, 12.15, "X-t2", "Y-t5", ["Y-t2"], model)
        self.assertAlmostEqual(dw, 0, 1)

    def test_durbin_watson_test2(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        dw = durbin_watson_test(1000, 12.15, "X-t2", "Y-t3", ["X-t1"], model)
        self.assertGreater(dw, 0.1)

    def test_durbin_watson_test3(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        dw = durbin_watson_test(1000, 12.15, "X-t2", "Y-t4", ["X-t1"], model)
        self.assertGreater(dw, 0.1)

    def test_durbin_watson_test3(self):
        model = copy.deepcopy(MODEL_DAG_TWO_VARIABLES)

        dw = durbin_watson_test(1000, 12.15, "X-t2", "Y-t5", ["X-t1"], model)
        self.assertGreater(dw, 0.1)
