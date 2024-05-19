from data_generator import TimeseriesSampleGenerator
from models import MODEL_DAG_THREE_VARIABLES, MODEL_DAG_TWO_VARIABLES
import unittest
import torch


class TestDataGenerator(unittest.TestCase):
    def test_generate(self):
        model = MODEL_DAG_THREE_VARIABLES
        model.generate(1000)
        self.assertEqual(model.generate(1000)[0]["X"].shape, torch.Size([1000]))

    def test_generate_coefficient_matrix(self):
        model = MODEL_DAG_TWO_VARIABLES
        self.assertEqual(len(model.generate_coefficient_matrix()), 2)
        self.assertEqual(len(model.generate_coefficient_matrix()[0]), 2)
        self.assertEqual(model.generate_coefficient_matrix()[0][0], 0.9)
        self.assertEqual(model.generate_coefficient_matrix()[1][1], 0.9)
        self.assertEqual(model.generate_coefficient_matrix()[1][0], 5)
        self.assertEqual(model.generate_coefficient_matrix()[0][1], 0)

    def test_compute_covariance_matrix(self):
        model = MODEL_DAG_TWO_VARIABLES
        self.assertEqual(len(model.compute_covariance_matrix()), 2)
        self.assertAlmostEqual(model.compute_covariance_matrix()[0][0].item(), 5.26, 1)
        self.assertAlmostEqual(model.compute_covariance_matrix()[1][1].item(), 6602, 0)
        self.assertAlmostEqual(model.compute_covariance_matrix()[0][1].item(), 124.6, 0)

    def test_compute_covariance_matrix(self):
        model = MODEL_DAG_THREE_VARIABLES
        self.assertEqual(len(model.compute_covariance_matrix()), 3)
        self.assertAlmostEqual(model.compute_covariance_matrix()[0][0].item(), 5.26, 1)
        self.assertAlmostEqual(model.compute_covariance_matrix()[1][1].item(), 6602, 0)
        self.assertAlmostEqual(model.compute_covariance_matrix()[0][1].item(), 124.6, 0)
