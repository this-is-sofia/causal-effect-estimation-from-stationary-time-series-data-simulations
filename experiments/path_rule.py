import numpy as np
import torch
from matplotlib import pyplot as plt

import config
from models import MODEL_DAG_TWO_VARIABLES, MODEL_DAG_THREE_VARIABLES
from rooted_mean_squared_errors import calculate_estimated_effect_and_squared_error

def run_experiment():

    model = MODEL_DAG_TWO_VARIABLES
    sample_sizes = config.SAMPLE_SIZES_SMALL
    # Compute the causal effect of X-t2 on Y-t5 given Henckel's adjustment set ["Y-t2"]
    estimated_effects_henckel = {}
    for sample_size in sample_sizes:
        print(f"henckel={sample_size}")
        estimated_effects_henckel[sample_size] = []
        for _ in range(100):
            estimated_effects_henckel[sample_size].append(calculate_estimated_effect_and_squared_error(sample_size, 5, "X-t2", "Y-t5", ["Y-t2"], model)[1])

    # Compute the causal effect of X-t2 on Y-t5 given a suboptimal valid adjustment set ["X-t1"]
    estimated_effect_adj_set = {}
    for sample_size in sample_sizes:
        print(f"adj_set={sample_size}")
        estimated_effect_adj_set[sample_size] = []
        for _ in range(100):
            estimated_effect_adj_set[sample_size].append(calculate_estimated_effect_and_squared_error(sample_size, 5, "X-t2", "Y-t5", ["X-t1"], model)[1])

    # Compute the causal effect of X-t2 on Y-t5 given the path-based estimator: Computing the edge coefficients of edges on causal paths via regressing Y in a directed edge (X,Y) on all its parents and taking the coefficient corresponding to X
    estimated_effect_path_based = {}
    for sample_size in sample_sizes:
        estimated_effect_path_based[sample_size] = []
        print(f"estimated_effect_path_based={sample_size}")
        coefficients_Yt4_on_parents_means = []
        coefficient_Xt3_on_parents_means = []

        coefficients_X1X2 = []
        coefficients_Y1Y2 = []
        coefficients_X1Y2 = []

        for _ in range(100):
            coefficient_X1X2 = calculate_estimated_effect_and_squared_error(sample_size, true_total_causal_effect=0, start_node="X-t1", end_node="X-t2", adjustment_set=[], model=model)[1]
            coefficient_Y1Y2 = calculate_estimated_effect_and_squared_error(sample_size, true_total_causal_effect=0, start_node="Y-t1", end_node="Y-t2", adjustment_set=["X-t1"], model=model)[1]
            coefficient_X1Y2 = calculate_estimated_effect_and_squared_error(sample_size, true_total_causal_effect=0, start_node="X-t1", end_node="Y-t2", adjustment_set=["Y-t1"], model=model)[1]

            coefficients_X1Y2.append(coefficient_X1Y2)
            coefficients_X1X2.append(coefficient_X1X2)
            coefficients_Y1Y2.append(coefficient_Y1Y2)

            path_rule_estimator = (coefficient_X1X2**2 * coefficient_X1Y2) + (coefficient_X1X2 * coefficient_X1Y2 * coefficient_Y1Y2) + (coefficient_X1Y2 * coefficient_Y1Y2**2)
            estimated_effect_path_based[sample_size].append(path_rule_estimator)

    exp_henckel = []
    exp_adj = []
    exp_path_based = []

    var_henckel = []
    var_adj = []
    var_path_based = []

    for sample_size in sample_sizes:
        exp_henckel.append(np.mean(estimated_effects_henckel[sample_size]))
        exp_adj.append(np.mean(estimated_effect_adj_set[sample_size]))
        exp_path_based.append(np.mean(estimated_effect_path_based[sample_size]))

        var_henckel.append(np.var(estimated_effects_henckel[sample_size]))
        var_adj.append(np.var(estimated_effect_adj_set[sample_size]))
        var_path_based.append(np.var(estimated_effect_path_based[sample_size]))

    plt.plot(sample_sizes, exp_henckel, label="henckel")
    plt.plot(sample_sizes, exp_adj, label="adj")
    plt.plot(sample_sizes, exp_path_based, label="path_based")
    plt.legend()
    plt.ylim(9,14)
    plt.savefig(f"output/path_based/expectations", dpi=config.DPI)
    plt.clf()

    plt.plot(sample_sizes, var_henckel, label="henckel")
    plt.plot(sample_sizes, var_adj, label="adj")
    plt.plot(sample_sizes, var_path_based, label="path_based")
    plt.ylim(0,4)
    plt.legend()
    plt.savefig(f"output/path_based/variances", dpi=config.DPI)

    model = MODEL_DAG_THREE_VARIABLES
    sample_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    estimated_effects_henckel = {}
    for sample_size in sample_sizes:
        print(f"henckel={sample_size}")
        estimated_effects_henckel[sample_size] = []
        for _ in range(100):
            estimated_effects_henckel[sample_size].append(calculate_estimated_effect_and_squared_error(sample_size, 35, "X-t2", "Z-t4", ["Z-t3", "Y-t2"], model)[1])

    # Compute the causal effect of X-t2 on Y-t5 given a suboptimal valid adjustment set ["X-t1"]
    estimated_effect_adj_set = {}
    for sample_size in sample_sizes:
        print(f"adj_set={sample_size}")
        estimated_effect_adj_set[sample_size] = []
        for _ in range(100):
            estimated_effect_adj_set[sample_size].append(calculate_estimated_effect_and_squared_error(sample_size, 35, "X-t2", "Z-t4", ["Z-t3", "Y-t2", "X-t1"], model)[1])

    # Compute the causal effect of X-t2 on Y-t5 given the path-based estimator: Computing the edge coefficients of edges on causal paths via regressing Y in a directed edge (X,Y) on all its parents and taking the coefficient corresponding to X
    estimated_effect_path_based = {}
    for sample_size in sample_sizes:
        estimated_effect_path_based[sample_size] = []
        print(f"estimated_effect_path_based={sample_size}")

        coefficients_X1Y2 = []
        coefficients_Y1Z2 = []

        for _ in range(100):
            coefficient_X1Y2 = calculate_estimated_effect_and_squared_error(sample_size, true_total_causal_effect=0, start_node="X-t1", end_node="Y-t2", adjustment_set=["Y-t1"], model=model)[1]
            coefficient_Y1Z2 = calculate_estimated_effect_and_squared_error(sample_size, true_total_causal_effect=0, start_node="Y-t1", end_node="Z-t2", adjustment_set=["Z-t1"], model=model)[1]

            coefficients_X1Y2.append(coefficient_X1Y2)
            coefficients_Y1Z2.append(coefficient_Y1Z2)

            path_rule_estimator = coefficient_X1Y2 * coefficient_Y1Z2
            estimated_effect_path_based[sample_size].append(path_rule_estimator)

    exp_henckel = []
    exp_adj = []
    exp_path_based = []

    var_henckel = []
    var_adj = []
    var_path_based = []

    for sample_size in sample_sizes:
        exp_henckel.append(np.mean(estimated_effects_henckel[sample_size]))
        exp_adj.append(np.mean(estimated_effect_adj_set[sample_size]))
        exp_path_based.append(np.mean(estimated_effect_path_based[sample_size]))

        var_henckel.append(np.var(estimated_effects_henckel[sample_size]))
        var_adj.append(np.var(estimated_effect_adj_set[sample_size]))
        var_path_based.append(np.var(estimated_effect_path_based[sample_size]))


    plt.plot(sample_sizes, exp_henckel, label="henckel")
    plt.plot(sample_sizes, exp_adj, label="adj")
    plt.plot(sample_sizes, exp_path_based, label="path_based")
    plt.legend()
    plt.ylim(32, 38)
    plt.savefig(f"output/path_based/expectations_second_model", dpi=config.DPI)
    plt.clf()

    plt.plot(sample_sizes, var_henckel, label="henckel")
    plt.plot(sample_sizes, var_adj, label="adj")
    plt.plot(sample_sizes, var_path_based, label="path_based")
    plt.legend()
    plt.savefig(f"output/path_based/variances_second_model", dpi=config.DPI)

