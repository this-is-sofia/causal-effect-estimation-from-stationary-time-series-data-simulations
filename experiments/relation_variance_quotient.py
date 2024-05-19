import math
import multiprocessing as mp

import numpy as np
import torch

from causy.graph import Node
from matplotlib import pyplot as plt

import config
from analytic_conditional_variances import compute_lagged_covariance_helper_matrix, conditional_covariance
from models import MODEL_DAG_TWO_VARIABLES, MODEL_DAG_THREE_VARIABLES
from plot_helpers import fix_node_names_for_title
from rooted_mean_squared_errors import calculate_mean_variance_and_mean_squared_error


def plot_relation_mse_variance_quotient(comparision_dictionary, start_node, end_node, adjustment_set):
    """
    Plot the relation between the MSE and the variance quotient for each sample size.
    The MSE is multiplied by the square root of the sample size to make it comparable to the variance quotient.
    """
    for key in comparision_dictionary["rooted MSE"].keys():
        comparision_dictionary["rooted MSE"][key] = math.sqrt(comparision_dictionary[
            "rooted MSE"
        ][key]) * math.sqrt(key)

    plt.plot(
        list(comparision_dictionary["rooted MSE"].keys()),
        list(comparision_dictionary["rooted MSE"].values()),
        label="rooted MSE times square root of sample size",
    )
    plt.plot(
        list(comparision_dictionary["conditional variance quotient"].keys()),
        list(comparision_dictionary["conditional variance quotient"].values()),
        label="conditional variance quotient",
    )
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=5)
    plt.title(
        f"Relation MSE and variance quotient \n from {fix_node_names_for_title(start_node)} to {fix_node_names_for_title(end_node)} \n with adjustment set {fix_node_names_for_title(adjustment_set)}",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(f"output/relation/conditional_variance_quotient_test_{start_node}_{end_node}_{adjustment_set}", dpi=config.DPI)
    plt.show()


def retrieve_variables_for_regression_as_nodes(
    model, start_node: str, end_node: str, adjustment_set: list[str]
) -> tuple[Node, Node, list[Node]]:
    """
    generate graph nodes for node references
    :param model: the model the nodes should be generated from
    :param start_node: the start node reference (e.g. "X-t1")
    :param end_node: the end node reference (e.g. "Y-t4")
    :param adjustment_set: a list of adjustment set references (e.g. ["Z-t1", "Y-t1"])
    :return:
    """
    _, graph = model.generate(10)
    start_node_obj = graph.nodes[start_node]
    end_node_obj = graph.nodes[end_node]
    adjustment_set_nodes = []
    for string in adjustment_set:
        adjustment_set_nodes.append(graph.nodes[string])
    return start_node_obj, end_node_obj, adjustment_set_nodes


def generate_diagonal_matrix_with_variances(
    nodes: list[Node], model, covariance_matrix: torch.Tensor
):
    """
    generate a diagonal matrix with the variances of the nodes
    :param nodes: the nodes in order of [end_node, start_node, adjustment_set_nodes]
    :param model:
    :param covariance_matrix: the covariance matrix in arbitrary order
    :return:
    """
    variances = []
    for node in nodes:
        var_name = node.metadata["variable"]
        variances.append(
            covariance_matrix[model.matrix_position_mapping()[var_name]][
                model.matrix_position_mapping()[var_name]
            ]
        )
    return np.diag(variances)


def compute_lagged_covariance_matrix(
    start_node: str, end_node: str, adjustment_set: list[str], model
):
    """
    :param start_node:
    :param end_node:
    :param adjustment_set:
    :param model:
    :return:
    """
    # compute covariance matrix
    coefficient_matrix = model.generate_coefficient_matrix()
    covariance_matrix = model.compute_covariance_matrix()
    coefficient_matrix_size, _ = np.array(coefficient_matrix).shape

    # initialize model
    (
        start_node,
        end_node,
        adjustment_set_nodes,
    ) = retrieve_variables_for_regression_as_nodes(
        model, start_node, end_node, adjustment_set
    )

    ordered_nodes = [end_node] + [start_node] + adjustment_set_nodes

    # initialize lagged covariance matrix by adding the variances on the diagonal in the order of [end_node, start_node, adjustment_set_nodes]
    lagged_covariance_matrix = generate_diagonal_matrix_with_variances(
        ordered_nodes, model, covariance_matrix
    )

    postion_mapping = model.matrix_position_mapping()

    # fill in covariances
    for count, node1 in enumerate(ordered_nodes):
        var_name_1 = node1.metadata["variable"]
        for count2, node2 in enumerate(ordered_nodes):
            if count2 > count:
                var_name_2 = node2.metadata["variable"]
                lag = node1.metadata["time"] - node2.metadata["time"]
                lagged_covariance_helper_matrix = (
                    compute_lagged_covariance_helper_matrix(
                        coefficient_matrix, covariance_matrix, lag
                    )
                )
                lagged_covariance_matrix[count][
                    count2
                ] = lagged_covariance_helper_matrix[postion_mapping[var_name_1]][
                    postion_mapping[var_name_2]
                ]
                lagged_covariance_matrix[count2][
                    count
                ] = lagged_covariance_helper_matrix[postion_mapping[var_name_1]][
                    postion_mapping[var_name_2]
                ]
    return lagged_covariance_matrix


def conditional_standard_deviation_quotient(lagged_covariance_matrix):
    list_of_indeces = list(range(1, len(lagged_covariance_matrix)))
    list_of_indeces_2 = list(range(2, len(lagged_covariance_matrix)))
    return math.sqrt(
        conditional_covariance(lagged_covariance_matrix, 0, list_of_indeces)[0][0]
    ) / math.sqrt(
        conditional_covariance(lagged_covariance_matrix, 1, list_of_indeces_2)[0][0]
    )


def arguments_wrapper(args):
    return args[0](*args[1:])


def compare_conditional_variance_and_mse(
    start_node,
    end_node,
    adjustment_set,
    true_effect,
    model,
    repetitions,
    sample_sizes,
):
    lagged_covariance_matrix = compute_lagged_covariance_matrix(
        start_node, end_node, adjustment_set, model
    )
    comparision_dictionary = {"rooted MSE": {}, "conditional variance quotient": {}}

    parallel = True
    with mp.Pool(mp.cpu_count()) as pool:
        if parallel:
            rooted_mses = pool.map(
                arguments_wrapper,
                [
                    [
                        calculate_mean_variance_and_mean_squared_error,
                        sample_size,
                        true_effect,
                        start_node,
                        end_node,
                        adjustment_set,
                        repetitions,
                        model,
                    ]
                    for sample_size in sample_sizes
                ],
            )
        for i, sample_size in enumerate(sample_sizes):
            print(f"sample size: {sample_size}")
            if not parallel:
                comparision_dictionary["rooted MSE"][sample_size] = (
                    calculate_mean_variance_and_mean_squared_error(
                        sample_size,
                        true_effect,
                        start_node,
                        end_node,
                        adjustment_set,
                        repetitions,
                        model,
                    )[0]
                    ** 0.5
                )

            if parallel:
                comparision_dictionary["rooted MSE"][sample_size] = rooted_mses[i][0]
            standard_deviation_quotient = conditional_standard_deviation_quotient(
                lagged_covariance_matrix
            )
            comparision_dictionary["conditional variance quotient"][
                sample_size
            ] = standard_deviation_quotient
        return comparision_dictionary

def run_comparision(start_node, end_node, adjustment_set, true_effect, model, repetitions, sample_sizes):
    comparision_dictionary = compare_conditional_variance_and_mse(
        start_node=start_node, end_node=end_node, adjustment_set=adjustment_set, true_effect=true_effect, model=model, repetitions=repetitions,
        sample_sizes=sample_sizes)

    plot_relation_mse_variance_quotient(comparision_dictionary=comparision_dictionary, start_node=start_node,
                                        end_node=end_node, adjustment_set=adjustment_set)

def run_experiment():
    model = MODEL_DAG_TWO_VARIABLES
    run_comparision("X-t2", "Y-t3", ["X-t1"], 5, model, 10000, config.SAMPLE_SIZES_EQUIDISTANT)
    run_comparision("X-t2", "Y-t3", ["Y-t2"], 5, model, 10000, config.SAMPLE_SIZES_EQUIDISTANT)
    run_comparision("X-t2", "Y-t4", ["X-t1"], 9, model, 10000, config.SAMPLE_SIZES_EQUIDISTANT)
    run_comparision("X-t2", "Y-t4", ["Y-t2"], 9, model, 10000, config.SAMPLE_SIZES_EQUIDISTANT)
    run_comparision("X-t2", "Y-t5", ["X-t1"], 12.15, model, 10000, config.SAMPLE_SIZES_EQUIDISTANT)
    run_comparision("X-t2", "Y-t5", ["Y-t2"], 12.15, model, 10000, config.SAMPLE_SIZES_EQUIDISTANT)

    model = MODEL_DAG_THREE_VARIABLES
    run_comparision("X-t2", "Z-t4", ["Y-t2", "Z-t3"], 35, model, 10000, config.SAMPLE_SIZES_EQUIDISTANT)
    run_comparision("X-t2", "Z-t4", ["X-t1"], 35, model, 10000, config.SAMPLE_SIZES_EQUIDISTANT)