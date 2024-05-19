import numpy as np
import torch


def calculate_estimated_effect_and_squared_error(
    sample_size: int,
    true_total_causal_effect: float,
    start_node: str,
    end_node: str,
    adjustment_set: list[str],
    model,
):
    """
    Compute the square of the difference between the true total causal effect and the estimated total causal effect.
    :param sample_size: number of time steps
    :param true_total_causal_effect: true total causal effect of X_{t} on Y_{t+k}
    :param start_node: name of the start node, e.g. "X-t1"
    :param end_node: name of the end node, e.g. "Y-t4"
    :param adjustment_set: list of names of nodes in the adjustment set, e.g. ["Z-t1", "Y-t1"]
    """
    data, graph = model.generate(sample_size)

    start_node = graph.nodes[start_node]
    end_node = graph.nodes[end_node]
    adjustment_set = [graph.nodes[adj] for adj in adjustment_set]

    # combine the adjustment set and the start node

    # add all the time stamps to a list to compute the cutoff threshold
    time_stamps = []
    for adj in adjustment_set:
        time_stamps.append(adj.metadata["time"])
    time_stamps.append(start_node.metadata["time"])
    time_stamps.append(end_node.metadata["time"])
    time_stamp_gap = max(time_stamps) - min(time_stamps)

    cut_datasets = []
    for adj in [start_node] + adjustment_set:
        if adj.metadata["time"] == max(time_stamps):
            cut_datasets.append(data[adj.metadata["variable"]][time_stamp_gap:])
        else:
            cut_datasets.append(
                data[adj.metadata["variable"]][
                    adj.metadata["time"]
                    - min(time_stamps) : -(
                        time_stamp_gap - (adj.metadata["time"] - min(time_stamps))
                    )
                ]
            )

    if end_node.metadata["time"] == max(time_stamps):
        cut_data_end_node = data[end_node.metadata["variable"]][time_stamp_gap:]
    else:
        cut_data_end_node = data[end_node.metadata["variable"]][
            end_node.metadata["time"]
            - min(time_stamps) : -(
                time_stamp_gap - (end_node.metadata["time"] - min(time_stamps))
            )
        ]
    regressors = torch.stack(cut_datasets)
    regressors = regressors.T

    coefficients_ols = torch.linalg.lstsq(
        regressors,
        cut_data_end_node,
        driver="gelsd",
    ).solution

    estimated_total_causal_effect = coefficients_ols[0]
    squared_error = (true_total_causal_effect - estimated_total_causal_effect) ** 2

    return squared_error.item(), estimated_total_causal_effect.item()


def durbin_watson_test(
    sample_size: int,
    true_total_causal_effect: float,
    start_node: str,
    end_node: str,
    adjustment_set: list[str],
    model,
):
    residuals = []
    squared_errors = []
    for data_point in range(sample_size):
        squared_error, _ = calculate_estimated_effect_and_squared_error(
            sample_size,
            true_total_causal_effect,
            start_node,
            end_node,
            adjustment_set,
            model,
        )
        residuals.append(squared_error**0.5)
        squared_errors.append(squared_error)
    diff_res = np.diff(residuals)
    dw = np.sum(diff_res**2) / np.sum(squared_errors)
    return dw


def calculate_mean_variance_and_mean_squared_error(
    sample_size: int,
    true_total_causal_effect: float,
    start_node: str,
    end_node: str,
    adjustment_set: list[str],
    repetitions: int,
    model,
):
    """
    :param sample_size: number of time steps
    :param true_total_causal_effect: true total causal effect of X_{t} on Y_{t+k}
    :param start_node: name of the start node, e.g. "X-t1"
    :param end_node: name of the end node, e.g. "Y-t4"
    :param adjustment_set: list of names of nodes in the adjustment set, e.g. ["Z-t1", "Y-t1"]
    :param repetitions: number of repetitions
    """
    squared_errors = []
    estimated_total_causal_effects = []
    for _ in range(repetitions):
        (
            squared_error,
            estimated_total_causal_effect,
        ) = calculate_estimated_effect_and_squared_error(
            sample_size,
            true_total_causal_effect,
            start_node,
            end_node,
            adjustment_set,
            model,
        )
        squared_errors.append(squared_error)
        estimated_total_causal_effects.append(estimated_total_causal_effect)
    mean_squared_error = np.mean(squared_errors)
    mean_of_estimation = np.mean(estimated_total_causal_effects)
    variance_of_estimation = np.var(estimated_total_causal_effects)
    return (
        mean_squared_error,
        mean_of_estimation,
        variance_of_estimation,
    )
