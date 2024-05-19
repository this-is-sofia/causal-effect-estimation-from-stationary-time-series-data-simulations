from matplotlib import pyplot as plt

import config
from models import MODEL_DAG_THREE_VARIABLES, MODEL_DAG_TWO_VARIABLES
from rooted_mean_squared_errors import calculate_mean_variance_and_mean_squared_error, \
    calculate_estimated_effect_and_squared_error
from plot_helpers import fix_node_names_for_title


def plot_means_rooted_errors_and_variances(
    list_of_sample_sizes: list,
    true_causal_effect: str,
    start_node: str,
    end_node: str,
    adjustment_set: str,
    repetitions: int,
    model,
):
    """
    Plot the means, rooted mean squared errors and variances for a list of sample sizes.
    :param list_of_sample_sizes:
    :param true_causal_effect:
    :param start_node:
    :param end_node:
    :param adjustment_set:
    :param repetitions:
    :return:
    """
    means = []
    errors = []
    variances = []
    for sample_size in list_of_sample_sizes:
        (
            error_value,
            mean_value,
            var_value,
        ) = calculate_mean_variance_and_mean_squared_error(
            sample_size,
            true_causal_effect,
            start_node,
            end_node,
            adjustment_set,
            repetitions,
            model,
        )
        means.append(mean_value)
        errors.append(error_value)
        variances.append(var_value)
    print(f"means={means}, errors={errors}, variances={variances}")

    # plot
    plt.plot(list_of_sample_sizes, means, label="means")
    plt.plot(list_of_sample_sizes, errors, label="MSE")
    plt.plot(list_of_sample_sizes, variances, label="variances")
    plt.plot(
       list_of_sample_sizes,
       [true_causal_effect for _ in list_of_sample_sizes],
       label="true causal effect",
    )
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=5)
    # add a title for the plot over two lines
    plt.title(
       f"Means, Mean Squared Errors and Variances for effect from {fix_node_names_for_title(start_node)} to {fix_node_names_for_title(end_node)} \n with adjustment set {fix_node_names_for_title(adjustment_set)}",
       pad=15,
    )
    plt.tight_layout()
    name_string_with_dots = (
        "output/dag/"
        + str(true_causal_effect)
        + "_"
        + start_node
        + "_"
        + end_node
        + "_"
        + str(adjustment_set)
        + "_"
        + str(repetitions)
    )
    name_string_without_dots = name_string_with_dots.replace(".", "-")
    print(name_string_with_dots)
    plt.savefig(name_string_without_dots, dpi=config.DPI)
    plt.show()
    return means, errors, variances

def run_experiment():
    model = MODEL_DAG_TWO_VARIABLES
    sample_sizes = config.SAMPLE_SIZES_EQUIDISTANT_SHORTER

    calculate_estimated_effect_and_squared_error(10, 5, "X-t2", "Y-t3", [], model)

    plot_means_rooted_errors_and_variances(
        sample_sizes, 12.15, "X-t2", "Y-t5", ["X-t1", "Y-t2"], 100, model
    )
    plot_means_rooted_errors_and_variances(
        sample_sizes, 12.15, "X-t2", "Y-t5", ["Y-t2"], 100, model
    )
    plot_means_rooted_errors_and_variances(
        sample_sizes, 12.15, "X-t2", "Y-t5", ["X-t1"], 100, model
    )

    model = MODEL_DAG_THREE_VARIABLES
    plot_means_rooted_errors_and_variances(
        sample_sizes,
        35,
        "X-t3",
        "Z-t5",
        ["X-t2", "Y-t3"],
        100,
        model,
    )

    plot_means_rooted_errors_and_variances(
        sample_sizes, 35, "X-t3", "Z-t5", ["Z-t4", "Y-t3"], 100, model
    )

    plot_means_rooted_errors_and_variances(
        sample_sizes, 35, "X-t3", "Z-t5", ["X-t2", "Z-t4", "Y-t3"], 100, model
    )
