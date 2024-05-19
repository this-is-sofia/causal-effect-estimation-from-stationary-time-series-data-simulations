import matplotlib.pyplot as plt

from models import MODEL_ADMG
from rooted_mean_squared_errors import (
    calculate_mean_variance_and_mean_squared_error,
)
import config
from plot_helpers import fix_node_names_for_title


def plot_means_and_rooted_errors_fixed_y_axis(
    list_of_sample_sizes: list,
    true_causal_effect: str,
    start_node: str,
    end_node: str,
    adjustment_set: str,
    repetitions: int,
    model,
):
    """
    Plot the means and rooted mean squared errors for a list of sample sizes with fixed y axis to see small differences in the ADMG case.
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
    fig, axs = plt.subplots(2)
    fig.suptitle(
        f"Means and Mean Squared Errors for effect from {fix_node_names_for_title(start_node)} to {fix_node_names_for_title(end_node)} \n with adjustment set {fix_node_names_for_title(adjustment_set)}"
    )
    axs[0].plot(list_of_sample_sizes, means, label="means", color="green")
    axs[0].set_ylim([4.5, 5.5])
    axs[0].legend(loc="upper right")
    axs[1].plot(list_of_sample_sizes, errors, label="MSE", color="blue")
    axs[1].set_ylim([0, 0.003])
    axs[1].legend(loc="upper right")
    name_string_with_dots = (
        "output/admg/"
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
    """
    Generate time series ADMG on variables X and Y (using Z as an unobservable variable).
    :return:
    """
    model = MODEL_ADMG
    sample_sizes = config.SAMPLE_SIZES_EQUIDISTANT_SHORTER

    plot_means_and_rooted_errors_fixed_y_axis(
        sample_sizes, 5, "X-t8", "Y-t9", ["X-t7"], 100, model
    )

    plot_means_and_rooted_errors_fixed_y_axis(
        sample_sizes, 5, "X-t8", "Y-t9", ["X-t7", "Y-t8"], 100, model
    )

    plot_means_and_rooted_errors_fixed_y_axis(
        sample_sizes, 5, "X-t8", "Y-t9", ["X-t7", "Y-t8", "X-t6", "Y-t7"], 100, model
    )

    plot_means_and_rooted_errors_fixed_y_axis(
        sample_sizes,
        5,
        "X-t8",
        "Y-t9",
        ["X-t7", "Y-t8", "X-t6", "Y-t7", "X-t5", "Y-t6"],
        100,
        model,
    )

    plot_means_and_rooted_errors_fixed_y_axis(
        sample_sizes,
        5,
        "X-t8",
        "Y-t9",
        ["X-t7", "Y-t8", "X-t6", "Y-t7", "X-t5", "Y-t6", "X-t4", "Y-t5"],
        100,
        model,
    )

