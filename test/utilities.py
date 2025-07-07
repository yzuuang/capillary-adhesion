import numpy as np
import matplotlib.pyplot as plt


def central_difference_jacobian(func, arg, step):
    """Return the jacobian computed numerically via central differnce scheme.
    
    It assumes the function is single argument, i.e. func(arg) yields th result.
    """
    return_shape = np.shape(func(arg))
    # Preprocess 'arg' so that it supports iterations
    arg_shape = np.shape(arg)
    if len(arg_shape) == 0:
        arg = np.array([arg])
        arg_shape = (1,)
    jacobian = np.full((*arg_shape, *return_shape), np.nan)
    # For each element
    for index in np.ndindex(*arg_shape):
        # Backup the original value
        original_value = arg[index].copy()

        # Get finite difference jacobian
        arg[index] = original_value + step
        value_plus = func(arg)
        arg[index] = original_value - step
        value_minus = func(arg)
        jacobian[index] = (value_plus - value_minus) / (2 * step)

        # Set the value back to origin
        arg[index] = original_value

    return jacobian


def plot_precisions(step_sizes, deviations):
    plt.plot(
        step_sizes,
        deviations,
        "x-",
        label=r"Difference from a numerical method of $\mathcal{O}(\delta^2)$",
    )

    plt.loglog()
    plt.xlabel(r"Step size $\delta$")
    plt.ylabel(r"Deviation $\varepsilon$")
    plt.legend()
