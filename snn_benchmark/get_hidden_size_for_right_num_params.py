import numpy as np


def get_number_of_parameters(ni, n_hidden_layers, no, h):
    """Get the number of parameters for a given number of hidden neurons.

    Args:
        ni (int): the number of input neurons
        n_hidden_layers (int): the number of hidden layers
        no (int): the number of output neurons
        h (int): the size of hidden layers

    Returns:
        int: the number of parameters
    """
    return (ni + 1) * h + h * (h + 1) * (n_hidden_layers - 1) + (h + 1) * no


def get_hidden_size_for_right_num_params_fn(ni, n_hidden_layers, no, p_target):
    """Get the size of hidden layers that will allows to obtain the right number of parameters.

    Args:
        ni (int): the number of input neurons
        k (int): the number of hidden layers
        no (int): the number of output neurons
        p_target (int): the number of parameters we want to obtain

    Returns:
        int: the size of hidden layers
    """
    a = n_hidden_layers - 1
    b = ni + n_hidden_layers + no
    c = no - p_target
    delta = b**2 - 4 * a * c
    return (-b + np.sqrt(delta)) / (2 * a)


k = 1000
M = k**2

if __name__ == "__main__":
    # IRIS
    ni = 4
    no = 3
    n_hidden_layers = 2
    for n_param in [100, k, 10 * k, 100 * k, M]:
        print(
            f"IRIS: for {n_param} parameters, use {get_hidden_size_for_right_num_params_fn(ni, n_hidden_layers, no, n_param)} hidden neurons"
        )

    # CIFAR10
    ni = 3 * 32 * 32
    no = 10
    n_hidden_layers = 4
    for n_param in [100, k, 10 * k, 100 * k, M, 10 * M]:
        print(
            f"CIFAR10: for {n_param} parameters, use {get_hidden_size_for_right_num_params_fn(ni, n_hidden_layers, no, n_param)} hidden neurons"
        )
