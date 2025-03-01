import numpy as np


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    jacobian = np.zeros((len(x), len(x)))
    np.fill_diagonal(jacobian, (x > 0).astype(int))
    return jacobian


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    jacobian = np.zeros((len(x), len(x)))
    np.fill_diagonal(jacobian, 1 - np.tanh(x) ** 2)
    return jacobian


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax_derivative(x):
    softmax_vals = softmax(x)

    softmax_outer = np.outer(softmax_vals, softmax_vals)

    diagonal_softmax = np.diag(softmax_vals)

    jacobian = diagonal_softmax - softmax_outer

    return jacobian

