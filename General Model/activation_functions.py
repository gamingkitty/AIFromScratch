import numpy as np


class Activation:
    def __init__(self, function, derivative, is_elementwise=True):
        self.function = function
        self.derivative = derivative
        self.is_elementwise = is_elementwise

    def __call__(self, x):
        return self.function(x)

    def derivative(self, x):
        return self.derivative(x)


def f_relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return (x > 0).astype(int)


def f_tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


def f_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax_derivative(x):
    softmax_vals = softmax(x)

    softmax_outer = np.outer(softmax_vals, softmax_vals)

    diagonal_softmax = np.diag(softmax_vals)

    jacobian = diagonal_softmax - softmax_outer

    return jacobian

relu = Activation(f_relu, relu_derivative)
tanh = Activation(f_tanh, tanh_derivative)
softmax = Activation(f_softmax, softmax_derivative, False)
