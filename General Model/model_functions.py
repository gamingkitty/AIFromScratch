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

class Loss:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

    def __call__(self, output, expected):
        return self.function(output, expected)

    def derivative(self, output, expected):
        return self.derivative(output, expected)


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


def mse_loss(output, expected):
    return np.sum(np.power(expected - output, 2)) / output.shape[0]


def mse_loss_derivative(output, expected):
    return -(2 / output.shape[0]) * (expected - output)


def categorical_entropy_loss(output, expected):
    output_clipped = np.clip(output, 1e-12, 1 - 1e-12)

    loss = -np.sum(expected * np.log(output_clipped))
    return loss


def categorical_entropy_loss_derivative(output, expected):
    output_clipped = np.clip(output, 1e-12, 1 - 1e-12)

    gradient = -expected / output_clipped
    return gradient

# Activation functions
relu = Activation(f_relu, relu_derivative)
tanh = Activation(f_tanh, tanh_derivative)
softmax = Activation(f_softmax, softmax_derivative, False)

# Loss functions
mse = Loss(mse_loss, mse_loss_derivative)
categorical_entropy = Loss(categorical_entropy_loss, categorical_entropy_loss_derivative)