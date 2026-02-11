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


# Jacobian like [[z_0 -> a_0, z_1 -> a_0],
#                [z_0 -> a_1, z_1 -> a_1]]
def softmax_derivative(x):
    softmax_vals = softmax(x)

    softmax_outer = np.outer(softmax_vals, softmax_vals)

    diagonal_softmax = np.diag(softmax_vals)

    return diagonal_softmax - softmax_outer


def f_linear(x):
    return x


def f_derivative(x):
    return np.ones_like(x)


def softmax_vectorized(x):
    e_xs = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_xs / np.sum(e_xs, axis=1, keepdims=True)


def softmax_vectorized_derivative(x):
    softmax_vals = softmax_vectorized(x)
    n, m = softmax_vals.shape
    idx = np.arange(m)
    diagonals = np.zeros((n, m, m), dtype=softmax_vals.dtype)
    diagonals[:, idx, idx] = softmax_vals

    return diagonals - (softmax_vals[:, :, np.newaxis] * softmax_vals[:, np.newaxis, :])


def mse_loss(output, expected):
    return np.sum(np.power(expected - output, 2)) / output.shape[0]


def mse_loss_derivative(output, expected):
    return -(2 / output.shape[0]) * (expected - output)


def cross_entropy_loss(output, expected):
    output_clipped = np.clip(output, 1e-12, 1 - 1e-12)

    loss = -np.sum(expected * np.log(output_clipped)) / output_clipped.shape[0]
    return loss


def cross_entropy_loss_derivative(output, expected):
    output_clipped = np.clip(output, 1e-12, 1 - 1e-12)

    gradient = (-expected / output_clipped) / output_clipped.shape[0]
    return gradient


# Activation functions
relu = Activation(f_relu, relu_derivative)
tanh = Activation(f_tanh, tanh_derivative)
softmax = Activation(f_softmax, softmax_derivative, False)
vectorized_softmax = Activation(softmax_vectorized, softmax_vectorized_derivative, False)
linear = Activation(f_linear, f_derivative)

# Loss functions
mse = Loss(mse_loss, mse_loss_derivative)
cross_entropy = Loss(cross_entropy_loss, cross_entropy_loss_derivative)


def causal_mask(t):
    return np.triu(np.ones((t, t), dtype=bool))
