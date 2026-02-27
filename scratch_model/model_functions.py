import numpy as np


class Activation:
    def __init__(self, function, derivative, is_elementwise=True):
        self.function = function
        self.derivative_fn = derivative
        self.is_elementwise = is_elementwise

    def __call__(self, x):
        return self.function(x)

    def derivative(self, x):
        return self.derivative_fn(x)


class Loss:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative_fn = derivative

    def __call__(self, output, expected):
        return self.function(output, expected)

    def derivative(self, output, expected):
        return self.derivative_fn(output, expected)


def f_relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return (x > 0).astype(int)


def f_tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


def f_softmax(x):
    e_xs = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_xs / np.sum(e_xs, axis=1, keepdims=True)


# Jacobian like [[z_0 -> a_0, z_1 -> a_0],
#                [z_0 -> a_1, z_1 -> a_1]]
def softmax_derivative(x):
    softmax_vals = softmax_vectorized(x)
    n, m = softmax_vals.shape
    idx = np.arange(m)
    diagonals = np.zeros((n, m, m), dtype=softmax_vals.dtype)
    diagonals[:, idx, idx] = softmax_vals

    return diagonals - (softmax_vals[:, :, np.newaxis] * softmax_vals[:, np.newaxis, :])


def f_linear(x):
    return x


def f_derivative(x):
    return np.ones_like(x)


def softmax_vectorized(x):
    e_xs = np.exp(x - np.max(x, axis=2, keepdims=True))
    return e_xs / np.sum(e_xs, axis=2, keepdims=True)


def softmax_vectorized_derivative(x):
    softmax_vals = softmax_vectorized(x)
    b, n, m = softmax_vals.shape
    idx = np.arange(m)
    diagonals = np.zeros((b, n, m, m), dtype=softmax_vals.dtype)
    diagonals[:, :, idx, idx] = softmax_vals

    # h, n, m = attention_scores.shape
    # idx = np.arange(m)
    # diagonals = np.zeros((h, n, m, m), dtype=attention_scores.dtype)
    # diagonals[:, :, idx, idx] = attention_scores
    #
    # dattention_draw = diagonals - (attention_scores[:, :, :, np.newaxis] * attention_scores[:, :, np.newaxis, :])

    return diagonals - (softmax_vals[:, :, :, np.newaxis] * softmax_vals[:, :, np.newaxis, :])


def f_gelu(x):
    a = np.sqrt(2.0 / np.pi)
    u = a * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(u))


def gelu_derivative(x):
    a = np.sqrt(2.0 / np.pi)
    u = a * (x + 0.044715 * x**3)
    t = np.tanh(u)

    u_prime = a * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * u_prime


def mse_loss(output, expected):
    return np.sum(np.power(expected - output, 2)) / output.shape[1]


def mse_loss_derivative(output, expected):
    return -(2 / output.shape[1]) * (expected - output)

def vectorized_cross_entropy_loss(output, expected):
    output_clipped = np.clip(output, 1e-12, 1 - 1e-12)

    # 1 is the time dimension
    loss = -np.sum(expected * np.log(output_clipped)) / output_clipped.shape[1]
    return loss

def vectorized_softmax_cross_entropy_derivative(output, expected):
    return (output - expected) / output.shape[1]


def cross_entropy_loss(output, expected):
    output_clipped = np.clip(output, 1e-12, 1 - 1e-12)

    loss = -np.sum(expected * np.log(output_clipped))
    return loss

def softmax_cross_entropy_loss_derivative(output, expected):
    return output - expected


# Activation functions
relu = Activation(f_relu, relu_derivative)
tanh = Activation(f_tanh, tanh_derivative)

softmax = Activation(f_softmax, softmax_derivative, False)
vectorized_softmax = Activation(softmax_vectorized, softmax_vectorized_derivative, False)

linear = Activation(f_linear, f_derivative)
gelu = Activation(f_gelu, gelu_derivative)

# Loss functions
mse = Loss(mse_loss, mse_loss_derivative)

cross_entropy_softmax = Activation(f_softmax, f_derivative)
vectorized_cross_entropy_softmax = Activation(softmax_vectorized, f_derivative)
softmax_cross_entropy = Loss(cross_entropy_loss, softmax_cross_entropy_loss_derivative)
vectorized_softmax_cross_entropy = Loss(vectorized_cross_entropy_loss, vectorized_softmax_cross_entropy_derivative)


def causal_mask(t):
    return np.tril(np.ones((t, t), dtype=bool))

