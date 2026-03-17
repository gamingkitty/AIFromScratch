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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative_from_output(y):
    return y * (1 - y)


def softmax_axis(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def yolo_out_softmax_classes(x):
    out = x.copy()

    out[..., 0:3] = sigmoid(out[..., 0:3])
    out[..., 5:] = softmax_axis(out[..., 5:], axis=-1)

    return out


def yolo_out_softmax_classes_derivative(output):
    derivative = np.ones_like(output)

    derivative[..., 0:3] = sigmoid_derivative_from_output(output[..., 0:3])

    return derivative


# Lambda parameters define scale of individual loss components
# Input expected: (Batch, Spatial, Spatial, Anchors, (Objectiveness, x, y, w, h, classes))
def yolo_loss_softmax_classes(output, expected, lambda_obj=1.0, lambda_noobj=0.1, lambda_box=5.0, lambda_class=1.0):
    eps = 1e-12
    output_clipped = np.clip(output, eps, 1 - eps)

    obj_target = expected[..., 0]
    obj_pred = output_clipped[..., 0]

    pos_mask = (obj_target == 1).astype(output.dtype)
    neg_mask = (obj_target == 0).astype(output.dtype)

    obj_loss = -np.sum(
        pos_mask * np.log(obj_pred)
    )

    noobj_loss = -np.sum(
        neg_mask * np.log(1 - obj_pred)
    )

    # MSE for x, y, w, h where there is an object
    box_loss = np.sum(
        pos_mask[..., None] * (output[..., 1:5] - expected[..., 1:5]) ** 2
    )

    # Class cross entropy only where object exists
    class_pred = np.clip(output[..., 5:], eps, 1 - eps)
    class_target = expected[..., 5:]

    class_loss = -np.sum(
        pos_mask[..., None] * class_target * np.log(class_pred)
    )

    return lambda_obj * obj_loss + lambda_noobj * noobj_loss + lambda_box * box_loss + lambda_class * class_loss


def yolo_loss_softmax_classes_derivative(output, expected, lambda_obj=1.0, lambda_noobj=0.1, lambda_box=5.0, lambda_class=1.0):
    eps = 1e-12
    output_clipped = np.clip(output, eps, 1 - eps)

    grad = np.zeros_like(output)

    obj_target = expected[..., 0]
    obj_pred = output_clipped[..., 0]

    pos_mask = (obj_target == 1).astype(output.dtype)
    neg_mask = (obj_target == 0).astype(output.dtype)

    # Objectness derivatives
    grad[..., 0] += lambda_obj * pos_mask * (-1 / obj_pred)
    grad[..., 0] += lambda_noobj * neg_mask * (1 / (1 - obj_pred))

    # MSE derivative for x, y, w, h where there is an object
    grad[..., 1:5] = (
        lambda_box * 2 * pos_mask[..., None] * (output[..., 1:5] - expected[..., 1:5])
    )

    # Only account for class loss in spatial positions with an object in them
    grad[..., 5:] = lambda_class * pos_mask[..., None] * (output[..., 5:] - expected[..., 5:])

    return grad


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

yolo_activation = Activation(yolo_out_softmax_classes, yolo_out_softmax_classes_derivative)
yolo_loss = Loss(yolo_loss_softmax_classes, yolo_loss_softmax_classes_derivative)


def causal_mask(t):
    return np.tril(np.ones((t, t), dtype=bool))

