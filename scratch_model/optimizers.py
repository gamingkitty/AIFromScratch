from ._gpu_patch import xp as np


class Adam:
    def __init__(self, b1=0.9, b2=0.999):
        self.b1 = b1
        self.b2 = b2

        self.b1_pow = 1
        self.b2_pow = 1

        self.gradient_square = None

        self.m = None
        self.v = None

        self.weights = None

        self.epsilon = 1e-8

    def initialize(self, weights, dtype=np.float32):
        self.m = np.zeros(weights.shape, dtype=dtype)
        self.v = np.zeros(weights.shape, dtype=dtype)
        self.gradient_square = np.zeros(weights.shape, dtype=dtype)
        self.weights = weights
        self.b1_pow = 1
        self.b2_pow = 1

    def update_weights(self, gradient, learning_rate):
        self.m *= self.b1
        self.m += (1.0 - self.b1) * gradient

        np.square(gradient, out=self.gradient_square)
        self.v *= self.b2
        self.v += (1.0 - self.b2) * self.gradient_square

        self.b1_pow *= self.b1
        self.b2_pow *= self.b2

        m_hat = self.m / (1 - self.b1_pow)
        v_hat = self.v / (1 - self.b2_pow)
        self.weights -= learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))


class AdamW:
    def __init__(self, b1=0.9, b2=0.999, weight_decay=0.01):
        self.adam = Adam(b1, b2)
        self.weight_decay = weight_decay
        self.weights = None

    def initialize(self, weights, dtype=np.float32):
        self.weights = weights
        self.adam.initialize(weights, dtype=dtype)

    def update_weights(self, gradient, learning_rate):
        self.weights *= 1 - learning_rate * self.weight_decay
        self.adam.update_weights(gradient, learning_rate)


class GradientDescent:
    def __init__(self):
        self.weights = None

    def initialize(self, weights, dtype=np.float32):
        self.weights = weights

    def update_weights(self, gradient, learning_rate):
        self.weights -= learning_rate * gradient
