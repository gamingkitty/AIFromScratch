import numpy as np


class AdamW:
    def __init__(self, b1=0.9, b2=0.999, weight_decay=0.01):
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay

        self.b1_pow = 1
        self.b2_pow = 1

        self.gradient_square = None

        self.m = None
        self.v = None

        self.weights = None

        self.epsilon = 1e-8

        self.clip = 1

    def initialize(self, weights, dtype=np.float32):
        self.m = np.zeros(weights.shape, dtype=dtype)
        self.v = np.zeros(weights.shape, dtype=dtype)
        self.gradient_square = np.zeros(weights.shape, dtype=dtype)
        self.weights = weights
        self.b1_pow = 1
        self.b2_pow = 1

    def update_weights(self, gradient, learning_rate):
        g_norm = np.linalg.norm(gradient)
        if g_norm > self.clip:
            gradient = gradient * (self.clip / (g_norm + 1e-12))

        self.m *= self.b1
        self.m += (1.0 - self.b1) * gradient

        np.square(gradient, out=self.gradient_square)
        self.v *= self.b2
        self.v += (1.0 - self.b2) * self.gradient_square

        self.b1_pow *= self.b1
        self.b2_pow *= self.b2

        m_hat = self.m / (1 - self.b1_pow)
        v_hat = self.v / (1 - self.b2_pow)
        self.weights *= 1 - learning_rate * self.weight_decay
        self.weights -= learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))


class GradientDescent:
    def __init__(self):
        pass

    def initialize(self, gradient_shape, dtype=np.float32):
        pass

    def get_weight_update(self, gradient, learning_rate):
        return -learning_rate * gradient
