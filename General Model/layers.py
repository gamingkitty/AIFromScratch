import math
import numpy as np


def get_windows(window_shape, matrix):
    win_h, win_w = window_shape
    mat_h, mat_w = matrix.shape

    out_h = mat_h - win_h + 1
    out_w = mat_w - win_w + 1

    windows = np.empty((out_h * out_w, win_h, win_w), dtype=matrix.dtype)

    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            windows[idx] = matrix[i:i + win_h, j:j + win_w]
            idx += 1

    return windows


def get_windows_stride(window_shape, matrix, stride):
    win_h, win_w = window_shape
    mat_h, mat_w = matrix.shape

    out_h = mat_h - win_h + 1
    out_w = mat_w - win_w + 1

    windows = np.empty((out_h * out_w, win_h, win_w), dtype=matrix.dtype)

    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            windows[idx] = matrix[i:i + win_h, j:j + win_w]
            idx += 1

    return windows


class Dense:
    def __init__(self, neuron_num, activation_function, activation_function_derivative):
        self.neuron_num = neuron_num
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.weights = np.array([])
        self.gradient = np.array([])

    def init_weights(self, previous_layer_output_shape):
        self.weights = np.random.rand(np.prod(previous_layer_output_shape), self.neuron_num) - 0.5
        self.gradient = np.zeros_like(self.weights)

    def predict(self, prev_layer_activation):
        return self.activation_function(np.dot(prev_layer_activation.flatten(), self.weights))

    def forward_pass(self, prev_layer_activation):
        z_data = np.dot(prev_layer_activation.flatten(), self.weights)
        a_data = self.activation_function(z_data)

        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        prev_layer_a = prev_layer_a.flatten()
        # Need to return dc_da (error signal)
        # Can store gradient in the class
        dc_dz = np.dot(dc_da, self.activation_function_derivative(this_layer_z))
        self.gradient += dc_dz * prev_layer_a[:, np.newaxis]
        dc_da = np.dot(dc_dz, self.weights.T)
        return dc_da

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.gradient
        self.gradient = np.zeros_like(self.weights)

    def get_output_shape(self):
        return self.neuron_num, 1


class Convolution:
    def __init__(self, kernel_num, kernel_shape, activation_function, activation_function_derivative, input_shape=None):
        self.input_shape = input_shape
        self.input_num = 0
        self.kernel_shape = kernel_shape
        self.channel_kernel_shape = None
        self.kernel_size = None
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.kernels = None
        self.kernel_num = kernel_num
        self.output_shape = ()
        self.output_num = 0
        self.gradient = None
        self.gradient_size = None
        self.true_output_shape = None
        self.dz_da = None

    def init_weights(self, previous_layer_output_shape):
        self.channel_kernel_shape = (previous_layer_output_shape[0], *self.kernel_shape)
        self.kernel_size = np.prod(self.channel_kernel_shape)
        self.kernels = np.random.rand(self.kernel_num, *self.channel_kernel_shape) - 0.5
        self.gradient_size = self.kernel_size * self.kernel_num
        self.gradient = np.zeros(self.gradient_size)

        if self.input_shape is None:
            self.input_shape = previous_layer_output_shape

        self.input_num = np.prod(self.input_shape)
        self.output_shape = ((self.input_shape[1] - self.kernel_shape[0]) + 1, (self.input_shape[2] - self.kernel_shape[1]) + 1)
        self.true_output_shape = (self.kernel_num, *self.output_shape)
        self.output_num = np.prod(self.output_shape)

        for k in range(self.kernel_num):
            self.dz_da = np.zeros((self.output_num, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            for y in range(self.output_shape[0]):
                for x in range(self.output_shape[1]):
                    self.dz_da[y * self.output_shape[1] + x, :, y:y + self.kernel_shape[0], x:x + self.kernel_shape[1]] = self.kernels[k]
            self.dz_da = self.dz_da.reshape(self.dz_da.shape[0], -1)

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        prev_layer_activation = prev_layer_activation.reshape(self.input_shape)

        windows = np.transpose(np.array([get_windows(self.kernel_shape, channel) for channel in prev_layer_activation]), (1, 0, 2, 3))

        z_data = np.tensordot(windows, self.kernels, axes=([1, 2, 3], [1, 2, 3])).T.flatten()

        a_data = self.activation_function(z_data)
        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        gradient = np.zeros((self.kernel_num, self.kernel_size))

        dc_da = dc_da.reshape((self.kernel_num, self.output_num))
        this_layer_z = this_layer_z.reshape((self.kernel_num, self.output_num))
        new_dc_da = np.zeros(self.input_num)
        prev_layer_a = prev_layer_a.reshape(self.input_shape)

        # We don't need to calculate this in the loop because all kernels are the same size and have the same derivative wrt z.
        dz_dw = np.array([get_windows(self.output_shape, channel) for channel in prev_layer_a]).reshape(-1, self.output_num)

        for k in range(self.kernel_num):
            # Calculate dc_dz
            dc_dz = np.dot(dc_da[k], self.activation_function_derivative(this_layer_z[k]))

            # Calculate gradient
            gradient[k] = np.dot(dc_dz, dz_dw.T)

            # Calculate new dc_da
            new_dc_da += np.dot(dc_dz, self.dz_da)

        self.gradient += gradient.flatten()

        return new_dc_da

    def update_weights(self, learning_rate):
        self.kernels -= learning_rate * self.gradient.reshape((self.kernel_num, *self.channel_kernel_shape))
        self.gradient = np.zeros(self.gradient_size)

        for k in range(self.kernel_num):
            self.dz_da = np.zeros((self.output_num, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            for y in range(self.output_shape[0]):
                for x in range(self.output_shape[1]):
                    self.dz_da[y * self.output_shape[1] + x, :, y:y + self.kernel_shape[0], x:x + self.kernel_shape[1]] = self.kernels[k]
            self.dz_da = self.dz_da.reshape(self.dz_da.shape[0], -1)

    def get_output_shape(self):
        return self.true_output_shape


class MaxPooling:
    def __init__(self, kernel_shape, stride, input_shape=None):
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.input_shape = input_shape
        self.output_shape = None
        self.output_num = None

    def init_weights(self, previous_layer_output_shape):
        if self.input_shape is None:
            self.input_shape = previous_layer_output_shape
        self.output_shape = (self.input_shape[0], math.ceil(self.input_shape[1] / self.stride), math.ceil(self.input_shape[2] / self.stride))
        self.output_num = np.prod(self.output_shape)

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        prev_layer_activation = prev_layer_activation.reshape(self.input_shape)

        a_data = np.zeros(self.output_shape)

        for y in range(self.output_shape[1]):
            for x in range(self.output_shape[2]):
                stride_y = y * self.stride
                stride_x = x * self.stride
                lower_bounds = (stride_y, stride_x)
                upper_bounds = (min(stride_y + self.kernel_shape[0], prev_layer_activation.shape[1]), min(stride_x + self.kernel_shape[1], prev_layer_activation.shape[2]))
                for c in range(self.output_shape[0]):
                    a_data[c, y, x] += np.max(prev_layer_activation[c, lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]])

        return a_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        prev_layer_a = prev_layer_a.reshape(self.input_shape)

        new_dc_da = np.zeros(self.input_shape)
        dc_da = dc_da.reshape(self.output_shape)
        for y in range(self.output_shape[1]):
            for x in range(self.output_shape[2]):
                stride_y = y * self.stride
                stride_x = x * self.stride
                lower_bounds = (stride_y, stride_x)
                upper_bounds = (min(stride_y + self.kernel_shape[0], prev_layer_a.shape[1]), min(stride_x + self.kernel_shape[1], prev_layer_a.shape[2]))
                for c in range(self.output_shape[0]):
                    window = prev_layer_a[c, lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]]
                    max_index = np.unravel_index(np.argmax(window), window.shape)
                    new_dc_da[c, stride_y + max_index[0], stride_x + max_index[1]] += dc_da[c, y, x]

        return new_dc_da.flatten()

    def update_weights(self, learning_rate):
        pass

    def get_output_shape(self):
        return self.output_shape


class Embedding:
    def __init__(self, embedding_dimension, activation_function, activation_function_derivative, input_shape=None):
        self.neuron_num = embedding_dimension
        self.input_shape = input_shape
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.weights = np.array([])
        self.gradient = np.array([])

    def init_weights(self, previous_layer_output_shape):
        if self.input_shape is None:
            self.input_shape = previous_layer_output_shape
        self.weights = np.random.rand(self.input_shape[1], self.neuron_num) - 0.5
        self.gradient = np.zeros_like(self.weights)

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        prev_layer_activation = prev_layer_activation.reshape(self.input_shape)
        z_data = np.sum(np.dot(prev_layer_activation, self.weights), axis=0)
        a_data = self.activation_function(z_data)

        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        # Need to return dc_da (error signal)
        # Can store gradient in the class
        dc_dz = np.dot(dc_da, self.activation_function_derivative(this_layer_z))
        self.gradient += dc_dz * np.sum(prev_layer_a.reshape(self.input_shape), axis=0)[:, np.newaxis]
        dc_da = np.dot(dc_dz, self.weights.T)
        return dc_da

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.gradient
        self.gradient = np.zeros_like(self.weights)

    def get_output_shape(self):
        return self.neuron_num, 1


class Dropout:
    def __init__(self, dropout_percent):
        self.dropout_percent = dropout_percent
        self.input_shape = None
        self.input_num = None
        self.dz_da = None

    def init_weights(self, previous_layer_output_shape):
        self.input_shape = previous_layer_output_shape
        self.input_num = np.prod(self.input_shape)

    def predict(self, prev_layer_activation):
        return prev_layer_activation

    def forward_pass(self, prev_layer_activation):
        mask = np.random.rand(*prev_layer_activation.shape) > self.dropout_percent
        self.dz_da = mask.astype(float)
        a_data = prev_layer_activation * self.dz_da
        return a_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        return dc_da * self.dz_da.flatten()

    def update_weights(self, learning_rate):
        pass

    def get_output_shape(self):
        return self.input_shape
