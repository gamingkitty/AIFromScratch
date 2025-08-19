import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.lib.stride_tricks import sliding_window_view
import time


total_convolution_time = 0
total_pooling_time = 0
dc_da_time = 0
dz_da_time = 0

times = 0

recurrent_time = 0
loop_time = 0


# def get_windows(window_shape, matrix):
#     win_h, win_w = window_shape
#     out_c, mat_h, mat_w = matrix.shape
#     out_h = mat_h - win_h + 1
#     out_w = mat_w - win_w + 1
#
#     # Strides for moving through matrix
#     stride_c, stride_h, stride_w = matrix.strides
#
#     # Shape of output: (channels, out_h, out_w, win_h, win_w)
#     shape = (out_c, out_h, out_w, win_h, win_w)
#     strides = (stride_c, stride_h, stride_w, stride_h, stride_w)
#
#     windows = as_strided(matrix, shape=shape, strides=strides).reshape(out_c, out_h * out_h, win_h, win_w)
#     return windows


def get_windows(window_shape, matrix):
    win_h, win_w = window_shape
    mat_h, mat_w = matrix.shape[1], matrix.shape[2]

    out_c = matrix.shape[0]
    out_h = mat_h - win_h + 1
    out_w = mat_w - win_w + 1
    windows = np.empty((out_c, out_h * out_w, win_h, win_w), dtype=matrix.dtype)
    for c in range(out_c):
        idx = 0
        for i in range(out_h):
            for j in range(out_w):
                windows[c, idx] = matrix[c, i:i + win_h, j:j + win_w]
                idx += 1
    return windows


class Dense:
    def __init__(self, neuron_num, activation_function):
        self.neuron_num = neuron_num
        self.activation_function = activation_function
        self.weights = None
        self.gradient = None

    def init_weights(self, previous_layer_output_shape):
        in_num = np.prod(previous_layer_output_shape)
        out_num = self.neuron_num
        weight_limit = math.sqrt(6 / (in_num + out_num))
        self.weights = np.random.uniform(-weight_limit, weight_limit, size=(in_num + 1, out_num))
        self.gradient = np.zeros_like(self.weights)

    def predict(self, prev_layer_activation):
        return self.activation_function(np.dot(np.append(prev_layer_activation.flatten(), 1), self.weights))

    def forward_pass(self, prev_layer_activation):
        z_data = np.dot(np.append(prev_layer_activation.flatten(), 1), self.weights)
        a_data = self.activation_function(z_data)

        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        prev_layer_a = np.append(prev_layer_a.flatten(), 1)
        if self.activation_function.is_elementwise:
            dc_dz = dc_da * self.activation_function.derivative(this_layer_z)
        else:
            dc_dz = np.dot(dc_da, self.activation_function.derivative(this_layer_z))
        self.gradient += dc_dz * prev_layer_a[:, np.newaxis]
        # Exclude bias neuron from prev layer because it is not an output of that layer.
        dc_da = np.dot(dc_dz, self.weights[:-1].T)
        return dc_da

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.gradient
        self.gradient = np.zeros_like(self.weights)

    def get_output_shape(self):
        return self.neuron_num, 1

    def count_params(self):
        return np.prod(self.weights.shape)


class Convolution:
    def __init__(self, kernel_num, kernel_shape, activation_function, input_shape=None):
        self.input_shape = input_shape
        self.input_num = 0
        self.kernel_shape = kernel_shape
        self.channel_kernel_shape = None
        self.kernel_size = None
        self.activation_function = activation_function
        self.kernels = None
        self.kernel_num = kernel_num
        self.biases = None
        self.output_shape = ()
        self.output_num = 0
        self.gradient = None
        self.gradient_size = None
        self.bias_gradient = None
        self.true_output_shape = None
        self.dz_da = None

    def init_weights(self, previous_layer_output_shape):
        self.channel_kernel_shape = (previous_layer_output_shape[0], *self.kernel_shape)
        self.kernel_size = np.prod(self.channel_kernel_shape)

        if self.input_shape is None:
            self.input_shape = previous_layer_output_shape

        self.input_num = np.prod(self.input_shape)
        self.output_shape = ((self.input_shape[1] - self.kernel_shape[0]) + 1, (self.input_shape[2] - self.kernel_shape[1]) + 1)
        self.true_output_shape = (self.kernel_num, *self.output_shape)
        self.output_num = np.prod(self.output_shape)

        in_num = np.prod(self.channel_kernel_shape)
        weight_limit = math.sqrt(2 / in_num)
        self.kernels = np.random.uniform(-weight_limit, weight_limit, size=(self.kernel_num, *self.channel_kernel_shape))
        self.biases = np.zeros(self.kernel_num)

        self.gradient_size = self.kernel_size * self.kernel_num
        self.gradient = np.zeros(self.gradient_size)
        self.bias_gradient = np.zeros(self.kernel_num)

        self.dz_da = np.zeros((self.kernel_num, self.output_num, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        for k in range(self.kernel_num):
            for y in range(self.output_shape[0]):
                for x in range(self.output_shape[1]):
                    self.dz_da[k, y * self.output_shape[1] + x, :, y:y + self.kernel_shape[0], x:x + self.kernel_shape[1]] = self.kernels[k]
        self.dz_da = self.dz_da.reshape(self.kernel_num, self.output_num, -1)

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        global total_convolution_time
        t0 = time.perf_counter()
        prev_layer_activation = prev_layer_activation.reshape(self.input_shape)

        # Transpose so that its (window_num, channel_num, k_h, k_w) this way it matches the kernels and math can be done on it easily.
        windows = np.transpose(get_windows(self.kernel_shape, prev_layer_activation), (1, 0, 2, 3))

        z_data = (np.tensordot(windows, self.kernels, axes=([1, 2, 3], [1, 2, 3])).T + self.biases[:, np.newaxis]).flatten()

        a_data = self.activation_function(z_data)
        total_convolution_time += time.perf_counter() - t0
        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        global total_convolution_time, dc_da_time
        t0 = time.perf_counter()
        dc_da = dc_da.reshape((self.kernel_num, self.output_num))
        this_layer_z = this_layer_z.reshape((self.kernel_num, self.output_num))
        prev_layer_a = prev_layer_a.reshape(self.input_shape)

        # Is the same for every kernel, so we don't need to calculate for every kernel.
        dz_dw = get_windows(self.kernel_shape, prev_layer_a).reshape(-1, self.output_num)

        activation_derivative = self.activation_function.derivative(this_layer_z)

        # dc_da is (kernel_num, output_num)
        # activation_derivative is (kernel_num, output_num, output_num) or (kernel_num, output_num)
        if self.activation_function.is_elementwise:
            dc_dz = dc_da * activation_derivative
        else:
            dc_dz = np.einsum('ko,koo->ko', dc_da, activation_derivative)

        # dz_dw is (kernel_size, output_num)
        # dc_dz is (kernel_num, output_num)
        # output is kernel_num, kernel_size
        self.gradient += np.tensordot(dc_dz, dz_dw, axes=([1], [1])).flatten()

        self.bias_gradient += np.sum(dc_dz, axis=1)

        # Calculate new dc_da
        # dc_dz is (kernel_num, output_num)
        # dz_da is (kernel_num, output_num, input_num)
        t1 = time.perf_counter()
        dc_da = np.dot(dc_dz.reshape(-1), self.dz_da.reshape(-1, self.input_num))
        dc_da_time += time.perf_counter() - t1

        total_convolution_time += time.perf_counter() - t0

        return dc_da.flatten()

    def update_weights(self, learning_rate):
        global dz_da_time
        # Divide by output num because to get the gradient you have to sum over the affects of each weight on every output in a specific channel.
        self.kernels -= learning_rate * self.gradient.reshape((self.kernel_num, *self.channel_kernel_shape)) / self.output_num
        self.biases -= learning_rate * self.bias_gradient / self.output_num
        self.bias_gradient = np.zeros(self.kernel_num)
        self.gradient = np.zeros(self.gradient_size)

        # dz_da will be (kernel_num, output_num, input_num) because it tracks how the input affects
        # the output, and output is (kernel_num, output_num)
        # Very slow right now, should look for speed up
        t0 = time.perf_counter()
        self.dz_da = np.zeros((self.kernel_num, self.output_num, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        for k in range(self.kernel_num):
            for y in range(self.output_shape[0]):
                for x in range(self.output_shape[1]):
                    self.dz_da[k, y * self.output_shape[1] + x, :, y:y + self.kernel_shape[0], x:x + self.kernel_shape[1]] = self.kernels[k]
        self.dz_da = self.dz_da.reshape(self.kernel_num, self.output_num, -1)
        dz_da_time += time.perf_counter() - t0

    def get_output_shape(self):
        return self.true_output_shape

    def count_params(self):
        return np.prod(self.kernels.shape) + np.prod(self.biases.shape)


# Very slow currently
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
        self.output_shape = (self.input_shape[0], math.ceil(self.input_shape[1] / self.stride) - math.ceil(self.kernel_shape[0] / self.stride) + 1,
                                                  math.ceil(self.input_shape[2] / self.stride) - math.ceil(self.kernel_shape[1] / self.stride) + 1)
        self.output_num = np.prod(self.output_shape)

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        global total_pooling_time
        t0 = time.perf_counter()
        prev_layer_activation = prev_layer_activation.reshape(self.input_shape)

        a_data = np.zeros(self.output_shape)
        windows = np.zeros((self.output_num, *self.kernel_shape))

        idx = 0
        for y in range(self.output_shape[1]):
            for x in range(self.output_shape[2]):
                stride_y = y * self.stride
                stride_x = x * self.stride
                lower_bounds = (stride_y, stride_x)
                upper_bounds = (min(stride_y + self.kernel_shape[0], prev_layer_activation.shape[1]), min(stride_x + self.kernel_shape[1], prev_layer_activation.shape[2]))
                for c in range(self.output_shape[0]):
                    window = prev_layer_activation[c, lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]]
                    windows[idx] = window
                    a_data[c, y, x] += np.max(window)
                    idx += 1

        total_pooling_time += time.perf_counter() - t0
        return windows, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        global total_pooling_time
        t0 = time.perf_counter()

        new_dc_da = np.zeros(self.input_shape)
        dc_da = dc_da.reshape(self.output_shape)
        idx = 0
        for y in range(self.output_shape[1]):
            for x in range(self.output_shape[2]):
                stride_y = y * self.stride
                stride_x = x * self.stride
                # lower_bounds = (stride_y, stride_x)
                # upper_bounds = (min(stride_y + self.kernel_shape[0], prev_layer_a.shape[1]), min(stride_x + self.kernel_shape[1], prev_layer_a.shape[2]))
                for c in range(self.output_shape[0]):
                    # window = prev_layer_a[c, lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]]
                    window = this_layer_z[idx]
                    max_index = np.unravel_index(np.argmax(window), window.shape)
                    new_dc_da[c, stride_y + max_index[0], stride_x + max_index[1]] += dc_da[c, y, x]
                    idx += 1

        total_pooling_time += time.perf_counter() - t0
        return new_dc_da.flatten()

    def update_weights(self, learning_rate):
        pass

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return 0


# Assumes input is something one hot encoded.
class Embedding:
    def __init__(self, embedding_dimension, vocab_size, activation_function, input_shape=None):
        self.neuron_num = embedding_dimension
        self.vocab_size = vocab_size
        self.input_shape = input_shape
        self.activation_function = activation_function
        self.weights = np.array([])
        self.gradient = np.array([])
        self.output_shape = None

    def init_weights(self, previous_layer_output_shape):
        if self.input_shape is None:
            self.input_shape = previous_layer_output_shape
        in_num = np.prod(self.input_shape)
        out_num = self.neuron_num
        weight_limit = math.sqrt(6 / (in_num + out_num))
        self.weights = np.random.uniform(-weight_limit, weight_limit, size=(self.vocab_size, self.neuron_num))
        self.gradient = np.zeros_like(self.weights)

        self.output_shape = (*self.input_shape, self.neuron_num)

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        prev_layer_activation = prev_layer_activation.reshape(self.input_shape)
        z_data = self.weights[prev_layer_activation]
        a_data = self.activation_function(z_data)

        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        dc_da = dc_da.reshape(self.output_shape)
        if self.activation_function.is_elementwise:
            dc_dz = dc_da * self.activation_function.derivative(this_layer_z)
        else:
            dc_dz = np.dot(dc_da, self.activation_function.derivative(this_layer_z))

        # self.gradient[prev_layer_a.reshape(self.input_shape)] += dc_dz
        np.add.at(self.gradient, prev_layer_a.flatten(), dc_dz.reshape(-1, self.neuron_num))
        # self.gradient[prev_layer_a] += dc_dz

        dc_da = np.dot(dc_dz, self.weights.T)
        return dc_da

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.gradient
        self.gradient = np.zeros_like(self.weights)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return np.prod(self.weights.shape)


# Change so that doesnt store self.dz_da
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

    def count_params(self):
        return 0


# Currently returns entire sequence. Might add option in future to only return final output of layer.
class Recurrent:
    def __init__(self, neuron_num, activation_function, input_shape=None):
        self.neuron_num = neuron_num
        self.activation_function = activation_function

        self.hidden_weights = None
        self.hidden_gradient = None
        self.input_weights = None
        self.input_gradient = None

        # Input shape will be something like (-1, n) because size can vary based on number of tokens passed in
        self.input_shape = input_shape
        self.output_shape = None

        self.this_layer_a = None

    def init_weights(self, previous_layer_output_shape):
        if self.input_shape is None:
            self.input_shape = previous_layer_output_shape

        self.output_shape = (self.input_shape[0], self.neuron_num)

        in_num = np.prod(previous_layer_output_shape[1:])
        out_num = self.neuron_num
        weight_limit = math.sqrt(6 / (in_num + out_num))

        self.hidden_weights = np.random.uniform(-weight_limit, weight_limit, size=(out_num, out_num))
        self.hidden_gradient = np.zeros_like(self.hidden_weights)
        self.input_weights = np.random.uniform(-weight_limit, weight_limit, size=(in_num, out_num))
        self.input_gradient = np.zeros_like(self.input_weights)

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        before_activation = np.dot(prev_layer_activation[0].flatten(), self.input_weights)
        last_output = self.activation_function(before_activation)

        z_states = [before_activation]
        a_states = [last_output]

        for token in prev_layer_activation[1:]:
            before_activation = np.dot(last_output, self.hidden_weights) + np.dot(token.flatten(), self.input_weights)
            z_states.append(before_activation)
            last_output = self.activation_function(before_activation)
            a_states.append(last_output)

        self.this_layer_a = np.array(a_states)
        return np.array(z_states), self.this_layer_a

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        global recurrent_time
        start_time = time.perf_counter()
        dc_da = dc_da.reshape(self.output_shape)
        this_layer_z = this_layer_z.reshape(self.output_shape)
        prev_layer_a = prev_layer_a.reshape(self.input_shape)

        activation_derivatives = [self.activation_function.derivative(z) for z in this_layer_z]

        # Calculate new dc_da
        # Go backwards through time to determine dc_dh at every timestep (not just dc_da because h affects later hs too)
        # then use dc_dh to find dc_da
        dc_dh = dc_da[-1]
        dc_dh_list = [dc_dh]
        da_dz = activation_derivatives[-1]
        if self.activation_function.is_elementwise:
            dc_dz = dc_dh * da_dz
        else:
            dc_dz = np.dot(dc_dh, da_dz)
        new_dc_da = [np.dot(dc_dz, self.input_weights.T)]
        for t in reversed(range(len(this_layer_z) - 1)):
            dc_dh = np.dot(dc_dz, self.hidden_weights.T) + dc_da[t]
            dc_dh_list.append(dc_dh)
            da_dz = activation_derivatives[t]
            if self.activation_function.is_elementwise:
                dc_dz = dc_dh * da_dz
            else:
                dc_dz = np.dot(dc_dh, da_dz)
            new_dc_da.append(np.dot(dc_dz, self.input_weights.T))

        new_dc_da = np.array(new_dc_da[::-1]).flatten()
        dc_dh_list = np.array(dc_dh_list[::-1])

        da_dw_hidden = np.zeros((self.neuron_num, self.neuron_num, self.neuron_num))
        dc_dw_hidden_sum = np.zeros((self.neuron_num, self.neuron_num))

        # Initialize dc_dw_input_sum and da_dw_input for t = 0 which isn't covered by the loop
        da_dw_input = np.zeros((self.input_shape[1], self.neuron_num, self.neuron_num))
        diag_indices = np.arange(self.neuron_num)
        da_dw_input[np.arange(self.input_shape[1])[:, None], diag_indices, diag_indices] = prev_layer_a[0][:, None]
        if self.activation_function.is_elementwise:
            da_dw_input *= activation_derivatives[0]
        else:
            da_dw_input = np.tensordot(da_dw_input, activation_derivatives[0].T, axes=[2, 0])
        dc_dw_input_sum = np.sum(da_dw_input * dc_dh_list[0], axis=2)

        for t in range(1, len(this_layer_z)):
            # Calculate da_dw_hidden and dc_dw_hidden_sum
            da_dz = activation_derivatives[t]
            dz_dw_hidden = np.tensordot(da_dw_hidden, self.hidden_weights, axes=[2, 0])
            diag_indices = np.arange(self.neuron_num)
            dz_dw_hidden[diag_indices[:, None], diag_indices, diag_indices] += self.this_layer_a[t - 1][:, None]
            if self.activation_function.is_elementwise:
                da_dw_hidden = da_dz * dz_dw_hidden
            else:
                da_dw_hidden = np.tensordot(dz_dw_hidden, da_dz.T, axes=[2, 0])

            dc_dw_hidden_sum += np.sum(da_dw_hidden * dc_dh_list[t], axis=2)

            # Calculate da_dw_input
            dz_dw_input = np.tensordot(da_dw_input, self.hidden_weights, axes=[2, 0])
            diag_indices = np.arange(self.neuron_num)
            dz_dw_input[np.arange(self.input_shape[1])[:, None], diag_indices, diag_indices] += prev_layer_a[t][:, None]
            if self.activation_function.is_elementwise:
                da_dw_input = da_dz * dz_dw_input
            else:
                da_dw_input = np.tensordot(dz_dw_input, da_dz.T, axes=[2, 0])

            dc_dw_input_sum += np.sum(da_dw_input * dc_dh_list[t], axis=2)

        # Divide by number of time steps to reduce spikes in gradient
        self.hidden_gradient += dc_dw_hidden_sum / len(this_layer_z)
        self.input_gradient += dc_dw_input_sum / len(this_layer_z)

        recurrent_time += time.perf_counter() - start_time

        return new_dc_da

    def update_weights(self, learning_rate):
        self.input_weights -= learning_rate * self.input_gradient
        self.hidden_weights -= learning_rate * self.hidden_gradient

        self.input_gradient = np.zeros_like(self.input_weights)
        self.hidden_gradient = np.zeros_like(self.hidden_weights)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return np.prod(self.input_weights.shape) + np.prod(self.hidden_weights.shape)


class Loop:
    def __init__(self, *layers):
        self.layers = layers
        self.output_shape = None
        self.z_data = None
        self.a_data = None

    def init_weights(self, previous_layer_output_shape):
        loop_num = previous_layer_output_shape[0]
        previous_layer_output_shape = previous_layer_output_shape[1:]
        for layer in self.layers:
            layer.init_weights(previous_layer_output_shape)
            previous_layer_output_shape = layer.get_output_shape()
        self.output_shape = (loop_num, *self.layers[-1].get_output_shape())

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        z_data = []
        a_data = []
        self.z_data = []
        self.a_data = []
        for activation in prev_layer_activation:
            z_data_current = None
            a_data_current = None
            last_value = activation
            self.z_data.append([])
            self.a_data.append([activation])
            for i in range(len(self.layers)):
                z_data_current, a_data_current = self.layers[i].forward_pass(last_value)
                last_value = a_data_current
                self.z_data[-1].append(z_data_current)
                self.a_data[-1].append(a_data_current)

            z_data.append(z_data_current)
            a_data.append(a_data_current)

        return np.array(z_data), np.array(a_data)

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        global loop_time
        start_time = time.perf_counter()
        dc_da = dc_da.reshape(self.output_shape)
        new_dc_da_list = []
        for i in range(len(prev_layer_a)):
            new_dc_da = dc_da[i].flatten()
            for j in reversed(range(len(self.layers))):
                layer = self.layers[j]
                new_dc_da = layer.backwards_pass(self.a_data[i][j], self.z_data[i][j], new_dc_da)
            new_dc_da_list.append(new_dc_da)

        loop_time += time.perf_counter() - start_time

        return np.array(new_dc_da_list).flatten()

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        param_num = 0
        for layer in self.layers:
            param_num += layer.count_params()
        return param_num


class Stack:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.output_shape = None

    def init_weights(self, prev_layer_activation_shape):
        self.output_shape = (prev_layer_activation_shape[0], self.stack_size, *prev_layer_activation_shape[1:])

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        padded = np.pad(prev_layer_activation, ((self.stack_size - 1, 0),) + ((0, 0),) * (prev_layer_activation.ndim - 1))
        stacked = np.moveaxis(sliding_window_view(padded, window_shape=self.stack_size, axis=0), -1, 1)
        return stacked, stacked

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        dc_da = dc_da.reshape(self.output_shape)
        new_dc_da = np.array([np.sum(np.fliplr(dc_da).diagonal(offset).T, axis=0).T
                              for offset in range(-dc_da.shape[0] + 1, dc_da.shape[1] - self.stack_size + 1)][::-1])
        return new_dc_da

    def update_weights(self, learning_rate):
        pass

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return 0
