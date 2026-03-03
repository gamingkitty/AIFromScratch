import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.lib.stride_tricks import sliding_window_view
import time
from scratch_model import optimizers


attention_time = 0
dense_time = 0
layer_norm_time = 0
positional_time = 0
dropout_time = 0


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
    def __init__(self, neuron_num, activation_function=None):
        self.neuron_num = neuron_num
        self.activation_function = activation_function
        self.weights = None
        self.gradient = None
        self.weight_optimizer = None

        self.biases = None
        self.bias_gradient = None
        self.bias_optimizer = None

        self.output_shape = None

        self.has_activation = self.activation_function is not None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        self.output_shape = (self.neuron_num,)

        in_num = np.prod(previous_layer_output_shape)
        out_num = self.neuron_num
        weight_limit = math.sqrt(6 / (in_num + out_num))

        self.weights = np.random.uniform(-weight_limit, weight_limit, size=(in_num, out_num)).astype(dtype)
        self.gradient = np.zeros_like(self.weights)
        self.weight_optimizer = optimizer(*optimizer_args)
        self.weight_optimizer.initialize(self.weights, dtype=dtype)

        self.biases = np.zeros(out_num)
        self.bias_gradient = np.zeros_like(self.biases)
        self.bias_optimizer = optimizer(*optimizer_args)
        self.bias_optimizer.initialize(self.biases, dtype=dtype)

        if optimizer is optimizers.AdamW:
            self.bias_optimizer.weight_decay = 0

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        z_data = np.dot(prev_layer_activation, self.weights)
        z_data += self.biases
        if self.has_activation:
            a_data = self.activation_function(z_data)
        else:
            a_data = z_data

        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        if not self.has_activation:
            dc_dz = dc_da
        elif self.activation_function.is_elementwise:
            dc_dz = dc_da * self.activation_function.derivative(this_layer_z)
        else:
            dc_dz = np.einsum('ij,ijk->ik', dc_da, self.activation_function.derivative(this_layer_z))
        self.gradient += np.dot(prev_layer_a.T, dc_dz)
        self.bias_gradient += np.sum(dc_dz, axis=0)

        dc_da = np.dot(dc_dz, self.weights.T)
        return dc_da

    def update_weights(self, learning_rate, grad_scale=1):
        self.weight_optimizer.update_weights(self.gradient * grad_scale, learning_rate)
        self.bias_optimizer.update_weights(self.bias_gradient * grad_scale, learning_rate)

        self.gradient.fill(0.0)
        self.bias_gradient.fill(0.0)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def get_norm(self):
        return np.sum(self.gradient * self.gradient) + np.sum(self.bias_gradient * self.bias_gradient)


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
        prev_layer_activation = prev_layer_activation.reshape(self.input_shape)

        # Transpose so that its (window_num, channel_num, k_h, k_w) this way it matches the kernels and math can be done on it easily.
        windows = np.transpose(get_windows(self.kernel_shape, prev_layer_activation), (1, 0, 2, 3))

        z_data = (np.tensordot(windows, self.kernels, axes=([1, 2, 3], [1, 2, 3])).T + self.biases[:, np.newaxis]).flatten()

        a_data = self.activation_function(z_data)
        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
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
        dc_da = np.dot(dc_dz.reshape(-1), self.dz_da.reshape(-1, self.input_num))

        return dc_da.flatten()

    def update_weights(self, learning_rate):
        # Divide by output num because to get the gradient you have to sum over the affects of each weight on every output in a specific channel.
        self.kernels -= learning_rate * self.gradient.reshape((self.kernel_num, *self.channel_kernel_shape)) / self.output_num
        self.biases -= learning_rate * self.bias_gradient / self.output_num
        self.bias_gradient = np.zeros(self.kernel_num)
        self.gradient = np.zeros(self.gradient_size)

        # dz_da will be (kernel_num, output_num, input_num) because it tracks how the input affects
        # the output, and output is (kernel_num, output_num)
        # Very slow right now, should look for speed up
        self.dz_da = np.zeros((self.kernel_num, self.output_num, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        for k in range(self.kernel_num):
            for y in range(self.output_shape[0]):
                for x in range(self.output_shape[1]):
                    self.dz_da[k, y * self.output_shape[1] + x, :, y:y + self.kernel_shape[0], x:x + self.kernel_shape[1]] = self.kernels[k]
        self.dz_da = self.dz_da.reshape(self.kernel_num, self.output_num, -1)

    def get_output_shape(self):
        return self.true_output_shape

    def count_params(self):
        return np.prod(self.kernels.shape) + np.prod(self.biases.shape)

    def get_norm(self):
        return np.sum(self.gradient * self.gradient) + np.sum(self.bias_gradient * self.bias_gradient)


# Very slow currently
class MaxPooling:
    def __init__(self, kernel_shape, stride, input_shape=None):
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.input_shape = input_shape
        self.output_shape = None
        self.output_num = None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        if self.input_shape is None:
            self.input_shape = previous_layer_output_shape
        self.output_shape = (self.input_shape[0], math.ceil(self.input_shape[1] / self.stride) - math.ceil(self.kernel_shape[0] / self.stride) + 1,
                                                  math.ceil(self.input_shape[2] / self.stride) - math.ceil(self.kernel_shape[1] / self.stride) + 1)
        self.output_num = np.prod(self.output_shape)

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
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

        return windows, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
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

        return new_dc_da

    def update_weights(self, learning_rate, grad_scale=1):
        pass

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return 0

    def get_norm(self):
        return 0


# Assumes input is something one hot encoded.
class Embedding:
    def __init__(self, embedding_dimension, vocab_size):
        self.neuron_num = embedding_dimension
        self.vocab_size = vocab_size
        self.weights = None
        self.gradient = None
        self.weights_optimizer = None
        self.output_shape = None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        in_num = np.prod(previous_layer_output_shape)
        out_num = self.neuron_num
        weight_limit = math.sqrt(6 / (in_num + out_num))
        self.weights = np.random.uniform(-weight_limit, weight_limit, size=(self.vocab_size, self.neuron_num)).astype(dtype)
        self.gradient = np.zeros_like(self.weights)
        self.weights_optimizer = optimizer(*optimizer_args)
        self.weights_optimizer.initialize(self.weights, dtype=dtype)

        self.output_shape = (*previous_layer_output_shape, self.neuron_num)

        if optimizer is optimizers.AdamW:
            self.weights_optimizer.weight_decay = 0

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        prev_layer_activation = prev_layer_activation
        z_data = self.weights[prev_layer_activation]

        return z_data, z_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        # self.gradient[prev_layer_a.reshape(self.input_shape)] += dc_dz
        np.add.at(self.gradient, prev_layer_a, dc_da)
        # self.gradient[prev_layer_a] += dc_dz

        # Can turn on if want for some visualization or something
        # dc_da = np.dot(dc_da, self.weights.T)
        # return dc_da
        return None

    def update_weights(self, learning_rate, grad_scale=1):
        self.weights_optimizer.update_weights(self.gradient * grad_scale, learning_rate)
        self.gradient.fill(0.0)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return np.prod(self.weights.shape)

    def get_norm(self):
        return np.sum(self.gradient * self.gradient)


# Change so that doesnt store self.dz_da
class Dropout:
    def __init__(self, dropout_percent):
        self.dropout_percent = dropout_percent
        self.input_shape = None
        self.input_num = None
        self.scale = 1 / (1 - dropout_percent)

        self.rng = np.random.default_rng()

    # To make pickle work with rng generator
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("rng", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rng = np.random.default_rng()

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        self.input_shape = previous_layer_output_shape
        self.input_num = np.prod(self.input_shape)

    def predict(self, prev_layer_activation):
        return prev_layer_activation

    def forward_pass(self, prev_layer_activation):
        # global dropout_time
        # t0 = time.perf_counter()
        mask = self.rng.random(prev_layer_activation.shape) > self.dropout_percent
        a_data = prev_layer_activation * mask * self.scale

        # dropout_time += time.perf_counter() - t0
        return mask, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        return dc_da * this_layer_z * self.scale

    def update_weights(self, learning_rate, grad_scale=1):
        pass

    def get_output_shape(self):
        return self.input_shape

    def count_params(self):
        return 0

    def get_norm(self):
        return 0


# Currently returns entire sequence. Might add option in future to only return final output of layer.
class Recurrent:
    def __init__(self, neuron_num, activation_function, input_shape=None):
        self.neuron_num = neuron_num
        self.activation_function = activation_function

        self.hidden_weights = None
        self.hidden_gradient = None
        self.hidden_optimizer = None
        self.input_weights = None
        self.input_gradient = None
        self.input_optimizer = None

        # Input shape can be something like (-1, n) because size can vary based on number of tokens passed in
        self.input_shape = input_shape
        self.output_shape = None

        self.this_layer_a = None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        if self.input_shape is None:
            self.input_shape = previous_layer_output_shape

        self.output_shape = (self.input_shape[0], self.neuron_num)

        in_num = np.prod(previous_layer_output_shape[1:])
        out_num = self.neuron_num
        weight_limit = math.sqrt(6 / (in_num + out_num))

        self.hidden_weights = np.random.uniform(-weight_limit, weight_limit, size=(out_num, out_num))
        self.hidden_gradient = np.zeros_like(self.hidden_weights)
        self.hidden_optimizer = optimizer(*optimizer_args)
        self.hidden_optimizer.initialize(self.hidden_weights, dtype=dtype)

        self.input_weights = np.random.uniform(-weight_limit, weight_limit, size=(in_num, out_num))
        self.input_gradient = np.zeros_like(self.input_weights)
        self.input_optimizer = optimizer(*optimizer_args)
        self.input_optimizer.initialize(self.input_weights, dtype=dtype)

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        before_activation = np.dot(prev_layer_activation[0], self.input_weights)
        last_output = self.activation_function(before_activation)

        z_states = [before_activation]
        a_states = [last_output]

        for token in prev_layer_activation[1:]:
            before_activation = np.dot(last_output, self.hidden_weights) + np.dot(token, self.input_weights)
            z_states.append(before_activation)
            last_output = self.activation_function(before_activation)
            a_states.append(last_output)

        self.this_layer_a = np.array(a_states)
        return np.array(z_states), self.this_layer_a

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        # dc_da = dc_da.reshape(self.output_shape)
        # this_layer_z = this_layer_z.reshape(self.output_shape)
        # prev_layer_a = prev_layer_a.reshape(self.input_shape)

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

        new_dc_da = np.array(new_dc_da[::-1])
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

        return new_dc_da

    def update_weights(self, learning_rate, grad_scale):
        self.input_optimizer.update_weights(self.input_gradient * grad_scale, learning_rate)
        self.hidden_optimizer.update_weights(self.hidden_gradient * grad_scale, learning_rate)

        self.input_gradient.fill(0.0)
        self.hidden_gradient.fill(0.0)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return np.prod(self.input_weights.shape) + np.prod(self.hidden_weights.shape)

    def get_norm(self):
        return np.sum(self.input_gradient * self.input_gradient) + np.sum(self.hidden_gradient * self.hidden_gradient)


class Loop:
    def __init__(self, *layers):
        self.layers = layers
        self.output_shape = None
        self.z_data = None
        self.a_data = None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        loop_num = previous_layer_output_shape[0]
        previous_layer_output_shape = previous_layer_output_shape[1:]
        for layer in self.layers:
            layer.init_weights(previous_layer_output_shape, optimizer, dtype=dtype, optimizer_args=optimizer_args)
            previous_layer_output_shape = layer.get_output_shape()
        self.output_shape = (loop_num, *self.layers[-1].get_output_shape())

    # Fix in future for dropout
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
        return z_data, np.array(a_data)

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        dc_da = dc_da.reshape(self.output_shape)
        new_dc_da_list = []
        for i in range(len(prev_layer_a)):
            new_dc_da = dc_da[i] # .flatten()
            for j in reversed(range(len(self.layers))):
                layer = self.layers[j]
                new_dc_da = layer.backwards_pass(self.a_data[i][j], self.z_data[i][j], new_dc_da)
            new_dc_da_list.append(new_dc_da)

        return np.array(new_dc_da_list)

    def update_weights(self, learning_rate, grad_scale=1):
        for layer in self.layers:
            layer.update_weights(learning_rate, grad_scale=grad_scale)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        param_num = 0
        for layer in self.layers:
            param_num += layer.count_params()
        return param_num

    def get_norm(self):
        return sum(layer.get_norm() for layer in self.layers)


class Stack:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.output_shape = None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        self.output_shape = (previous_layer_output_shape[0], self.stack_size, *previous_layer_output_shape[1:])

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        padded = np.pad(prev_layer_activation, ((self.stack_size - 1, 0),) + ((0, 0),) * (prev_layer_activation.ndim - 1))
        stacked = np.moveaxis(sliding_window_view(padded, window_shape=self.stack_size, axis=0), -1, 1)
        return None, stacked

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        dc_da = dc_da.reshape(self.output_shape)
        new_dc_da = np.array([np.sum(np.fliplr(dc_da).diagonal(offset).T, axis=0).T for offset in range(-dc_da.shape[0] + 1, dc_da.shape[1] - self.stack_size + 1)][::-1])
        return new_dc_da

    def update_weights(self, learning_rate, grad_scale=1):
        pass

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return 0

    def get_norm(self):
        return 0


class Attention:
    def __init__(self, value_size, query_key_size, heads, mask=None):
        self.value_size = value_size
        self.query_key_size = query_key_size

        self.key_weights = None
        self.query_weights = None
        self.value_weights = None

        self.key_gradient = None
        self.query_gradient = None
        self.value_gradient = None

        self.key_optimizer = None
        self.query_optimizer = None
        self.value_optimizer = None

        self.output_shape = None

        self.heads = heads

        self.mask = mask

        self.scale = 1 / np.sqrt(self.query_key_size)

        self.key_cache = []
        self.value_cache = []

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        in_num = previous_layer_output_shape[1]

        self.output_shape = (previous_layer_output_shape[0], self.heads * self.value_size)

        weight_limit_qk = math.sqrt(6 / (in_num + self.query_key_size))
        weight_limit_v = math.sqrt(6 / (in_num + self.value_size))

        self.key_weights = np.random.uniform(-weight_limit_qk, weight_limit_qk, (self.heads, previous_layer_output_shape[1], self.query_key_size)).astype(dtype)
        self.query_weights = np.random.uniform(-weight_limit_qk, weight_limit_qk, (self.heads, previous_layer_output_shape[1], self.query_key_size)).astype(dtype)
        self.value_weights = np.random.uniform(-weight_limit_v, weight_limit_v, (self.heads, previous_layer_output_shape[1], self.value_size)).astype(dtype)

        self.key_gradient = np.zeros_like(self.key_weights)
        self.query_gradient = np.zeros_like(self.query_weights)
        self.value_gradient = np.zeros_like(self.value_weights)

        self.key_optimizer = optimizer(*optimizer_args)
        self.query_optimizer = optimizer(*optimizer_args)
        self.value_optimizer = optimizer(*optimizer_args)
        self.key_optimizer.initialize(self.key_weights, dtype=dtype)
        self.query_optimizer.initialize(self.query_weights, dtype=dtype)
        self.value_optimizer.initialize(self.value_weights, dtype=dtype)

    def predict_cache(self, new_token):
        query = np.einsum('bti,hiv->bhtv', new_token, self.query_weights)
        key = np.einsum('bti,hiv->bhtv', new_token, self.key_weights)
        value = np.einsum('bti,hiv->bhtv', new_token, self.value_weights)

    def predict(self, prev_layer_activation):
        z, a = self.forward_pass(prev_layer_activation)
        return a

    def forward_pass(self, prev_layer_activation):
        # global attention_time
        # t0 = time.perf_counter()
        t = prev_layer_activation.shape[1]

        # Shape (head, time, query/key/value size)
        queries = np.einsum('bti,hiv->bhtv', prev_layer_activation, self.query_weights)
        keys = np.einsum('bti,hiv->bhtv', prev_layer_activation, self.key_weights)
        values = np.einsum('bti,hiv->bhtv', prev_layer_activation, self.value_weights)

        # Dot key and query values to get the attention scores
        # Shape (head, query, key), so each row contains the values of query_i dot all keys,
        # or how much token_i attends to all other tokens
        raw_attention_scores = np.einsum('bhtv,bhsv->bhts', queries, keys) * self.scale

        if self.mask is not None:
            mask = self.mask(t)
            raw_attention_scores = np.where(mask, raw_attention_scores, -1e9)

        # Softmax attention scores along key axis
        e_xs = np.exp(raw_attention_scores - np.max(raw_attention_scores, axis=-1, keepdims=True))

        if self.mask is not None:
            e_xs *= mask

        attention_scores = e_xs / np.sum(e_xs, axis=-1, keepdims=True)

        # Multiply attention scores by corresponding values and sum
        output = np.einsum('bhts,bhsv->bhtv', attention_scores, values)

        # Concatenate the heads
        output = output.transpose(0, 2, 1, 3).reshape(output.shape[0], output.shape[2], -1)

        # attention_time += time.perf_counter() - t0

        return (queries, keys, values, attention_scores), output

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        # global attention_time
        # t0 = time.perf_counter()
        queries, keys, values, attention_scores = this_layer_z
        b = prev_layer_a.shape[0]
        t = prev_layer_a.shape[1]

        dc_da = dc_da.reshape(b, t, self.heads, self.value_size).transpose(0, 2, 1, 3)

        dc_dv = np.einsum('bhts,bhtv->bhsv', attention_scores, dc_da)
        self.value_gradient += np.einsum('bti,bhtv->hiv', prev_layer_a, dc_dv)

        dc_dattention = np.einsum('bhtv,bhsv->bhts', dc_da, values)

        # # Softmax derivative
        # h, n, m = attention_scores.shape
        # idx = np.arange(m)
        # diagonals = np.zeros((h, n, m, m), dtype=attention_scores.dtype)
        # diagonals[:, :, idx, idx] = attention_scores
        #
        # dattention_draw = diagonals - (attention_scores[:, :, :, np.newaxis] * attention_scores[:, :, np.newaxis, :])
        #
        # # Compute dc_draw
        # dc_draw = np.sum((dc_dattention[:, :, :, np.newaxis] * dattention_draw), axis=2)

        # Faster direct dc_draw computation
        dc_draw = attention_scores * (dc_dattention - np.sum(dc_dattention * attention_scores, axis=-1, keepdims=True))

        if self.mask is not None:
            dc_draw *= self.mask(t)

        # Works because dr_dq/dc_dk for all queries/keys is the same, just (keys/queries) / sqrt(n)
        dc_dquery = np.einsum('bhst,bhtk->bhsk', dc_draw, keys * self.scale)
        dc_dkey = np.einsum('bhts,bhtk->bhsk', dc_draw, queries * self.scale)

        self.query_gradient += np.einsum('bti,bhtq->hiq', prev_layer_a, dc_dquery)
        self.key_gradient += np.einsum('bti,bhtk->hik', prev_layer_a, dc_dkey)

        new_dc_da = np.sum(np.einsum('bhtv,hiv->bhti', dc_dv, self.value_weights) + np.einsum('bhtq,hiq->bhti', dc_dquery, self.query_weights) + np.einsum('bhtk,hik->bhti', dc_dkey, self.key_weights), axis=1)
        # attention_time += time.perf_counter() - t0
        return new_dc_da

    def update_weights(self, learning_rate, grad_scale):
        self.value_optimizer.update_weights(self.value_gradient * grad_scale, learning_rate)
        self.query_optimizer.update_weights(self.query_gradient * grad_scale, learning_rate)
        self.key_optimizer.update_weights(self.key_gradient * grad_scale, learning_rate)

        self.value_gradient.fill(0.0)
        self.query_gradient.fill(0.0)
        self.key_gradient.fill(0.0)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return np.prod(self.value_weights.shape) + np.prod(self.query_weights.shape) + np.prod(self.key_weights.shape)

    def get_norm(self):
        return np.sum(self.value_gradient * self.value_gradient) + np.sum(self.query_gradient * self.query_gradient) + np.sum(self.value_gradient * self.value_gradient)


# Must take in vectorized activation function
class TimeDistributedDense:
    def __init__(self, neuron_num, activation_function=None):
        self.neuron_num = neuron_num
        self.activation_function = activation_function
        self.weights = None
        self.gradient = None
        self.weight_optimizer = None

        self.biases = None
        self.bias_gradient = None
        self.bias_optimizer = None

        self.output_shape = None

        self.has_activation = self.activation_function is not None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        self.output_shape = (previous_layer_output_shape[0], self.neuron_num)

        previous_layer_output_shape = previous_layer_output_shape[1:]
        in_num = np.prod(previous_layer_output_shape)
        out_num = self.neuron_num
        weight_limit = math.sqrt(6 / (in_num + out_num))

        self.weights = np.random.uniform(-weight_limit, weight_limit, size=(in_num, out_num)).astype(dtype)
        self.gradient = np.zeros_like(self.weights)
        self.weight_optimizer = optimizer(*optimizer_args)
        self.weight_optimizer.initialize(self.weights, dtype=dtype)

        self.biases = np.zeros(out_num, dtype=dtype)
        self.bias_gradient = np.zeros_like(self.biases)
        self.bias_optimizer = optimizer(*optimizer_args)
        self.bias_optimizer.initialize(self.biases, dtype=dtype)

        if optimizer is optimizers.AdamW:
            self.bias_optimizer.weight_decay = 0

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        # global dense_time
        # t0 = time.perf_counter()
        z_data = np.dot(prev_layer_activation, self.weights)
        z_data += self.biases
        if self.has_activation:
            a_data = self.activation_function(z_data)
        else:
            a_data = z_data

        # dense_time += time.perf_counter() - t0
        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        # global dense_time
        # t0 = time.perf_counter()
        if not self.has_activation:
            dc_dz = dc_da
        elif self.activation_function.is_elementwise:
            dc_dz = dc_da * self.activation_function.derivative(this_layer_z)
        else:
            dc_dz = np.einsum('bti,btio->bto', dc_da, self.activation_function.derivative(this_layer_z))
        self.gradient += np.einsum('bti,bto->io', prev_layer_a, dc_dz) # np.dot(prev_layer_a.T, dc_dz)
        self.bias_gradient += np.sum(dc_dz, axis=(0, 1))

        dc_da = np.dot(dc_dz, self.weights.T)
        # dense_time += time.perf_counter() - t0
        return dc_da

    def update_weights(self, learning_rate, grad_scale=1):
        self.weight_optimizer.update_weights(self.gradient * grad_scale, learning_rate)
        self.bias_optimizer.update_weights(self.bias_gradient * grad_scale, learning_rate)

        self.gradient.fill(0.0)
        self.bias_gradient.fill(0.0)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def get_norm(self):
        return np.sum(self.gradient * self.gradient) + np.sum(self.bias_gradient * self.bias_gradient)


class Sum:
    def __init__(self):
        self.output_shape = None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        self.output_shape = previous_layer_output_shape[1:]

    def predict(self, prev_layer_activation):
        return np.sum(prev_layer_activation, axis=0)

    def forward_pass(self, prev_layer_activation):
        return None, np.sum(prev_layer_activation, axis=0)

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        return np.tile(dc_da, (len(prev_layer_a), 1))

    def update_weights(self, learning_rate, grad_scale=1):
        pass

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return 0

    def get_norm(self):
        return 0


class Flatten:
    def __init__(self):
        self.input_shape = None
        self.output_size = None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        self.input_shape = (-1, *previous_layer_output_shape)
        self.output_size = np.prod(previous_layer_output_shape)

    def predict(self, prev_layer_activation):
        return prev_layer_activation.reshape(prev_layer_activation.shape[0], -1)

    def forward_pass(self, prev_layer_activation):
        return None, prev_layer_activation.reshape(prev_layer_activation.shape[0], -1)

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        return np.reshape(dc_da, self.input_shape)

    def update_weights(self, learning_rate, grad_scale=1):
        pass

    def get_output_shape(self):
        return self.output_size,

    def count_params(self):
        return 0

    def get_norm(self):
        return 0


class Reshape:
    def __init__(self, shape):
        self.input_shape = None
        self.output_shape = shape

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        self.input_shape = previous_layer_output_shape

    def predict(self, prev_layer_activation):
        return np.reshape(prev_layer_activation, self.output_shape)

    def forward_pass(self, prev_layer_activation):
        return None, np.reshape(prev_layer_activation, (-1, *self.output_shape))

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        return np.reshape(dc_da, (-1, *self.input_shape))

    def update_weights(self, learning_rate, grad_scale=1):
        pass

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return 0

    def get_norm(self):
        return 0


class ResidualBlock:
    def __init__(self, *layers):
        self.layers = layers
        self.output_shape = None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        for layer in self.layers:
            layer.init_weights(previous_layer_output_shape, optimizer, dtype=dtype, optimizer_args=optimizer_args)
            previous_layer_output_shape = layer.get_output_shape()
        self.output_shape = self.layers[-1].get_output_shape()

        if self.output_shape != previous_layer_output_shape:
            raise ValueError("Input and output shape mismatch! Projection not yet implemented.")

    def predict(self, prev_layer_activation):
        last_layer_out = prev_layer_activation
        for layer in self.layers:
            last_layer_out = layer.predict(last_layer_out)
        return last_layer_out + prev_layer_activation

    def forward_pass(self, prev_layer_activation):
        z_data = []
        a_data = [prev_layer_activation]

        last_value = prev_layer_activation
        for i in range(len(self.layers)):
            z_data_current, a_data_current = self.layers[i].forward_pass(last_value)
            last_value = a_data_current
            a_data.append(a_data_current)
            z_data.append(z_data_current)

        a_data[-1] = a_data[-1] + prev_layer_activation

        return (a_data, z_data), a_data[-1]

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        a_data, z_data = this_layer_z
        current_dc_da = dc_da
        for i in reversed(range(len(self.layers))):
            current_dc_da = self.layers[i].backwards_pass(a_data[i], z_data[i], current_dc_da)
        return current_dc_da + dc_da

    def update_weights(self, learning_rate, grad_scale=1):
        for layer in self.layers:
            layer.update_weights(learning_rate, grad_scale=grad_scale)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        param_num = 0
        for layer in self.layers:
            param_num += layer.count_params()
        return param_num

    def get_norm(self):
        return sum(layer.get_norm() for layer in self.layers)


class LayerNorm:
    def __init__(self):
        self.epsilon = 1e-5
        self.weights = None
        self.weights_gradient = None
        self.weights_optimizer = None
        self.biases = None
        self.biases_gradient = None
        self.bias_optimizer = None
        self.output_shape = None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        in_num = previous_layer_output_shape[-1]
        self.output_shape = previous_layer_output_shape

        self.weights = np.ones(in_num, dtype=dtype)
        self.weights_gradient = np.zeros_like(self.weights)
        self.weights_optimizer = optimizer(*optimizer_args)
        self.weights_optimizer.initialize(self.weights, dtype=dtype)

        self.biases = np.zeros(in_num, dtype=dtype)
        self.biases_gradient = np.zeros_like(self.biases)
        self.bias_optimizer = optimizer(*optimizer_args)
        self.bias_optimizer.initialize(self.biases, dtype=dtype)

        if optimizer is optimizers.AdamW:
            self.weights_optimizer.weight_decay = 0
            self.bias_optimizer.weight_decay = 0

    def predict(self, prev_layer_activation):
        z, a = self.forward_pass(prev_layer_activation)
        return a

    def forward_pass(self, prev_layer_activation):
        # global layer_norm_time
        # t0 = time.perf_counter()
        dif_mean = (prev_layer_activation - prev_layer_activation.mean(axis=-1, keepdims=True))
        var = (dif_mean * dif_mean).mean(axis=-1, keepdims=True)
        inv_std = 1 / np.sqrt(var + self.epsilon)

        quotient = dif_mean * inv_std

        out = (self.weights * quotient) + self.biases

        # layer_norm_time += time.perf_counter() - t0
        return (inv_std, quotient), out

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        # global layer_norm_time
        # t0 = time.perf_counter()
        inv_std, quotient = this_layer_z

        self.weights_gradient += np.sum(quotient * dc_da, axis=tuple(range(dc_da.ndim - 1)))
        self.biases_gradient += np.sum(dc_da, axis=tuple(range(dc_da.ndim - 1)))

        g = dc_da * self.weights

        # new_dc_da = ((g - (g.sum() / n)) / epsilon_std) - ((dif_mean * np.dot(dif_mean, g)) / (n * np.pow(epsilon_std, 3)))
        # Equivalent but slightly faster version
        new_dc_da = (g - g.mean(axis=-1, keepdims=True) - quotient * (g * quotient).mean(axis=-1, keepdims=True)) * inv_std

        # layer_norm_time += time.perf_counter() - t0
        return new_dc_da

    def update_weights(self, learning_rate, grad_scale=1):
        self.weights_optimizer.update_weights(self.weights_gradient * grad_scale, learning_rate)
        self.bias_optimizer.update_weights(self.biases_gradient * grad_scale, learning_rate)

        self.weights_gradient.fill(0.0)
        self.biases_gradient.fill(0.0)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def get_norm(self):
        return np.sum(self.weights_gradient * self.weights_gradient) + np.sum(self.biases_gradient * self.biases_gradient)


class PositionalEncoder:
    def __init__(self):
        self.output_shape = None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        self.output_shape = previous_layer_output_shape

    def predict(self, prev_layer_activation):
        z, a = self.forward_pass(prev_layer_activation)
        return a

    def forward_pass(self, prev_layer_activation):
        # global positional_time
        # t0 = time.perf_counter()
        t = prev_layer_activation.shape[1]
        e = prev_layer_activation.shape[2]

        positional_encodings = np.zeros(prev_layer_activation.shape)
        positional_encodings[:, :, 0::2] = np.sin(np.outer(np.arange(t), np.power(10000, -(2 / e) * np.arange((e + 1) // 2))))
        positional_encodings[:, :, 1::2] = np.cos(np.outer(np.arange(t), np.power(10000, -(2 / e) * np.arange(e // 2))))

        # positional_time += time.perf_counter() - t0
        return None, prev_layer_activation + positional_encodings

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        return dc_da

    def update_weights(self, learning_rate, grad_scale=1):
        pass

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return 0

    def get_norm(self):
        return 0


class EmbeddingTiedOutput:
    def __init__(self, vocab_size, activation_function=None):
        self.neuron_num = vocab_size
        self.activation_function = activation_function
        self.weights = None
        self.embedding_gradient = None

        self.biases = None
        self.bias_gradient = None
        self.bias_optimizer = None

        self.output_shape = None

        self.has_activation = self.activation_function is not None

    def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
        self.output_shape = (previous_layer_output_shape[0], self.neuron_num)

        self.biases = np.zeros(self.neuron_num, dtype=dtype)
        self.bias_gradient = np.zeros_like(self.biases)
        self.bias_optimizer = optimizer(*optimizer_args)
        self.bias_optimizer.initialize(self.biases, dtype=dtype)

        if optimizer is optimizers.AdamW:
            self.bias_optimizer.weight_decay = 0

    def set_from_embedding(self, embedding_layer):
        self.weights = embedding_layer.weights.T
        self.embedding_gradient = embedding_layer.gradient

    def predict(self, prev_layer_activation):
        z_data, a_data = self.forward_pass(prev_layer_activation)
        return a_data

    def forward_pass(self, prev_layer_activation):
        # global dense_time
        # t0 = time.perf_counter()
        z_data = np.dot(prev_layer_activation, self.weights)
        z_data += self.biases
        if self.has_activation:
            a_data = self.activation_function(z_data)
        else:
            a_data = z_data

        # dense_time += time.perf_counter() - t0
        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        # global dense_time
        # t0 = time.perf_counter()
        if not self.has_activation:
            dc_dz = dc_da
        elif self.activation_function.is_elementwise:
            dc_dz = dc_da * self.activation_function.derivative(this_layer_z)
        else:
            dc_dz = np.einsum('bti,btio->bto', dc_da, self.activation_function.derivative(this_layer_z))
        self.embedding_gradient += np.einsum('bti,bto->oi', prev_layer_a, dc_dz)
        self.bias_gradient += np.sum(dc_dz, axis=(0, 1))

        dc_da = np.dot(dc_dz, self.weights.T)
        # dense_time += time.perf_counter() - t0
        return dc_da

    def update_weights(self, learning_rate, grad_scale=1):
        self.bias_optimizer.update_weights(self.bias_gradient * grad_scale, learning_rate)
        self.bias_gradient.fill(0.0)

    def get_output_shape(self):
        return self.output_shape

    def count_params(self):
        return np.prod(self.biases.shape)

    def get_norm(self):
        return np.sum(self.bias_gradient * self.bias_gradient)


# class Convolution2:
#     def __init__(self, kernel_num, kernel_shape, activation_function):
#         self.kernel_num = kernel_num
#         self.kernel_shape = kernel_shape
#         self.activation_function = activation_function
#
#         self.kernels = None
#         self.biases = None
#
#     def init_weights(self, previous_layer_output_shape, optimizer, dtype=np.float32, optimizer_args=()):
#         self.channel_kernel_shape = (previous_layer_output_shape[0], *self.kernel_shape)
#         in_num = previous
#         self.kernels =

