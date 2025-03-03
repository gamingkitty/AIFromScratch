import numpy as np


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

    def forward_pass(self, prev_layer_activation):
        z_data = np.dot(prev_layer_activation.flatten(), self.weights)
        a_data = self.activation_function(z_data)

        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
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
    def __init__(self, kernel_shape, activation_function, activation_function_derivative, input_shape=None):
        self.input_shape = input_shape
        self.input_num = 0
        self.kernel_shape = kernel_shape
        self.kernel_num = np.prod(kernel_shape)
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.kernel = np.zeros(kernel_shape)
        self.output_shape = ()
        self.output_num = 0
        self.gradient = np.zeros(self.kernel_num)

    def init_weights(self, previous_layer_output_shape):
        self.kernel = np.random.rand(self.kernel_shape[0], self.kernel_shape[1]) - 0.5
        if self.input_shape is None:
            self.input_shape = previous_layer_output_shape
        self.input_num = np.prod(self.input_shape)
        self.output_shape = ((self.input_shape[0] - self.kernel_shape[0]) + 1, (self.input_shape[1] - self.kernel_shape[1]) + 1)
        self.output_num = np.prod(self.output_shape)

    def forward_pass(self, prev_layer_activation):
        prev_layer_activation = prev_layer_activation.reshape(self.input_shape)

        z_data = np.zeros(self.output_shape)
        for y in range(self.output_shape[0]):
            for x in range(self.output_shape[1]):
                z_data[y, x] = np.sum(prev_layer_activation[y:y + self.kernel_shape[0], x:x + self.kernel_shape[1]] * self.kernel)

        z_data = z_data.flatten()
        a_data = self.activation_function(z_data)
        return z_data, a_data

    def backwards_pass(self, prev_layer_a, this_layer_z, dc_da):
        prev_layer_a = prev_layer_a.reshape(self.input_shape)

        # Calculate dc_dz
        dc_dz = np.dot(dc_da, self.activation_function_derivative(this_layer_z))

        # Calculate dz_da
        dz_da = np.zeros((self.output_num, self.input_shape[0], self.input_shape[1]))
        for y in range(self.output_shape[0]):
            for x in range(self.output_shape[1]):
                dz_da[y * self.output_shape[1] + x, y:y + self.kernel_shape[0], x:x + self.kernel_shape[1]] = self.kernel
        dz_da = dz_da.reshape(dz_da.shape[0], -1)

        # Calculate dz_dw
        dz_dw = np.zeros((self.kernel_num, self.output_num))
        for y in range(self.kernel_shape[0]):
            for x in range(self.kernel_shape[1]):
                dz_dw[y * self.kernel_shape[1] + x] = prev_layer_a[y:y + self.output_shape[0], x:x + self.output_shape[1]].flatten()
        dz_dw = dz_dw.reshape(dz_dw.shape[0], -1)

        # Calculate gradient
        self.gradient += np.dot(dc_dz, dz_dw.T)

        # Calculate new dc_da
        dc_da = np.dot(dc_dz, dz_da)
        return dc_da

    def update_weights(self, learning_rate):
        self.kernel -= learning_rate * self.gradient.reshape(self.kernel_shape)
        self.gradient = np.zeros(self.kernel_num)

    def get_output_shape(self):
        return self.output_shape
