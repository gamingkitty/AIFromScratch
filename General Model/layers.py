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


# class Convolution:
#     def __init__(self, kernel_shape, activation_function, activation_function_derivative):
#         self.input_shape = None
#         self.kernel_shape = kernel_shape
#         self.activation_function = activation_function
#         self.activation_function_derivative = activation_function_derivative
#         self.kernel = np.zeros_like(kernel_shape)
#
#     def init_weights(self, previous_layer_output_shape):
#         self.kernel = np.random.rand(self.kernel_shape) - 0.5
#         self.input_shape = previous_layer_output_shape
