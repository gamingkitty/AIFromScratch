import numpy as np


class Model:
    def __init__(self):
        self.input_num = 0
        # Weights are addressed by self.weights[layer][neuron][weight], so if you wanted to get the third weight of
        # the second neuron in the first layer, you would do self.weights[1][2][3]
        self.weights = []

        # Activation functions will take in a list of inputs and output a list of outputs, so with something like ReLU,
        # you would have to make it work when it takes in a list.
        self.layer_activation_functions = []
        # The derivatives will output a jacobian matrix, so for things like ReLU, the derivative will need to be a
        # jacobian matrix instead of just a list.
        self.layer_activation_function_derivatives = []
        self.output_num = 0

    def add_layer(self, neuron_num, layer_type, activation_function=None, activation_function_derivative=None):
        if layer_type == "input":
            self.input_num = neuron_num
        else:
            if self.input_num == 0:
                raise ValueError("Input layer needs to be added before other layers")

            if layer_type == "hidden":
                if self.output_num != 0:
                    raise ValueError("Cannot add hidden layers after output layer has been added")

                self.add_hidden_layer(neuron_num, activation_function, activation_function_derivative)
            elif layer_type == "output":
                self.output_num = neuron_num
                self.add_hidden_layer(neuron_num, activation_function, activation_function_derivative)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}")

    def add_hidden_layer(self, neuron_num, activation_function, activation_function_derivative):
        if activation_function is None:
            raise ValueError("No activation function was given for hidden layer")
        if neuron_num <= 0:
            raise ValueError("Number of neurons in hidden layer must be 1 or greater")

        if len(self.weights) == 0:
            self.weights.append(np.random.rand(neuron_num, self.input_num) - 0.5)
        else:
            self.weights.append(np.random.rand(neuron_num, len(self.weights[-1])) - 0.5)

        self.layer_activation_functions.append(activation_function)
        self.layer_activation_function_derivatives.append(activation_function_derivative)

    def predict(self, input_data):
        if len(input_data) != self.input_num:
            raise ValueError(
                f"Tried inputting data of size {len(input_data)} to a model that only accepts input of size {self.input_num}")
        if self.output_num == 0:
            raise ValueError("Model does not contain a proper output layer.")

        prediction = input_data

        for i in range(len(self.weights)):
            prediction = self.layer_activation_functions[i](np.dot(self.weights[i], prediction))

        return prediction

    def forward_propagate(self, input_data):
        z_data = []
        a_data = []

        last_value = input_data
        for i in range(len(self.weights)):
            z_data.append(np.dot(self.weights[i], last_value))
            last_value = self.layer_activation_functions[i](z_data[-1])
            a_data.append(last_value)

        return z_data, a_data

    def backwards_propagate(self, input_data, expected_output):
        gradient = []

        return gradient

