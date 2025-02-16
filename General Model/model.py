import numpy as np
import pickle


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
                raise ValueError(f"Invalid layer type: {layer_type}. Valid layer types are: 'input', 'hidden', and 'output'.")

    def add_hidden_layer(self, neuron_num, activation_function, activation_function_derivative):
        if activation_function is None:
            raise ValueError("No activation function was given for hidden layer")
        if neuron_num <= 0:
            raise ValueError("Number of neurons in hidden layer must be 1 or greater")

        if len(self.weights) == 0:
            self.weights.append(np.random.rand(self.input_num, neuron_num) - 0.5)
        else:
            self.weights.append(np.random.rand(len(self.weights[-1][0]), neuron_num) - 0.5)

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
            prediction = self.layer_activation_functions[i](np.dot(prediction, self.weights[i]))

        return prediction

    def forward_propagate(self, input_data):
        z_data = []
        # Starts with input_data in it, will be 1 longer than z.
        a_data = [input_data]

        last_value = input_data
        for i in range(len(self.weights)):
            z_data.append(np.dot(last_value, self.weights[i]))
            last_value = self.layer_activation_functions[i](z_data[-1])
            a_data.append(last_value)

        return z_data, a_data

    def backwards_propagate(self, z_data, a_data, expected_output):
        gradient = []

        # Assuming MSE. Might update so can be any loss function in the future.
        dc_da = -(2 / self.output_num) * (expected_output - a_data[-1])

        # Will be the derivative of the cost with respect to the current z in the for loop.
        dc_dz = np.dot(dc_da, self.layer_activation_function_derivatives[-1](z_data[-1]))

        gradient.append(dc_dz * a_data[-2][:, np.newaxis])

        # Continue to chain back the derivative for every hidden layer.
        for i in reversed(range(len(z_data) - 1)):
            dc_da = np.dot(dc_dz, self.weights[i + 1].T)
            dc_dz = np.dot(dc_da, self.layer_activation_function_derivatives[i](z_data[i]))
            gradient.insert(0, dc_dz * a_data[i][:, np.newaxis])

        return gradient

    def loss(self, expected_output, prediction):
        return np.sum(np.power(expected_output - prediction, 2)) / self.output_num

    def fit(self, data, labels, epochs, learning_rate, shuffle_data=True):
        print("Training model...")
        data_size = len(data)

        for i in range(epochs):
            if shuffle_data:
                indices = np.arange(data_size)
                np.random.shuffle(indices)
                data, labels = data[indices], labels[indices]

            total_loss = 0
            total_correct = 0
            for j in range(data_size):
                z_data, a_data = self.forward_propagate(data[j])
                total_loss += self.loss(labels[j], a_data[-1])
                total_correct += np.argmax(a_data[-1]) == np.argmax(labels[j])
                gradient = self.backwards_propagate(z_data, a_data, labels[j])

                for k in range(len(self.weights)):
                    self.weights[k] -= learning_rate * gradient[k]

            print(f"Finished epoch {i + 1} with an average loss of {(total_loss / data_size):.6f} and {(100 * (total_correct / data_size)):.4f}% accuracy.")
        print("Finished training model.")

    def test(self, data, labels):
        total_correct = 0
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if np.argmax(prediction) == np.argmax(labels[i]):
                total_correct += 1
        return total_correct / len(data)

    def save(self, filename):
        with open(filename + ".pkl", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename + ".pkl", "rb") as file:
            return pickle.load(file)
