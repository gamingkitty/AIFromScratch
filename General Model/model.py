import numpy as np
import pickle


class Model:
    def __init__(self, input_shape, *layers):
        self.input_shape = input_shape
        self.input_num = np.prod(input_shape)

        self.layers = layers

        prev_output_shape = input_shape
        for layer in self.layers:
            layer.init_weights(prev_output_shape)
            prev_output_shape = layer.get_output_shape()

        self.output_shape = prev_output_shape
        self.output_num = np.prod(prev_output_shape)

    def loss(self, expected_output, prediction):
        return np.sum(np.power(expected_output - prediction, 2)) / self.output_num

    def predict(self, input_data):
        prediction = input_data

        for layer in self.layers:
            z_data, prediction = layer.forward_pass(prediction)

        return prediction

    def forward_propagate(self, input_data):
        z_data = []
        # Starts with input_data in it, will be 1 longer than z.
        a_data = [input_data]

        last_value = input_data
        for i in range(len(self.layers)):
            z_data_current, a_data_current = self.layers[i].forward_pass(last_value)
            last_value = a_data_current
            a_data.append(a_data_current)
            z_data.append(z_data_current)

        return z_data, a_data

    def backwards_propagate(self, z_data, a_data, expected_output):
        dc_da = -(2 / self.output_num) * (expected_output - a_data[-1])
        for i in reversed(range(len(self.layers))):
            dc_da = self.layers[i].backwards_pass(a_data[i], z_data[i], dc_da)

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
                self.backwards_propagate(z_data, a_data, labels[j])
                for layer in self.layers:
                    layer.update_weights(learning_rate)

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
