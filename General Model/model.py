import numpy as np
import layers
import model_functions
import pickle


class Model:
    def __init__(self, loss_function, input_shape, *model_layers):
        self.input_shape = input_shape
        self.input_num = np.prod(input_shape)

        self.layers = model_layers

        self.loss = loss_function
        self.final_dc_da = None

        prev_output_shape = input_shape
        for layer in self.layers:
            layer.init_weights(prev_output_shape)
            prev_output_shape = layer.get_output_shape()

        self.output_shape = prev_output_shape
        self.output_num = np.prod(prev_output_shape)

    def predict(self, input_data):
        prediction = input_data

        for layer in self.layers:
            prediction = layer.predict(prediction)

        return prediction

    def forward_propagate(self, input_data):
        input_data = input_data.flatten()
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

    def backwards_propagate(self, z_data, a_data, expected_output, reward_mult):
        dc_da = self.loss.derivative(a_data[-1], expected_output) * reward_mult
        for i in reversed(range(len(self.layers))):
            dc_da = self.layers[i].backwards_pass(a_data[i], z_data[i], dc_da)
        self.final_dc_da = dc_da

    def fit(self, data, labels, epochs, learning_rate, batch_size=1, shuffle_data=True, console_updates=True, reward_mults=None):
        if console_updates:
            print("Training model...")
        data_size = len(data)

        if reward_mults is None:
            reward_mults = np.ones((len(data)))

        for i in range(epochs):
            if shuffle_data:
                indices = np.arange(data_size)
                np.random.shuffle(indices)
                data, labels, reward_mults = data[indices], labels[indices], reward_mults[indices]

            total_loss = 0
            total_correct = 0
            for j in range(data_size):
                z_data, a_data = self.forward_propagate(data[j])
                total_loss += self.loss(a_data[-1], labels[j])
                total_correct += np.argmax(a_data[-1]) == np.argmax(labels[j])
                self.backwards_propagate(z_data, a_data, labels[j], reward_mults[j])

                if j % batch_size == 0 and j != 0:
                    for layer in self.layers:
                        layer.update_weights(learning_rate)
                # if j % 10 == 0 and j != 0:
                #     print(f"So far there is loss of {(total_loss / j):.6f} and {(100 * (total_correct / j)):.4f}% accuracy.")
                    # print(f"Total convolution time: {layers.total_convolution_time:.6f}")
                    # print(f"Total dc da time: {layers.dc_da_time}")
                    # print(f"Total pooling time: {layers.total_pooling_time:.6f}")

            for layer in self.layers:
                layer.update_weights(learning_rate)

            if console_updates:
                print(f"Finished epoch {i + 1} with an average loss of {(total_loss / data_size):.6f} and {(100 * (total_correct / data_size)):.4f}% accuracy.")
        if console_updates:
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
