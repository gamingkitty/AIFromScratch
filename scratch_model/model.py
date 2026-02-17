import numpy as np
import pickle
import random
from scratch_model import layers


def default_accuracy(prediction, label):
    return np.argmax(prediction) == np.argmax(label)


class Model:
    def __init__(self, loss_function, input_shape, model_layers, accuracy_function=default_accuracy):
        self.input_shape = input_shape
        self.input_num = np.prod(input_shape)

        self.layers = model_layers

        self.loss = loss_function
        # Currently in here for visualization purposes
        self.final_dc_da = None

        prev_output_shape = input_shape
        for layer in self.layers:
            layer.init_weights(prev_output_shape)
            prev_output_shape = layer.get_output_shape()

        self.output_shape = prev_output_shape
        self.output_num = np.prod(prev_output_shape)

        self.accuracy_function = accuracy_function

    def predict(self, input_data):
        prediction = input_data

        for layer in self.layers:
            prediction = layer.predict(prediction)

        return prediction

    def forward_propagate(self, input_data):
        input_data = input_data
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
                combined = list(zip(data, labels, reward_mults))
                random.shuffle(combined)
                data, labels, reward_mults = map(list, zip(*combined))

            total_loss = 0
            total_correct = 0
            for j in range(data_size):
                z_data, a_data = self.forward_propagate(data[j])
                total_loss += self.loss(a_data[-1], labels[j])
                total_correct += self.accuracy_function(a_data[-1], labels[j])
                self.backwards_propagate(z_data, a_data, labels[j], reward_mults[j])

                if (j + 1) % batch_size == 0:
                    for layer in self.layers:
                        layer.update_weights(learning_rate / batch_size)
                if (j + 1) % 100 == 0 and console_updates:
                    print(f"So far there is loss of {(total_loss / (j + 1)):.6f} and {(100 * (total_correct / (j + 1))):.4f}% accuracy.")
                    # print(f"Attention Time: {layers.attention_time}")
                    # print(f"Dense Time: {layers.dense_time}")
                    # print(f"Norm Time: {layers.layer_norm_time}")
                    # print(f"Positional Time: {layers.positional_time}")
                    # print(f"Dropout Time: {layers.dropout_time}")

            if data_size % batch_size > 0:
                for layer in self.layers:
                    layer.update_weights(learning_rate / (data_size % batch_size))

            if console_updates:
                print(f"Finished epoch {i + 1} with an average loss of {(total_loss / data_size):.6f} and {(100 * (total_correct / data_size)):.4f}% accuracy.")
        if console_updates:
            print("Finished training model.")

    def test(self, data, labels):
        total_correct = 0
        total_loss = 0
        for i in range(len(data)):
            prediction = self.predict(data[i])
            total_correct += self.accuracy_function(prediction, labels[i])
            total_loss += self.loss(prediction, labels[i])
        return total_loss / len(data), total_correct / len(data)

    def get_param_num(self):
        param_num = 0
        for layer in self.layers:
            param_num += layer.count_params()
        return param_num

    def save(self, filename):
        path = filename if filename.endswith(".pkl") else filename + ".pkl"
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        # Test for loading models made on gpu while using cpu
        if not hasattr(np, "_from_pickle"):
            def _from_pickle(*args):
                if len(args) == 3 and isinstance(args[0], (bytes, bytearray, memoryview)):
                    rawdata, shape, dtype_obj = args
                    dtype = np.dtype(dtype_obj)
                    arr = np.frombuffer(rawdata, dtype=dtype)
                    return arr.reshape(tuple(shape), order="C")

                if len(args) == 4:
                    shape, dtype_obj, is_fortran, rawdata = args
                    dtype = np.dtype(dtype_obj)
                    arr = np.frombuffer(rawdata, dtype=dtype)
                    return arr.reshape(tuple(shape), order="F" if bool(is_fortran) else "C")

                raise TypeError(
                    f"Unsupported numpy._from_pickle signature: "
                    f"len={len(args)}, types={[type(a) for a in args]}"
                )

            np._from_pickle = _from_pickle

        path = filename if filename.endswith(".pkl") else filename + ".pkl"
        with open(path, "rb") as file:
            return pickle.load(file)
