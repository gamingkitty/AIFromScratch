import time

import numpy as np
import pickle
import random
from scratch_model import layers
from scratch_model import optimizers
import pandas as pd
import os


def default_accuracy(prediction, label):
    return np.sum((np.argmax(prediction, axis=-1) == np.argmax(label, axis=-1)))


def default_learning_rate(step):
    return 1


class Model:
    def __init__(self, loss_function, input_shape, model_layers, optimizer=optimizers.AdamW, optimizer_args=()):
        self.input_shape = input_shape
        self.input_num = np.prod(input_shape)

        self.layers = model_layers

        self.loss = loss_function
        # Currently in here for visualization purposes
        self.final_dc_da = None

        prev_output_shape = input_shape
        for layer in self.layers:
            layer.init_weights(prev_output_shape, optimizer, optimizer_args=optimizer_args)
            prev_output_shape = layer.get_output_shape()

        self.output_shape = prev_output_shape
        self.output_num = np.prod(prev_output_shape)

        self.steps = []
        self.loss_data = []
        self.accuracy_data = []

        self.clip = 1

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

    def backwards_propagate(self, z_data, a_data, expected_output):
        dc_da = self.loss.derivative(a_data[-1], expected_output)
        for i in reversed(range(len(self.layers))):
            dc_da = self.layers[i].backwards_pass(a_data[i], z_data[i], dc_da)
        self.final_dc_da = dc_da

    def fit(
        self,
        data,
        labels,
        epochs,
        learning_rate,
        batch_size=1,
        is_pre_batched=False,
        shuffle_data=True,
        console_updates=True,
        accuracy_function=default_accuracy,
        learning_rate_function=default_learning_rate,
        start_step=0,
    ):
        if console_updates:
            print("Training model...")

        data_size = len(data)

        for i in range(epochs):
            if shuffle_data:
                combined = list(zip(data, labels))
                random.shuffle(combined)
                data, labels = map(list, zip(*combined))

            if not is_pre_batched:
                data_batches = np.array_split(np.array(data), (len(data) + batch_size - 1) // batch_size)
                label_batches = np.array_split(np.array(labels), (len(labels) + batch_size - 1) // batch_size)
            else:
                data_batches = data
                label_batches = labels

            total_loss = 0
            total_correct = 0
            total_examples = 0
            last_time = time.perf_counter()
            for j in range(len(data_batches)):
                current_batch = data_batches[j]
                current_label_batch = label_batches[j]
                batch_len = len(current_batch)

                z_data, a_data = self.forward_propagate(current_batch)
                cur_loss = self.loss(a_data[-1], current_label_batch)
                cur_correct = accuracy_function(a_data[-1], current_label_batch)
                self.loss_data.append(cur_loss / batch_len)
                self.accuracy_data.append(cur_correct / batch_len)
                self.steps.append(j + start_step)
                total_loss += cur_loss
                total_correct += cur_correct

                self.backwards_propagate(z_data, a_data, current_label_batch)

                norm_sqr = 0
                for layer in self.layers:
                    norm_sqr += layer.get_norm()

                norm = np.sqrt(norm_sqr)
                scale = 1
                if norm > self.clip:
                    scale = (self.clip / norm)

                for layer in self.layers:
                    layer.update_weights(learning_rate * learning_rate_function(i * data_size + j + start_step), grad_scale=scale)

                total_examples += batch_len

                # if (j + 1) % 1 == 0 and console_updates:
                #     t = time.perf_counter()
                #     print(f"So far there is loss of {(total_loss / total_examples):.6f} and {(100 * (total_correct / total_examples)):.4f}% accuracy, took {t - last_time:.4f} seconds.")
                #     last_time = t
                    # print(f"Attention Time: {layers.attention_time}")
                    # print(f"Dense Time: {layers.dense_time}")
                    # print(f"Norm Time: {layers.layer_norm_time}")
                    # print(f"Positional Time: {layers.positional_time}")
                    # print(f"Dropout Time: {layers.dropout_time}")

            if console_updates:
                print(f"Finished epoch {i + 1} with an average loss of {(total_loss / total_examples):.6f} and {(100 * (total_correct / total_examples)):.4f}% accuracy.")
        if console_updates:
            print("Finished training model.")

    def test(self, data, labels, accuracy_function=default_accuracy):
        prediction = self.predict(data)
        total_correct = accuracy_function(prediction, labels)
        total_loss = self.loss(prediction, labels)
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

    def save_csv(self, path):
        cur_df = pd.DataFrame({
            "step": self.steps,
            "loss": self.loss_data,
            "accuracy": self.accuracy_data,
        })

        if os.path.exists(path):
            file_df = pd.read_csv(path)
            # merge; current overrides duplicates by step
            merged = pd.concat([file_df, cur_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["step"], keep="last")
        else:
            merged = cur_df

        merged = merged.sort_values("step").reset_index(drop=True)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        merged.to_csv(path, index=False)

        self.steps = merged["step"].astype(int).tolist()
        self.loss_data = merged["loss"].tolist()
        self.accuracy_data = merged["accuracy"].tolist()
