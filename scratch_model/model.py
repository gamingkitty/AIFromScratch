import time

from ._gpu_patch import xp as np
import pickle
import random
from scratch_model import layers
from scratch_model import optimizers
import pandas as pd
import os
import matplotlib.pyplot as plt


def default_accuracy(prediction, label):
    pred_classes = prediction.argmax(axis=-1)
    true_classes = label.argmax(axis=-1)

    return int((pred_classes == true_classes).sum().item())


def default_learning_rate(step):
    return 1


class Model:
    def __init__(self, loss_function, input_shape, model_layers, optimizer=optimizers.Adam, optimizer_args=(), dtype=np.float32):
        self.input_shape = input_shape
        self.input_num = np.prod(input_shape)

        self.layers = model_layers

        self.loss = loss_function
        # Currently in here for visualization purposes
        self.final_dc_da = None

        prev_output_shape = input_shape
        for layer in self.layers:
            layer.init_weights(prev_output_shape, optimizer, optimizer_args=optimizer_args, dtype=dtype)
            prev_output_shape = layer.get_output_shape()

        self.output_shape = prev_output_shape
        self.output_num = np.prod(prev_output_shape)
        self.dtype = dtype

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
            # print(self.layers[i])
            # print(last_value.dtype)
            z_data_current, a_data_current = self.layers[i].forward_pass(last_value)
            last_value = a_data_current
            # print(last_value.dtype)
            # print()
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
        data_augmentation_function=None,
        data_save_file=None,
        steps_to_update_weights=1,
        end_update_weights=True,
    ):
        if console_updates:
            print("Training model...")

        for i in range(epochs):
            if shuffle_data:
                combined = list(zip(data, labels))
                random.shuffle(combined)
                data, labels = map(list, zip(*combined))

            if data_augmentation_function is not None:
                augmented_data = data_augmentation_function(data)
            else:
                augmented_data = data

            if not is_pre_batched:
                data_batches = np.array_split(np.array(augmented_data), (len(augmented_data) + batch_size - 1) // batch_size)
                label_batches = np.array_split(np.array(labels), (len(labels) + batch_size - 1) // batch_size)
            else:
                data_batches = augmented_data
                label_batches = labels

            num_batches = len(data_batches)

            total_loss = 0
            total_correct = 0
            total_examples = 0
            last_time = time.perf_counter()
            update_counter = 0
            update_batch_total = 0
            for j in range(len(data_batches)):
                current_batch = data_batches[j]
                current_label_batch = label_batches[j]
                batch_len = len(current_batch)

                update_batch_total += batch_len

                z_data, a_data = self.forward_propagate(current_batch)

                cur_loss = self.loss(a_data[-1], current_label_batch)
                cur_correct = accuracy_function(a_data[-1], current_label_batch)
                self.loss_data.append(cur_loss / batch_len)
                self.accuracy_data.append(cur_correct / batch_len)
                cur_step = (i * num_batches) + j + start_step
                self.steps.append(cur_step)
                total_loss += cur_loss
                total_correct += cur_correct

                self.backwards_propagate(z_data, a_data, current_label_batch)

                update_counter += 1

                if update_counter >= steps_to_update_weights:
                    self.update_weights(learning_rate * learning_rate_function(cur_step), update_batch_total)

                    update_counter = 0
                    update_batch_total = 0

                total_examples += batch_len

                if (j + 1) % 1 == 0:
                    t = time.perf_counter()
                    print(f"So far there is loss of {(total_loss / total_examples):.6f} and {(100 * (total_correct / total_examples)):.4f}% accuracy, took {t - last_time:.4f} seconds.")
                    # print(f"Attention Time: Forward: {layers.forward_attention}, Backward: {layers.backward_attention}, Update: {layers.update_attention}")
                    # print(f"Dense Time: Forward: {layers.forward_dense}, Backward: {layers.backward_dense}, Update: {layers.update_dense}")
                    # print(f"Norm Time: Forward: {layers.forward_norm}, Backward: {layers.backward_norm}, Update: {layers.update_norm}")
                    # layers.forward_attention = 0
                    # layers.backward_attention = 0
                    # layers.update_attention = 0
                    # layers.forward_dense = 0
                    # layers.backward_dense = 0
                    # layers.update_dense = 0
                    # layers.forward_norm = 0
                    # layers.backward_norm = 0
                    # layers.update_norm = 0
                    last_time = t

            if update_counter > 0 and end_update_weights:
                self.update_weights(learning_rate * learning_rate_function(((i + 1) * num_batches) + start_step), update_batch_total)

            if data_save_file is not None:
                self.save_csv(data_save_file)
                self.steps = []
                self.loss_data = []
                self.accuracy_data = []

            if console_updates:
                print(f"Finished epoch {i + 1} with an average loss of {(total_loss / total_examples):.6f} and {(100 * (total_correct / total_examples)):.4f}% accuracy.")
        if console_updates:
            print("Finished training model.")

    def update_weights(self, learning_rate, batches_since_update):
        norm_sqr = 0
        for layer in self.layers:
            n = layer.get_norm()
            norm_sqr += n

        norm = np.sqrt(norm_sqr) / batches_since_update
        scale = 1

        if norm > self.clip:
            scale = self.clip / norm

        for layer in self.layers:
            layer.update_weights(learning_rate, grad_scale=scale / batches_since_update)

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
                    arr = np.frombuffer(rawdata, dtype=dtype).copy()
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

    def set_weights_dtype(self, dtype):
        for layer in self.layers:
            layer.set_weights_dtype(dtype)

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

    @classmethod
    def plot_csv(cls, path, ema_span=100):
        df = pd.read_csv(path).sort_values("step").reset_index(drop=True)

        df["ema_loss"] = df["loss"].ewm(span=ema_span, adjust=False).mean()

        print((df["ema_loss"][len(df["ema_loss"]) - 1] - df["ema_loss"][len(df["ema_loss"]) - 30001]) / 30000)

        fig, axes = plt.subplots(1, 2, figsize=(16, 4))

        axes[0].plot(df["step"], df["loss"], label="loss")
        axes[0].plot(df["step"], df["ema_loss"], label=f"ema_loss (span={ema_span})")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Loss vs Step")
        axes[0].legend()

        axes[1].plot(df["step"], df["accuracy"], label="accuracy")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("Accuracy vs Step")
        axes[1].legend()

        fig.tight_layout()
        plt.show()

    def save_weights(self, path):
        weights = []
        for layer in self.layers:
            weights += layer.get_weights()
        np.savez(path, *weights)

    def save_weights_int8(self, path):
        import numpy as real_np

        arrays_to_save = {}

        weights = []
        for layer in self.layers:
            weights += layer.get_weights()

        arrays_to_save["num_weights"] = real_np.array(len(weights), dtype=real_np.int64)

        for i, w in enumerate(weights):
            if hasattr(w, "get"):
                w_np = w.get()
            else:
                w_np = real_np.asarray(w)

            original_dtype = str(w_np.dtype)
            original_shape = real_np.array(w_np.shape, dtype=real_np.int64)

            w_float = w_np.astype(real_np.float32)

            max_abs = real_np.max(real_np.abs(w_float))

            if max_abs == 0:
                scale = real_np.float32(1.0)
                q = real_np.zeros_like(w_float, dtype=real_np.int8)
            else:
                scale = real_np.float32(max_abs / 127.0)
                q = real_np.round(w_float / scale)
                q = real_np.clip(q, -127, 127).astype(real_np.int8)

            arrays_to_save[f"weight_{i}"] = q
            arrays_to_save[f"scale_{i}"] = real_np.array(scale, dtype=real_np.float32)
            arrays_to_save[f"shape_{i}"] = original_shape
            arrays_to_save[f"dtype_{i}"] = real_np.array(original_dtype)

        real_np.savez_compressed(path, **arrays_to_save)

    def load_weights_int8(self, path, dtype=None):
        import numpy as real_np

        data = real_np.load(path, allow_pickle=True)

        num_weights = int(data["num_weights"])
        loaded_weights = []

        for i in range(num_weights):
            q = data[f"weight_{i}"]
            scale = real_np.float32(data[f"scale_{i}"])
            shape = tuple(data[f"shape_{i}"])

            if dtype is None:
                out_dtype = real_np.dtype(str(data[f"dtype_{i}"]))
            else:
                out_dtype = dtype

            w = q.astype(real_np.float32) * scale
            w = w.reshape(shape).astype(out_dtype)

            loaded_weights.append(w)

        weight_index = 0

        for layer in self.layers:
            old_layer_weights = layer.get_weights()
            num_layer_weights = len(old_layer_weights)

            new_layer_weights = loaded_weights[
                weight_index:weight_index + num_layer_weights
            ]

            # Convert NumPy weights back to CuPy if the current layer uses CuPy weights
            if num_layer_weights > 0 and hasattr(old_layer_weights[0], "get"):
                import cupy as cp

                if dtype is None:
                    target_dtype = old_layer_weights[0].dtype
                else:
                    target_dtype = dtype

                new_layer_weights = [
                    cp.asarray(w, dtype=target_dtype)
                    for w in new_layer_weights
                ]

            layer.set_weights(new_layer_weights)

            weight_index += num_layer_weights

        if weight_index != num_weights:
            raise ValueError(
                f"Loaded {num_weights} weights, but model used {weight_index}."
            )
