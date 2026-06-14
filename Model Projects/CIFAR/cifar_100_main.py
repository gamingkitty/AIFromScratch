from keras.datasets import cifar100
import cupy as cp
import numpy as np
import math


def load_cifar100():
    (train_images, train_labels), (test_images, test_labels) = cifar100.load_data(
        label_mode="fine"
    )

    train_images = train_images / 255
    test_images = test_images / 255

    train_images = np.transpose(train_images, (0, 3, 1, 2)).astype(np.float32)
    test_images = np.transpose(test_images, (0, 3, 1, 2)).astype(np.float32)

    train_images = random_horizontal_flip(train_images, 0.5)

    return train_images, train_labels.flatten(), test_images, test_labels.flatten()


def random_horizontal_flip(images, p=0.5):
    flipped = np.array(images).copy()
    mask = np.random.rand(flipped.shape[0]) < p
    flipped[mask] = flipped[mask, :, :, ::-1]
    return flipped


def random_crop(images, padding=4):
    """
    Randomly crops CIFAR images after padding them.

    Input shape:
        images: (N, C, H, W)

    Output shape:
        (N, C, H, W)
    """
    images = np.array(images).copy()

    n, c, h, w = images.shape

    padded = np.pad(
        images,
        pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    cropped = np.empty_like(images)

    for i in range(n):
        y = np.random.randint(0, padding * 2 + 1)
        x = np.random.randint(0, padding * 2 + 1)

        cropped[i] = padded[i, :, y:y + h, x:x + w]

    return cropped


def data_augmentation(images):
    images = random_horizontal_flip(images, p=0.5)
    images = random_crop(images, padding=4)
    return images


def lr_percent_cosine_step(step, total_steps=1563*250, warmup_steps=2000, min_percent=0.05):
    if total_steps <= 1:
        return 1.0

    step = max(0, min(int(step), total_steps - 1))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps

    denom = total_steps - warmup_steps
    if denom <= 1:
        return 1.0

    t = (step - warmup_steps) / denom
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_percent + (1.0 - min_percent) * cosine


def create_block(d_model, d_feed_forward, heads):
    return (
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.Attention(int(d_model / heads), int(d_model / heads), heads, use_rope=True, use_kv_cache=True),
            layers.TimeDistributedDense(d_model)
        ),
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.TimeDistributedDense(d_feed_forward, model_functions.gelu),
            layers.TimeDistributedDense(d_model)
        ),
    )


def main():
    train_labels = np.eye(100)[tr_labels]
    test_labels = np.eye(100)[te_labels]

    # ai_model = Model(
    #     model_functions.softmax_cross_entropy,
    #     (3, 32, 32),
    #     [
    #         layers.Convolution(64, (3, 3), model_functions.relu, padding=1),
    #         layers.Convolution(64, (3, 3), model_functions.relu, padding=1),
    #         layers.MaxPooling((2, 2), 2),
    #
    #         layers.Dropout(0.1),
    #
    #         # layers.LayerNorm(axis=(-3, -2, -1)),
    #
    #         layers.Convolution(128, (3, 3), model_functions.relu, padding=1),
    #         layers.Convolution(128, (3, 3), model_functions.relu, padding=1),
    #         layers.MaxPooling((2, 2), 2),
    #
    #         layers.Dropout(0.2),
    #
    #         # layers.LayerNorm(axis=(-3, -2, -1)),
    #         #
    #         layers.Convolution(256, (3, 3), model_functions.relu, padding=1),
    #         layers.Convolution(256, (3, 3), model_functions.relu, padding=1),
    #         layers.MaxPooling((2, 2), 2),
    #
    #         layers.Convolution(512, (3, 3), model_functions.relu, padding=1),
    #
    #         layers.Mean(axis=(-2, -1)),
    #
    #         layers.Dense(256, model_functions.relu),
    #
    #         layers.Dropout(0.3),
    #
    #         layers.Dense(100, model_functions.cross_entropy_softmax)
    #     ],
    #     optimizer=optimizers.AdamW,
    #     optimizer_args=(0.9, 0.999, 0.0001),
    #     dtype=np.float32,
    # )

    blocks = 6
    d_model = 128
    d_feed_forward = d_model * 4
    heads = 4

    ai_model = Model(
        model_functions.softmax_cross_entropy,
        (3, 32, 32),
        [
            layers.Convolution(32, (3, 3), model_functions.relu, padding=1),
            layers.Convolution(32, (3, 3), model_functions.relu, padding=1),
            layers.MaxPooling((2, 2), 2),

            layers.Convolution(64, (3, 3), model_functions.relu, padding=1),
            layers.Convolution(64, (3, 3), model_functions.relu, padding=1),
            layers.MaxPooling((2, 2), 2),

            layers.Convolution(128, (3, 3), model_functions.relu, padding=1),
            layers.Convolution(128, (3, 3), model_functions.relu, padding=1),

            # Shape (128, 8, 8)

            layers.Reshape((128, 64)),
            layers.Transpose((1, 0)),

            # Shape (64, 128)
            *[
                layer
                for _ in range(blocks)
                for layer in create_block(d_model, d_feed_forward, heads)
            ],
            # Shape (64, 128)
            layers.Flatten(),

            layers.Dense(100, model_functions.cross_entropy_softmax)
        ],
        optimizer=optimizers.AdamW,
        optimizer_args=(0.9, 0.999, 0.0001),
        dtype=cp.float32,
    )

    # ai_model = Model.load("Models/cifar_100_v2_40e")

    print(f"Param count: {ai_model.get_param_num()}")

    # accuracy = ai_model.test(test_images, test_labels)
    # print(f"Initial accuracy: {accuracy * 100:.4f}%")

    ai_model.fit(
        tr_images,
        train_labels,
        10,
        0.001,
        batch_size=32,
        start_step=0,
        data_augmentation_function=data_augmentation,
        learning_rate_function=lr_percent_cosine_step,
        data_save_file="Data/cifar_100_transformer"
    )

    ai_model.save("Models/cifar_100_transformer")

    loss, accuracy = ai_model.test(cp.array(te_images[:5000]), cp.array(test_labels[:5000]))
    print(f"Final accuracy: {accuracy * 100:.4f}%, final loss: {loss:.6f}")


tr_images, tr_labels, te_images, te_labels = load_cifar100()

from scratch_model import *
import numpy as np
if __name__ == "__main__":
    main()

