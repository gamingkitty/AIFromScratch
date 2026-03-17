from keras.datasets import cifar10
import numpy as np
import math


def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

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


def lr_percent_cosine_step(step, total_steps=1563*40, warmup_steps=1000, min_percent=0.05):
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


def main():
    train_labels = np.eye(10)[tr_labels]
    test_labels = np.eye(10)[te_labels]

    ai_model = Model(
        model_functions.softmax_cross_entropy,
        (3, 32, 32),
        [
            layers.Convolution(32, (3, 3), model_functions.relu, padding=1),
            layers.Convolution(32, (3, 3), model_functions.relu, padding=1),
            layers.MaxPooling((2, 2), 2),

            layers.Dropout(0.2),

            # layers.LayerNorm(axis=(-3, -2, -1)),

            layers.Convolution(64, (3, 3), model_functions.relu, padding=1),
            layers.Convolution(64, (3, 3), model_functions.relu, padding=1),
            layers.MaxPooling((2, 2), 2),

            layers.Dropout(0.3),

            # layers.LayerNorm(axis=(-3, -2, -1)),
            #
            layers.Convolution(128, (3, 3), model_functions.relu, padding=1),
            layers.Convolution(128, (3, 3), model_functions.relu, padding=1),
            layers.MaxPooling((2, 2), 2),
            # layers.Convolution(128, (3, 3), model_functions.relu, padding=1),

            # layers.Mean(axis=(-2, -1)),
            layers.Flatten(),
            layers.Dropout(0.4),

            layers.Dense(128, model_functions.relu),

            layers.Dropout(0.5),

            layers.Dense(10, model_functions.cross_entropy_softmax)
        ],
        optimizer=optimizers.AdamW,
        optimizer_args=(0.9, 0.999, 0.0),
        dtype=np.float32,
    )

    # ai_model = Model.load("Models/cifar_convolution")

    print(f"Param count: {ai_model.get_param_num()}")

    # accuracy = ai_model.test(test_images, test_labels)
    # print(f"Initial accuracy: {accuracy * 100:.4f}%")

    ai_model.fit(
        tr_images,
        train_labels,
        40,
        0.001,
        batch_size=32,
        data_augmentation_function=random_horizontal_flip,
        learning_rate_function=lr_percent_cosine_step,
        data_save_file="Data/cifar_convolution_7"
    )

    ai_model.save("Models/cifar_convolution_7")

    loss, accuracy = ai_model.test(np.array(te_images), test_labels)
    print(f"Final accuracy: {accuracy * 100:.4f}%, final loss: {loss:.6f}")


tr_images, tr_labels, te_images, te_labels = load_cifar10()

from scratch_model import *
import numpy as np
if __name__ == "__main__":
    main()

