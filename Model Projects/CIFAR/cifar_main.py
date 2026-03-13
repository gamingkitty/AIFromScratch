from keras.datasets import cifar10
import numpy as np


def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images / 255
    test_images = test_images / 255

    train_images = np.transpose(train_images, (0, 3, 1, 2))
    test_images = np.transpose(test_images, (0, 3, 1, 2))

    return train_images, train_labels.flatten(), test_images, test_labels.flatten()


def main():
    train_labels = np.eye(10)[tr_labels]
    test_labels = np.eye(10)[te_labels]

    # ai_model = Model(
    #     model_functions.softmax_cross_entropy,
    #     (3, 32, 32),
    #     [
    #         layers.Convolution(64, (3, 3), model_functions.relu),
    #         layers.MaxPooling((2, 2), 2),
    #         layers.Convolution(64, (3, 3), model_functions.relu),
    #
    #         layers.Flatten(),
    #         layers.Dense(512, model_functions.relu),
    #         layers.Dense(256, model_functions.relu),
    #         layers.Dense(128, model_functions.relu),
    #
    #         layers.Dense(10, model_functions.cross_entropy_softmax)
    #     ],
    #     optimizer=optimizers.AdamW,
    #     optimizer_args=(0.9, 0.999, 0.0),
    # )

    ai_model = Model.load("Models/cifar_convolution")

    print(f"Param count: {ai_model.get_param_num()}")

    # accuracy = ai_model.test(test_images, test_labels)
    # print(f"Initial accuracy: {accuracy * 100:.4f}%")

    # ai_model.fit(tr_images, train_labels, 10, 0.0005, batch_size=64)

    # ai_model.save("Models/cifar_convolution")

    loss, accuracy = ai_model.test(np.array(te_images), test_labels)
    print(f"Final accuracy: {accuracy * 100:.4f}%, final loss: {loss:.6f}")


tr_images, tr_labels, te_images, te_labels = load_cifar10()

from scratch_model import *
import numpy as np
if __name__ == "__main__":
    main()
