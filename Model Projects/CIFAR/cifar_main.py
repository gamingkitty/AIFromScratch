import numpy as np
from scratch_model import model
from keras.datasets import cifar10


def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images / 255
    test_images = test_images / 255

    train_images = np.transpose(train_images, (0, 3, 1, 2))
    test_images = np.transpose(test_images, (0, 3, 1, 2))

    return train_images, train_labels.flatten(), test_images, test_labels.flatten()


def main():
    train_images, train_labels, test_images, test_labels = load_cifar10()

    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]

    ai_model = model.Model.load("../Model Projects/Models/cifar_conv_4")
    # ai_model = model.Model(
    #     model_functions.cross_entropy,
    #     (3, 32, 32),
    #     layers.Convolution(32, (3, 3), model_functions.relu),
    #     layers.MaxPooling((2, 2), 2),
    #     layers.Convolution(64, (3, 3), model_functions.relu),
    #     layers.Dense(128, model_functions.relu),
    #     layers.Dense(64, model_functions.relu),
    #     layers.Dense(10, model_functions.softmax),
    # )
    # ai_model = model.Model(
    #     model_functions.cross_entropy,
    #     (3, 32, 32),
    #     layers.Dense(128, model_functions.relu),
    #     layers.Dense(64, model_functions.relu),
    #     layers.Dense(10, model_functions.softmax)
    # )

    # accuracy = ai_model.test(test_images, test_labels)
    # print(f"Initial accuracy: {accuracy * 100:.4f}%")

    ai_model.fit(train_images, train_labels, 3, 0.01, 32)

    accuracy = ai_model.test(test_images, test_labels)
    print(f"Final accuracy: {accuracy * 100:.4f}%")

    ai_model.save("Models/cifar_conv_5")


if __name__ == "__main__":
    main()
