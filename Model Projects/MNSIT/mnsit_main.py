from keras.datasets import mnist
from scratch_model import *
import numpy as np


def load_data():
    (train_images_f, train_labels_f), (test_images_f, test_labels_f) = mnist.load_data()

    # Normalize the image data to values between 0 and 1
    train_images_f = train_images_f.astype('float32') / 255
    test_images_f = test_images_f.astype('float32') / 255

    # Flatten the images from 28x28 to 784-dimensional vectors
    # train_images_f = train_images_f.reshape((-1, 28*28))
    # test_images_f = test_images_f.reshape((-1, 28*28))

    return train_images_f, train_labels_f, test_images_f, test_labels_f


def main():
    save_as = "Models/dense_test"

    train_images, train_labels, test_images, test_labels = load_data()

    ohe_train_labels = np.eye(10)[train_labels]
    ohe_test_labels = np.eye(10)[test_labels]

    # ai_model = model.Model.load(save_as)
    # ai_model = Model(
    #     model_functions.cross_entropy,
    #     (28, 28),
    #     [
    #         layers.Dense(64, model_functions.relu),
    #         layers.Dense(10, model_functions.linear)
    #     ]
    # )
    ai_model = Model(
        model_functions.softmax_cross_entropy,
        (1, 28, 28),
        [
            # layers.Reshape((1, 28, 28)),
            # layers.Convolution(64, (3, 3), model_functions.relu),
            # layers.MaxPooling((2, 2), 2),
            # layers.Convolution(32, (3, 3), model_functions.relu),
            # layers.MaxPooling((2, 2), 2),
            # layers.Dense(128, model_functions.relu),
            # layers.LayerNorm(),
            # layers.Dense(512, model_functions.relu),
            layers.Flatten(),
            layers.Dense(128, model_functions.relu),
            # layers.Dropout(0.5),

            layers.Dense(10, model_functions.cross_entropy_softmax)
        ],
        optimizer=optimizers.AdamW,
        optimizer_args=(0.9, 0.999, 0.0),
    )

    print(ai_model.get_param_num())

    # accuracy = ai_model.test(test_images, ohe_test_labels)
    # print(f"Initial model accuracy is {accuracy * 100}%")
    # print()

    ai_model.fit(train_images, ohe_train_labels, 5, 0.0025, 64)

    print()
    loss, accuracy = ai_model.test(np.array(test_images), ohe_test_labels)
    print(f"Final model accuracy is {accuracy * 100}% and a loss of {loss}")
    # print()

    # print("Saving model...")
    # ai_model.save(save_as)
    # print(f"Saved model as {save_as}.pkl")


if __name__ == "__main__":
    main()
