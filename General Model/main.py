import numpy as np
import model
import layers
import activation_functions
from keras.datasets import mnist


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
    save_as = "Models/model_convolution_multiple_kernels"

    train_images, train_labels, test_images, test_labels = load_data()
    possible_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    ohe_train_labels = np.zeros((len(train_labels), len(possible_labels)))
    ohe_test_labels = np.zeros((len(test_labels), len(possible_labels)))

    for i in range(len(ohe_train_labels)):
        ohe_train_labels[i][train_labels[i]] = 1

    for i in range(len(ohe_test_labels)):
        ohe_test_labels[i][test_labels[i]] = 1

    # ai_model = model.Model.load(save_as)
    # ai_model = model.Model(
    #     (1, 28, 28),
    #     layers.Convolution(2, (3, 3), activation_functions.relu, activation_functions.relu_derivative),
    #     layers.Dense(64, activation_functions.relu, activation_functions.relu_derivative),
    #     layers.Dense(32, activation_functions.relu, activation_functions.relu_derivative),
    #     layers.Dense(10, activation_functions.softmax, activation_functions.softmax_derivative)
    # )
    ai_model = model.Model(
        (1, 28, 28),
        layers.Convolution(32, (3, 3), activation_functions.relu),
        layers.MaxPooling((2, 2), 2),

        layers.Convolution(64, (3, 3), activation_functions.relu),
        layers.MaxPooling((2, 2), 2),

        layers.Dense(64, activation_functions.relu),
        layers.Dropout(0.5),

        layers.Dense(10, activation_functions.softmax)
    )

    # accuracy = ai_model.test(test_images, ohe_test_labels)
    # print(f"Initial model accuracy is {accuracy * 100}%")
    # print()

    ai_model.fit(train_images, ohe_train_labels, 1, 0.02, 32)

    print()
    accuracy = ai_model.test(test_images, ohe_test_labels)
    print(f"Final model accuracy is {accuracy * 100}%")
    print()

    print("Saving model...")
    ai_model.save(save_as)
    print(f"Saved model as {save_as}.pkl")


if __name__ == "__main__":
    main()
