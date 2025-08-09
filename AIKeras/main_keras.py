import numpy as np
import keras
from keras import layers
from keras.datasets import mnist, cifar10


def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images / 255
    test_images = test_images / 255

    # train_images = np.transpose(train_images, (0, 3, 1, 2))
    # test_images = np.transpose(test_images, (0, 3, 1, 2))

    return train_images, train_labels.flatten(), test_images, test_labels.flatten()


def load_data():
    (train_images_f, train_labels_f), (test_images_f, test_labels_f) = mnist.load_data()

    # Normalize the image data to values between 0 and 1
    train_images_f = train_images_f.astype('float32') / 255
    test_images_f = test_images_f.astype('float32') / 255

    # Flatten the images from 28x28 to 784-dimensional vectors
    train_images_f = train_images_f.reshape((-1, 28, 28, 1))
    test_images_f = test_images_f.reshape((-1, 28, 28, 1))

    return train_images_f, train_labels_f, test_images_f, test_labels_f


def main():
    train_images, train_labels, test_images, test_labels = load_data()  # load_cifar10()

    ohe_train_labels = np.zeros((len(train_labels), 10))
    ohe_test_labels = np.zeros((len(test_labels), 10))

    for i in range(len(ohe_train_labels)):
        ohe_train_labels[i][train_labels[i]] = 1

    for i in range(len(ohe_test_labels)):
        ohe_test_labels[i][test_labels[i]] = 1

    model = keras.Sequential(
        [layers.Input((28, 28, 1)),
         # layers.Conv2D(32, (3, 3), activation='relu'),
         # layers.MaxPooling2D((2, 2)),
         #
         # layers.Conv2D(64, (3, 3), activation='relu'),
         # # layers.MaxPooling2D((2, 2)),
         #
         layers.Flatten(),

         layers.Dense(512, activation='relu'),
         layers.Dense(256, activation='relu'),
         layers.Dense(128, activation='relu'),

         layers.Dense(10, activation='softmax')]
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    model.fit(train_images, ohe_train_labels, batch_size=32, epochs=7)

    num_correct = 0
    predictions = model.predict(test_images)
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == test_labels[i]:
            num_correct += 1

    print(f"Accuracy of model: {100 * num_correct/len(test_images)}%")


if __name__ == "__main__":
    main()


