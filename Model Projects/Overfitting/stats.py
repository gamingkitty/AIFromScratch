from scratch_model import model
from scratch_model import layers
from scratch_model import model_functions
import numpy as np
import pickle
from keras.datasets import mnist


def get_random_images_by_class(images, labels, n_per_class, filepath="random_images.pkl", seed=42):
    np.random.seed(seed)
    selected_images = []
    selected_labels = []

    classes = np.unique(labels)
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        chosen_indices = np.random.choice(cls_indices, n_per_class, replace=False)
        selected_images.append(images[chosen_indices])
        selected_labels.append(labels[chosen_indices])

    final_images = np.concatenate(selected_images, axis=0)
    final_labels = np.concatenate(selected_labels, axis=0)

    # Shuffle while preserving correspondence
    shuffle_idx = np.random.permutation(len(final_labels))
    final_images = final_images[shuffle_idx]
    final_labels = final_labels[shuffle_idx]

    with open(filepath, "wb") as f:
        pickle.dump((final_images, final_labels), f)

    return final_images, final_labels


def load_random_images(filepath="random_images.pkl"):
    with open(filepath, "rb") as f:
        images, labels = pickle.load(f)
    return images, labels


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
    train_images, train_labels, test_images, test_labels = load_data()
    possible_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    train_images, train_labels = load_random_images()
    train_images = train_images.astype(float) / 255.0

    ohe_test_labels = np.zeros((len(test_labels), len(possible_labels)))
    ohe_train_labels = np.zeros((len(train_labels), len(possible_labels)))

    for i in range(len(ohe_train_labels)):
        ohe_train_labels[i][train_labels[i]] = 1

    for i in range(len(ohe_test_labels)):
        ohe_test_labels[i][test_labels[i]] = 1

    differences = []

    for i in range(1):
        ai_model = model.Model(
            model_functions.categorical_entropy,
            (28, 28),
            layers.Dense(10000, model_functions.relu),
            layers.Dense(10, model_functions.softmax)
        )

        ai_model.fit(train_images, ohe_train_labels, 250, 0.01, 1, console_updates=True)

        train_accuracy = ai_model.test(train_images, ohe_train_labels)
        test_accuracy = ai_model.test(test_images, ohe_test_labels)
        print(f"Train accuracy: {train_accuracy * 100}%")
        print(f"Test accuracy: {test_accuracy * 100}%")
        print(f"Difference: {(train_accuracy - test_accuracy) * 100}%")
        print()
        differences.append(train_accuracy - test_accuracy)

    print(differences)


if __name__ == "__main__":
    main()
