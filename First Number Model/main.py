# This file is just attempting to make a simple model without generalizing it to everything, so I can learn more about
# backpropagation before jumping straight into making a model class.

import numpy as np
from keras.datasets import mnist


def load_data():
    (train_images_f, train_labels_f), (test_images_f, test_labels_f) = mnist.load_data()

    # Normalize the image data to values between 0 and 1
    train_images_f = train_images_f.astype('float32') / 255
    test_images_f = test_images_f.astype('float32') / 255

    # Flatten the images from 28x28 to 784-dimensional vectors
    train_images_f = train_images_f.reshape((-1, 28*28))
    test_images_f = test_images_f.reshape((-1, 28*28))

    return train_images_f, train_labels_f, test_images_f, test_labels_f


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return np.maximum(x, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def loss(output, label):
    return np.sum(np.power(label - output, 2)) / len(output)


def softmax_derivative(x):
    softmax_vals = softmax(x)

    n = len(x)
    jacobian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                jacobian[i, j] = softmax_vals[i] * (1 - softmax_vals[i])
            else:
                jacobian[i, j] = -softmax_vals[i] * softmax_vals[j]

    return jacobian


def forward_pass(input_data, input_to_hidden_weights, hidden_to_output_weights):
    # For bias in input layer
    input_data = np.append(input_data, 1)

    z_data = []
    a_data = []

    output = np.dot(input_data, input_to_hidden_weights)
    z_data.append(output)
    output = relu(output)

    # For bias in hidden layer
    output = np.append(output, 1)

    a_data.append(output)

    output = np.dot(output, hidden_to_output_weights)
    z_data.append(output)
    output = softmax(output)
    a_data.append(output)

    return z_data, a_data


def backwards_pass(z_data, a_data, label, hidden_to_output_weights, input_data):
    # For bias in input layer
    input_data = np.append(input_data, 1)

    # Gradient of cost function with respect to model output neurons.
    dc_da2 = -(1 / 5) * (label - a_data[1])

    softmax_derivative_jacobian = softmax_derivative(z_data[1])

    # Gradient of cost function with respect to output neurons before activation function (softmax)
    dc_dz2 = np.dot(dc_da2, softmax_derivative_jacobian)

    # Computed by multiplying dc_dz2 and dz2_dw
    last_layer_gradient = dc_dz2 * a_data[0][:, np.newaxis]

    # dz2_da1 (which is just the hidden_to_output_weights) * da1_dz1 (the derivative of the activation function, relu)
    # Need to exclude last weight, as that is for bias
    dz2_dz1 = hidden_to_output_weights[:-1, :] * relu_derivative(z_data[0])[:, np.newaxis]
    dc_dz1 = np.dot(dc_dz2, dz2_dz1.T)

    # The input_data, or z_0, is dz1_dw
    first_layer_gradient = dc_dz1 * input_data[:, np.newaxis]

    return first_layer_gradient, last_layer_gradient


def main():
    # Plus 1 for the biases
    input_num = 784 + 1
    hidden_neuron_num = 16 + 1
    output_num = 10

    learning_rate = 0.01

    save_as = "model_weights_bias"

    # -1 in input to hidden_weights so nothing is connected to the bias neuron.
    input_to_hidden_weights = np.random.rand(input_num, hidden_neuron_num - 1) - 0.5
    hidden_to_output_weights = np.random.rand(hidden_neuron_num, output_num) - 0.5
    # weights = np.load("model_weights.npz")
    # input_to_hidden_weights = weights["input_to_hidden"]
    # hidden_to_output_weights = weights["hidden_to_output"]

    train_images, train_labels, test_images, test_labels = load_data()
    possible_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    ohe_train_labels = np.zeros((len(train_labels), len(possible_labels)))
    ohe_test_labels = np.zeros((len(test_labels), len(possible_labels)))

    for i in range(len(ohe_train_labels)):
        ohe_train_labels[i][train_labels[i]] = 1

    for i in range(len(ohe_test_labels)):
        ohe_test_labels[i][test_labels[i]] = 1

    print("Testing initial model...")
    total_loss = 0
    num_correct = 0
    for i in range(len(test_images)):
        z_data, a_data = forward_pass(test_images[i], input_to_hidden_weights, hidden_to_output_weights)
        prediction = a_data[1]
        total_loss += loss(prediction, ohe_test_labels[i])
        if np.argmax(prediction) == np.argmax(ohe_test_labels[i]):
            num_correct += 1

    print(f"Initial model has loss of {total_loss / len(test_images)} with the number of correct guesses being {num_correct}/{len(test_images)}")
    print()

    print("Starting training...")
    for i in range(30):
        total_loss = 0
        total_correct = 0
        for j in range(len(train_images)):
            z_data, a_data = forward_pass(train_images[j], input_to_hidden_weights, hidden_to_output_weights)
            if np.argmax(a_data[1]) == np.argmax(ohe_train_labels[j]):
                total_correct += 1
            total_loss += loss(a_data[1], ohe_train_labels[j])
            first_layer_gradient, last_layer_gradient = backwards_pass(z_data, a_data, ohe_train_labels[j], hidden_to_output_weights, train_images[j])
            input_to_hidden_weights -= first_layer_gradient * learning_rate
            hidden_to_output_weights -= last_layer_gradient * learning_rate
        print(f"Finished epoch {i + 1} with an average loss of {total_loss / len(train_images)} and num correct {total_correct}/{len(train_images)}")
    print()

    print("Testing final model...")
    total_loss = 0
    num_correct = 0
    for i in range(len(test_images)):
        z_data, a_data = forward_pass(test_images[i], input_to_hidden_weights, hidden_to_output_weights)
        prediction = a_data[1]
        total_loss += loss(prediction, ohe_test_labels[i])
        if np.argmax(prediction) == np.argmax(ohe_test_labels[i]):
            num_correct += 1

    print(f"final model has loss of {total_loss / len(test_images)} with the number of correct guesses being {num_correct}/{len(test_images)}")
    print("Saving model...")
    np.savez(save_as + ".npz", input_to_hidden=input_to_hidden_weights, hidden_to_output=hidden_to_output_weights)
    print(f"Saved model as {save_as}.npz")


if __name__ == "__main__":
    main()
