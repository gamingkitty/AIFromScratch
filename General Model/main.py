import numpy as np


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return (x > 0).astype(int)


def main():
    array = np.array([[1, 0.5], [2, 3]])
    array2 = np.array([4, 2])
    print(array * array2)
    print("Hello, World!")


if __name__ == "__main__":
    main()
