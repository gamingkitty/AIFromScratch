import os
import numpy as np
import keras
from keras import layers
from keras.datasets import mnist
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from scipy.stats import linregress, t


def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    return train_images, train_labels.flatten(), test_images, test_labels.flatten()


def save_cifar_subset(train_images, train_labels, test_images, test_labels, n_per_class=50, path="cifar_subset.pkl"):
    num_classes = 10
    selected_train_images = []
    selected_train_labels = []

    for class_idx in range(num_classes):
        indices = np.where(train_labels == class_idx)[0]
        chosen = np.random.choice(indices, size=n_per_class, replace=False)
        selected_train_images.append(train_images[chosen])
        selected_train_labels.append(train_labels[chosen])

    selected_train_images = np.concatenate(selected_train_images, axis=0)
    selected_train_labels = np.concatenate(selected_train_labels, axis=0)

    with open(path, "wb") as f:
        pickle.dump({
            "train_images": selected_train_images,
            "train_labels": selected_train_labels,
            "test_images": test_images,
            "test_labels": test_labels,
        }, f)


def load_saved_subset(path="cifar_subset.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["train_images"], data["train_labels"], data["test_images"], data["test_labels"]


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


def add_to_pickle(pickle_file, new_data):
    if not isinstance(new_data, dict):
        raise ValueError("new_data must be a dictionary.")

    # Load existing data if file exists and is not empty
    if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
        with open(pickle_file, 'rb') as f:
            try:
                existing_data = pickle.load(f)
                if not isinstance(existing_data, dict):
                    raise ValueError("Pickle file does not contain a dictionary.")
            except Exception:
                existing_data = {}
    else:
        existing_data = {}

    # Update the dictionary
    existing_data.update(new_data)

    # Save it back to the pickle file
    with open(pickle_file, 'wb') as f:
        pickle.dump(existing_data, f)


def load_pickle_dict(pickle_file):
    if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
        with open(pickle_file, 'rb') as f:
            try:
                data = pickle.load(f)
                if isinstance(data, dict):
                    for k in list(data.keys()):
                        if k < 32 or k >= 20000 or k % 1000 == 0:
                            data.pop(k)
                    return data
                else:
                    raise ValueError("Pickle file does not contain a dictionary.")
            except Exception:
                pass
    return {}


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
    # train_images, train_labels, test_images, test_labels = load_data()

    # train_images, train_labels = get_random_images_by_class(train_images, train_labels, 50)

    # train_images, train_labels, test_images, test_labels = load_cifar10()
    #
    # save_cifar_subset(train_images, train_labels, test_images, test_labels, 10)

    train_images, train_labels, test_images, test_labels = load_saved_subset()

    ohe_train_labels = np.zeros((len(train_labels), 10))
    ohe_test_labels = np.zeros((len(test_labels), 10))

    for i in range(len(ohe_train_labels)):
        ohe_train_labels[i][train_labels[i]] = 1

    for i in range(len(ohe_test_labels)):
        ohe_test_labels[i][test_labels[i]] = 1

    param_nums = [1040861, 1289738, 1289738, 1305761, 1338083, 2899210, 3368507, 5151754, 5151754, 5183713, 5247907, 8047370, 10930388, 11586058, 15767818, 20592650, 20656481, 20784419, 32171530, 35384511, 38925578, 46322698, 54362890] # 82341898, 82469473, 82724899] #, 329310218] commented because only did one of them and is kind of an outlier rn

    accuracy_difs = [0.7535000000000001, 0.743, 0.7868999999999999, 0.7494000000000001, 0.7562, 0.7531, 0.7727999999999999, 0.7702, 0.7782, 0.7976, 0.7798, 0.744, 0.774, 0.7482, 0.7688, 0.7761, 0.8022, 0.7864, 0.7662, 0.7951, 0.7558, 0.8174, 0.7971] # 0.7859, 0.7928, 0.7606999999999999] #, 0.7773]

    # for k in range(10, 14):
    #     model = keras.Sequential(
    #         [layers.Input((32, 32, 3)),
    #          layers.Conv2D(int(32 * k), (3, 3), activation='relu'),
    #          layers.MaxPooling2D((2, 2)),
    #
    #          layers.Conv2D(int(64 * k), (3, 3), activation='relu'),
    #          layers.MaxPooling2D((2, 2)),
    #
    #          layers.Flatten(),
    #
    #          layers.Dense(int(128 * k), activation='relu'),
    #          layers.Dense(int(64 * k), activation='relu'),
    #
    #          layers.Dense(10, activation='softmax')]
    #     )
    #
    #     model.compile(optimizer='adam', loss='categorical_crossentropy')
    #
    #     model.fit(train_images, ohe_train_labels, batch_size=32, epochs=50)
    #
    #     num_correct = 0
    #     predictions = model.predict(train_images)
    #     for i in range(len(predictions)):
    #         if np.argmax(predictions[i]) == train_labels[i]:
    #             num_correct += 1
    #
    #     model_accuracy_train = num_correct / len(train_images)
    #     print(f"Param num: {model.count_params()}")
    #     print(f"K: {k}")
    #     print(f"Accuracy of model on train: {100 * model_accuracy_train}%")
    #
    #     num_correct = 0
    #     predictions = model.predict(test_images)
    #     for i in range(len(predictions)):
    #         if np.argmax(predictions[i]) == test_labels[i]:
    #             num_correct += 1
    #
    #     model_accuracy_test = num_correct/len(test_images)
    #
    #     print(f"Accuracy of model on test: {100 * model_accuracy_test}%")
    #     print()
    #
    #     param_nums.append(model.count_params())
    #
    #     accuracy_difs.append(model_accuracy_train - model_accuracy_test)

    # add_to_pickle(pickle_file="cifar_accuracy_difs.pkl", new_data=accuracy_difs)

    # accuracy_difs_total = load_pickle_dict("cifar_accuracy_difs.pkl")
    scale_factor = 10_000_000

    # Create scaled DataFrame
    df = pd.DataFrame({
        'x': np.array(param_nums) / scale_factor,
        'y': accuracy_difs
    }, columns=['x', 'y'])

    # Perform linear regression
    result = linregress(df['x'], df['y'])

    # Unpack regression results
    slope = result.slope
    intercept = result.intercept
    r_value = result.rvalue
    r_squared = r_value ** 2
    stderr = result.stderr
    t_stat = slope / stderr
    p_value = result.pvalue
    n = len(df)
    df_error = n - 2

    # 95% CI for slope
    alpha = 0.05
    t_crit = t.ppf(1 - alpha / 2, df_error)
    ci_low = slope - t_crit * stderr
    ci_high = slope + t_crit * stderr

    # Print Minitab-style regression output
    print("Regression Analysis (x scaled by 10 million parameters)")
    print(f"{'Predictor':<12}{'Coef':>10}{'SE Coef':>12}{'T':>10}{'P':>10}")
    print(f"{'Constant':<12}{intercept:10.4f}{'':>12}")
    print(f"{'x':<12}{slope:10.4f}{stderr:12.4f}{t_stat:10.4f}{p_value:10.4f}")
    print()
    print(f"S = {np.sqrt(result.stderr ** 2 * df_error / (n - 1)):.4f}   R-sq = {r_squared * 100:.2f}%")
    print(f"95% CI for slope: ({ci_low:.4f}, {ci_high:.4f})")

    # Plot
    ax = df.plot(x='x', y='y', style='.', marker='o', label='Data Points')
    x_vals = np.linspace(df['x'].min(), df['x'].max(), 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='red', label='LSRL')

    plt.xlabel('Parameters (10 millions)')
    plt.ylabel('Difference in Train and Test Accuracy')
    plt.title('Layer Size vs Difference in Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

