from scratch_model import *


def main():
    ai_model = Model(
        model_functions.yolo_loss,
        (3, 480, 480),
        [
            layers.Convolution(32, (7, 7), activation_function=model_functions.relu, padding=3, stride=2),
            # Shape: 240, 240
            layers.Convolution(64, (5, 5), activation_function=model_functions.relu, padding=2),
            layers.Convolution(64, (5, 5), activation_function=model_functions.relu, padding=2),
            layers.MaxPooling((2, 2), 2),
            # Shape: 120, 120
            layers.Convolution(128, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(128, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.MaxPooling((2, 2), 2),
            # Shape: 60, 60
            layers.Convolution(256, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(256, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.MaxPooling((2, 2), 2),
            # Shape: 512, 30, 30
            layers.Convolution(512, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(512, (3, 3), activation_function=model_functions.relu, padding=1),

            # Shape: 3 * (80 + 5), 30, 30, 30x30 spatial positions and 80 classes, 3 anchors
            layers.Convolution(3 * 85, (1, 1)),
            # Reshape and transpose to output shape: (30, 30, 3, 85)
            layers.Reshape((3, 85, 30, 30)),
            layers.Transpose((2, 3, 0, 1)),
            # Apply activation function
            layers.ActivationFunction(model_functions.yolo_activation)
        ],
        optimizer=optimizers.AdamW,
        optimizer_args=(0.9, 0.999, 0.0),
    )


if __name__ == "__main__":
    main()
