from scratch_model import *
import numpy as np
from coco_bucket_loader import COCOBucketBatchLoader


def main():
    batch_size = 64

    loader = COCOBucketBatchLoader(
        image_dir="coco2017/train2017",
        annotation_file="coco2017/annotations/instances_train2017.json",
        start_index=0,
        multiple=32,
        batch_size=batch_size,
        seed=42,
    )

    ai_model = Model(
        model_functions.yolo_loss,
        (3, -1, -1),
        [
            layers.Convolution(32, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(64, (3, 3), activation_function=model_functions.relu, padding=1, stride=2),

            layers.Convolution(64, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(128, (3, 3), activation_function=model_functions.relu, padding=1, stride=2),

            layers.Convolution(128, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(256, (3, 3), activation_function=model_functions.relu, padding=1, stride=2),
            layers.Convolution(256, (3, 3), activation_function=model_functions.relu, padding=1),

            layers.Convolution(256, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(512, (3, 3), activation_function=model_functions.relu, padding=1, stride=2),
            layers.Convolution(512, (3, 3), activation_function=model_functions.relu, padding=1),

            layers.Convolution(512, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(1024, (3, 3), activation_function=model_functions.relu, padding=1, stride=2),
            layers.Convolution(1024, (3, 3), activation_function=model_functions.relu, padding=1),

            layers.Convolution(3 * (80 + 5), (1, 1)),

            # Reshape and transpose to output shape: (30, 30, 3, 85)
            layers.ReshapeOnAxis((3, 85), axis=0),
            layers.Transpose((2, 3, 0, 1)),
            # Apply activation function
            layers.ActivationFunction(model_functions.yolo_activation)
        ],
        optimizer=optimizers.AdamW,
        optimizer_args=(0.9, 0.999, 0.0),
    )


if __name__ == "__main__":
    main()
