from scratch_model import *
import numpy as np
from coco_bucket_loader import COCOBucketBatchLoader


def get_data(batches):
    data = []
    labels = []
    for batch in batches:
        imgs = batch["images"]
        size = batch["resized_size"]
        boxes = batch["boxes"]
        classes = batch["labels"]
        out_h = int(size[0] // 32)
        out_w = int(size[1] // 32)

        data.append(np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32))

        # Batch, spatial_h, spatial_w, anchors, output
        # Output: (o, x, y, h, w, *classes)
        label = np.zeros((len(imgs), out_h, out_w, 3, 85), dtype=np.float32)

        for b in range(len(boxes)):
            img_boxes = boxes[b]
            img_classes = classes[b]
            for i in range(len(img_boxes)):
                img_box = img_boxes[i]
                img_class = img_classes[i]
                x, y, w, h = img_box
                spatial_x = int(x // 32)
                spatial_y = int(y // 32)

                rel_x = x - ((spatial_x * 32) + 16)
                rel_y = y - ((spatial_y * 32) + 16)

                # Anchors: 0: Wide 1: Square 2: Tall
                anchor = 1

                aspect_ratio = h / w
                if aspect_ratio <= 0.75:
                    anchor = 0
                elif aspect_ratio >= 1.333:
                    anchor = 2

                out = label[b][spatial_y][spatial_x][anchor]

                out[0] = 1.0
                out[1] = rel_x
                out[2] = rel_y
                out[3] = h
                out[4] = w
                out[img_class + 5] = 1.0

        labels.append(label)

    return data, labels




def main():
    batch_size = 4

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
            layers.Convolution(16, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(32, (3, 3), activation_function=model_functions.relu, padding=1, stride=2),

            layers.Convolution(32, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(64, (3, 3), activation_function=model_functions.relu, padding=1, stride=2),

            layers.Convolution(64, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(128, (3, 3), activation_function=model_functions.relu, padding=1, stride=2),
            layers.Convolution(128, (3, 3), activation_function=model_functions.relu, padding=1),

            layers.Convolution(128, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(256, (3, 3), activation_function=model_functions.relu, padding=1, stride=2),
            layers.Convolution(256, (3, 3), activation_function=model_functions.relu, padding=1),

            layers.Convolution(256, (3, 3), activation_function=model_functions.relu, padding=1),
            layers.Convolution(512, (3, 3), activation_function=model_functions.relu, padding=1, stride=2),
            layers.Convolution(512, (3, 3), activation_function=model_functions.relu, padding=1),

            layers.Convolution(3 * (80 + 5), (1, 1)),

            # Reshape and transpose to output shape: (in_h / 32, in_2 / 32, 3, 85)
            layers.ReshapeOnAxis((3, 85), axis=0),
            layers.Transpose((2, 3, 0, 1)),
            # Apply activation function
            layers.ActivationFunction(model_functions.yolo_activation)
        ],
        optimizer=optimizers.AdamW,
        optimizer_args=(0.9, 0.999, 0.0),
        dtype=np.float32,
    )

    batch_index = 0
    step = 50
    end = step
    while loader.has_batch(batch_index):
        batches = []
        while loader.has_batch(batch_index) and batch_index <= end:
            print(batch_index)
            batches.append(loader.get_batch(batch_index))
            batch_index += 1
        data, labels = get_data(batches)
        ai_model.fit(
            data,
            labels,
            1,
            0.000001,
            batch_size=batch_size,
            is_pre_batched=True
        )
        end += step

    ai_model.save("Models/coco_yolo_v1")


if __name__ == "__main__":
    main()
