import time
import math
from coco_bucket_loader import COCOBucketBatchLoader
from scratch_model import *
import numpy as np
import cupy as cp


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

        data.append(cp.array(np.transpose(imgs, (0, 3, 1, 2)), dtype=np.float32) / 255)

        # Batch, spatial_h, spatial_w, anchors, output
        # Output: (o, x, y, h, w, *classes)
        label = cp.zeros((len(imgs), out_h, out_w, 3, 85), dtype=cp.float32)

        for b in range(len(boxes)):
            img_boxes = boxes[b]
            img_classes = classes[b]
            for i in range(len(img_boxes)):
                img_box = img_boxes[i]
                img_class = img_classes[i]
                x, y, w, h = img_box
                cell_x = x / 32
                cell_y = y / 32

                spatial_x = int(cell_x)
                spatial_y = int(cell_y)

                rel_x = cell_x - spatial_x
                rel_y = cell_y - spatial_y

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
                out[3] = h / size[0]
                out[4] = w / size[1]
                out[img_class + 5] = 1.0

        labels.append(label)

    return data, labels


def yolo_proxy_accuracy(
    output,
    expected,
    obj_threshold=0.5,
    iou_threshold=0.5,
    return_parts=False
):
    eps = 1e-7

    obj_pred = output[..., 0]
    obj_true = expected[..., 0]

    pos_mask = (obj_true == 1)
    neg_mask = (obj_true == 0)

    obj_pred_binary = (obj_pred >= obj_threshold)

    # per-image objectness accuracy
    obj_acc = cp.mean(obj_pred_binary == pos_mask, axis=(1, 2, 3))  # (b,)

    pred_class = cp.argmax(output[..., 5:], axis=-1)
    true_class = cp.argmax(expected[..., 5:], axis=-1)

    batch_size = output.shape[0]
    class_acc = cp.ones(batch_size, dtype=output.dtype)
    box_acc = cp.ones(batch_size, dtype=output.dtype)
    mean_iou = cp.ones(batch_size, dtype=output.dtype)
    num_pos = cp.sum(pos_mask, axis=(1, 2, 3))  # (b,)

    for b in range(batch_size):
        if num_pos[b] > 0:
            cur_pos = pos_mask[b]

            class_acc[b] = cp.mean(pred_class[b][cur_pos] == true_class[b][cur_pos])

            pred_boxes = output[b, ..., 1:5][cur_pos]
            true_boxes = expected[b, ..., 1:5][cur_pos]

            pred_x = pred_boxes[:, 0]
            pred_y = pred_boxes[:, 1]
            pred_h = pred_boxes[:, 2]
            pred_w = pred_boxes[:, 3]

            true_x = true_boxes[:, 0]
            true_y = true_boxes[:, 1]
            true_h = true_boxes[:, 2]
            true_w = true_boxes[:, 3]

            pred_x1 = pred_x - pred_w / 2
            pred_y1 = pred_y - pred_h / 2
            pred_x2 = pred_x + pred_w / 2
            pred_y2 = pred_y + pred_h / 2

            true_x1 = true_x - true_w / 2
            true_y1 = true_y - true_h / 2
            true_x2 = true_x + true_w / 2
            true_y2 = true_y + true_h / 2

            inter_x1 = cp.maximum(pred_x1, true_x1)
            inter_y1 = cp.maximum(pred_y1, true_y1)
            inter_x2 = cp.minimum(pred_x2, true_x2)
            inter_y2 = cp.minimum(pred_y2, true_y2)

            inter_w = cp.maximum(0.0, inter_x2 - inter_x1)
            inter_h = cp.maximum(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            pred_area = cp.maximum(pred_w, 0.0) * cp.maximum(pred_h, 0.0)
            true_area = cp.maximum(true_w, 0.0) * cp.maximum(true_h, 0.0)

            union_area = pred_area + true_area - inter_area
            iou = inter_area / cp.maximum(union_area, eps)

            mean_iou[b] = cp.mean(iou)
            box_acc[b] = cp.mean(iou >= iou_threshold)

    combined = (0.5 * obj_acc) + (0.25 * class_acc) + (0.25 * box_acc)

    if return_parts:
        return {
            "combined": float(cp.sum(combined)),
            "objectness_accuracy": float(cp.sum(obj_acc)),
            "class_accuracy": float(cp.sum(class_acc)),
            "box_accuracy": float(cp.sum(box_acc)),
            "mean_iou": float(cp.sum(mean_iou)),
            "num_positive_anchors": int(cp.sum(num_pos)),
        }

    return float(cp.sum(combined))


def lr_percent_cosine_step(step, total_steps=29658*100, warmup_steps=3000, min_percent=0.05):
    if total_steps <= 1:
        return 1.0

    step = max(0, min(int(step), total_steps - 1))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps  # 0 -> (almost) 1

    denom = total_steps - warmup_steps
    if denom <= 1:
        return 1.0

    t = (step - warmup_steps) / denom
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_percent + (1.0 - min_percent) * cosine


def conv(channels, kernel_size, stride):
    return layers.Convolution(
        channels,
        (kernel_size, kernel_size),
        activation_function=model_functions.relu,
        padding=kernel_size // 2,
        stride=stride
    )


def csp_block(channels, num_inner_convs=2, split_ratio=0.5):
    split_channels = int(channels * split_ratio)
    other_channels = channels - split_channels

    before_layers = (
        [conv(split_channels, 1, 1)] +
        [conv(split_channels, 3, 1) for _ in range(num_inner_convs)]
    )

    after_layers = [conv(other_channels, 1, 1)]

    return [
        conv(channels, 1, 1),
        layers.Split(
            before_layers=before_layers,
            after_layers=after_layers,
            axis=0,
            partition=split_channels
        ),
        conv(channels, 1, 1),
    ]


def main():
    epochs = 100
    batch_size = 4

    # ai_model = Model(
    #     model_functions.yolo_loss,
    #     (3, -1, -1),
    #     [
    #         conv(32, 3, 1),
    #
    #         conv(64, 3, 2), *csp_block(64, num_inner_convs=1),
    #         conv(128, 3, 2), *csp_block(128, num_inner_convs=2),
    #         conv(256, 3, 2), *csp_block(256, num_inner_convs=3),
    #         conv(512, 3, 2), *csp_block(512, num_inner_convs=3),
    #         conv(1024, 3, 2), *csp_block(1024, num_inner_convs=2),
    #
    #         conv(512, 3, 1),
    #
    #         layers.Convolution(3 * (80 + 5), (1, 1)),
    #
    #         # Reshape and transpose to output shape: (in_h / 32, in_2 / 32, 3, 85)
    #         layers.ReshapeOnAxis((3, 85), axis=0),
    #         layers.Transpose((2, 3, 0, 1)),
    #         # Apply activation function
    #         layers.ActivationFunction(model_functions.yolo_activation)
    #     ],
    #     optimizer=optimizers.Adam,
    #     optimizer_args=(0.9, 0.999),
    #     dtype=cp.float32,
    # )
    ai_model = Model.load("Models/coco_yolo_v2_e1_21760")

    print(f"Param num: {ai_model.get_param_num()}")

    model_name = "coco_yolo_v2"
    data_path = f"Data/{model_name}"

    current_epoch = 1
    total_batches = 51418
    while current_epoch < epochs:
        loader = COCOBucketBatchLoader(
            image_dir="coco2017/train2017",
            annotation_file="coco2017/annotations/instances_train2017.json",
            start_index=0,
            multiple=32,
            batch_size=batch_size,
            seed=42 + current_epoch,
        )

        print(f"Num batches: {loader.get_batch_num()}")

        if current_epoch == 1:
            batch_index = 51418 - 29658
        else:
            batch_index = 0

        step = 64
        end = batch_index + step
        cur_save_step = 0
        num_to_save = 170
        while loader.has_batch(batch_index):
            batches = []
            while loader.has_batch(batch_index) and batch_index < end:
                batches.append(loader.get_batch(batch_index))
                batch_index += 1
            data, labels = get_data(batches)

            t0 = time.perf_counter()

            ai_model.fit(
                data,
                labels,
                1,
                0.00002,
                batch_size=batch_size,
                start_step=total_batches,
                is_pre_batched=True,
                accuracy_function=yolo_proxy_accuracy,
                data_save_file=data_path,
                learning_rate_function=lr_percent_cosine_step,
                steps_to_update_weights=4,
            )

            total_batches += len(batches)

            print(f"Took {(time.perf_counter() - t0):.2f}s")

            end += step

            cur_save_step += 1

            if cur_save_step >= num_to_save:
                ai_model.save(f"Models/{model_name}_e{current_epoch}_{batch_index}")
                cur_save_step = 0
                print(f"Saved model after training on {total_batches} batches.")

        print(f"Finished epoch {current_epoch}")
        ai_model.save(f"Models/{model_name}_e{current_epoch}")
        current_epoch += 1


if __name__ == "__main__":
    main()
