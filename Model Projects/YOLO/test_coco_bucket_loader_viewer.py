import argparse
import os
import sys
from typing import Dict, List, Tuple
import cv2
from scratch_model import *
import numpy as np
import cupy as cp

from coco_bucket_loader import COCOBucketBatchLoader


def build_class_mappings(loader: COCOBucketBatchLoader) -> Tuple[Dict[int, int], Dict[int, str], List[str]]:
    if not loader.remap_classes:
        raise ValueError("This viewer expects remap_classes=True so labels are 0..79.")

    index_to_cat_id = loader.index_to_cat_id
    cat_id_to_name = {
        cat["id"]: cat["name"]
        for cat in loader.coco.loadCats(loader.coco.getCatIds())
    }
    class_names = [cat_id_to_name[index_to_cat_id[i]] for i in range(len(index_to_cat_id))]
    return index_to_cat_id, cat_id_to_name, class_names


def draw_sample(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    batch_index: int,
    image_index: int,
    batch_size: int,
    file_name: str,
    image_id: int,
) -> np.ndarray:
    vis = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    for box, label in zip(boxes, labels):
        cx, cy, w, h = box
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(vis.shape[1] - 1, x2)
        y2 = min(vis.shape[0] - 1, y2)

        class_name = class_names[int(label)] if 0 <= int(label) < len(class_names) else str(label)
        text = f"{class_name} ({int(label)})"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(text_h + 4, y1)
        cv2.rectangle(
            vis,
            (x1, text_y - text_h - baseline - 4),
            (x1 + text_w + 4, text_y + 2),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            vis,
            text,
            (x1 + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    info_lines = [
        f"Batch {batch_index + 1} | Image {image_index + 1}/{batch_size}",
        # f"image_id={image_id} | {file_name}",
        # f"shape={image.shape[1]}x{image.shape[0]} | boxes={len(boxes)}",
        # "Keys: d/right next image, a/left prev image, s/down next batch, w/up prev batch, q/esc quit",
    ]

    y = 24
    for line in info_lines:
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        y += 26

    return vis


def get_boxes(image, yolo_model, object_threshold=0.5):
    image = np.transpose(image, (2, 0, 1))
    # Shape (spatial, spatial, anchors, 85)
    prediction = yolo_model.predict(cp.array([image]))[0]

    image_h, image_w = image.shape[1:]
    x, y, a, d = prediction.shape

    reshaped_pred = prediction.reshape(-1, d)
    object_mask = reshaped_pred[:, 0] >= object_threshold
    object_predictions = reshaped_pred[object_mask]

    # shape (x, y, 2)
    xy = cp.stack(cp.indices((x, y)), axis=-1)
    xy = cp.repeat(xy[:, :, None, :], a, axis=2)
    xy_flat = xy.reshape(-1, 2)

    # x, y locations of all objects
    locations = xy_flat[object_mask]

    center_x = 32 * (object_predictions[:, 1] + locations[:,  0])
    center_y = 32 * (object_predictions[:, 2] + locations[:, 1])
    widths = image_w * object_predictions[:, 4]
    heights = image_h * object_predictions[:, 3]

    # # (n, 4) where 4 is cx, cy, w, h
    boxes = cp.zeros((len(object_predictions), 4))
    boxes[:, 0] = center_x
    boxes[:, 1] = center_y
    boxes[:, 2] = widths
    boxes[:, 3] = heights

    # boxes[:, 0] = object_predictions[:, 0]
    # boxes[:, 0] = np.clip(center_x - widths / 2, 0, image_w)
    # boxes[:, 1] = np.clip(center_y - heights / 2, 0, image_h)
    # boxes[:, 2] = np.clip(center_x + widths / 2, 0, image_w)
    # boxes[:, 3] = np.clip(center_y + heights / 2, 0, image_h)

    labels = cp.argmax(object_predictions[:, 5:], axis=-1)

    return cp.asnumpy(boxes), cp.asnumpy(labels)


def main() -> None:
    loader = COCOBucketBatchLoader(
        image_dir="coco2017/train2017",
        annotation_file="coco2017/annotations/instances_train2017.json",
        start_index=0,
        multiple=32,
        batch_size=64,
        seed=42,
    )

    ai_model = Model.load("Models/coco_yolo_v2_e1")

    if loader.num_batches() == 0:
        print("No batches found.")
        return

    _, _, class_names = build_class_mappings(loader)

    batch_index = 0
    image_index = 0

    cv2.namedWindow("COCO", cv2.WINDOW_NORMAL)

    update_boxes = True
    boxes = None
    labels = None

    while True:
        batch = loader.get_batch(batch_index)
        if update_boxes:
            boxes, labels = get_boxes(batch["images"][image_index], ai_model, object_threshold=0.99)
            update_boxes = False

        batch_len = len(batch["images"])
        image_index = max(0, min(image_index, batch_len - 1))

        vis = draw_sample(
            image=np.array(batch["images"][image_index]),
            boxes=boxes,
            labels=labels,
            class_names=class_names,
            batch_index=batch_index,
            image_index=image_index,
            batch_size=batch_len,
            file_name=batch["file_names"][image_index],
            image_id=batch["image_ids"][image_index],
        )

        cv2.imshow("COCO", vis)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("d"), 83):
            if image_index < batch_len - 1:
                image_index += 1
            elif batch_index < loader.num_batches() - 1:
                batch_index += 1
                image_index = 0
            update_boxes = True
        elif key in (ord("a"), 81):
            if image_index > 0:
                image_index -= 1
            elif batch_index > 0:
                batch_index -= 1
                prev_batch = loader.get_batch(batch_index)
                image_index = len(prev_batch["images"]) - 1
            update_boxes = True
        elif key in (ord("s"), 84):
            if batch_index < loader.num_batches() - 1:
                batch_index += 1
                image_index = 0
            update_boxes = True
        elif key in (ord("w"), 82):
            if batch_index > 0:
                batch_index -= 1
                image_index = 0
            update_boxes = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
