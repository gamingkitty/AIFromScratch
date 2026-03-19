import argparse
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

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


def main() -> None:
    loader = COCOBucketBatchLoader(
        image_dir="coco2017/train2017",
        annotation_file="coco2017/annotations/instances_train2017.json",
        start_index=0,
        multiple=32,
        batch_size=64,
        seed=42,
    )

    if loader.num_batches() == 0:
        print("No batches found.")
        return

    _, _, class_names = build_class_mappings(loader)

    batch_index = 0
    image_index = 0

    cv2.namedWindow("COCO", cv2.WINDOW_NORMAL)

    while True:
        batch = loader.get_batch(batch_index)
        batch_len = len(batch["images"])
        image_index = max(0, min(image_index, batch_len - 1))

        vis = draw_sample(
            image=batch["images"][image_index],
            boxes=batch["boxes"][image_index],
            labels=batch["labels"][image_index],
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
        elif key in (ord("d"), 83):  # d or right arrow (common OpenCV code)
            if image_index < batch_len - 1:
                image_index += 1
            elif batch_index < loader.num_batches() - 1:
                batch_index += 1
                image_index = 0
        elif key in (ord("a"), 81):  # a or left arrow
            if image_index > 0:
                image_index -= 1
            elif batch_index > 0:
                batch_index -= 1
                prev_batch = loader.get_batch(batch_index)
                image_index = len(prev_batch["images"]) - 1
        elif key in (ord("s"), 84):  # s or down arrow
            if batch_index < loader.num_batches() - 1:
                batch_index += 1
                image_index = 0
        elif key in (ord("w"), 82):  # w or up arrow
            if batch_index > 0:
                batch_index -= 1
                image_index = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
