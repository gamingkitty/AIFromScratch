
import os
import math
import random
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO


class COCOBucketBatchLoader:
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        start_index: int = 0,
        multiple: int = 32,
        batch_size: int = 8,
        seed: int = 0,
        interpolation: int = cv2.INTER_LINEAR,
        remap_classes: bool = True,
        skip_crowd: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.multiple = multiple
        self.batch_size = batch_size
        self.seed = seed
        self.interpolation = interpolation
        self.remap_classes = remap_classes
        self.skip_crowd = skip_crowd

        all_image_ids = self.coco.getImgIds()
        if start_index < 0 or start_index > len(all_image_ids):
            raise ValueError(f"start_index must be between 0 and {len(all_image_ids)}")

        self.image_ids = list(all_image_ids[start_index:])

        self.rng = random.Random(seed)
        self.rng.shuffle(self.image_ids)

        if remap_classes:
            cat_ids = sorted(self.coco.getCatIds())
            self.cat_id_to_index = {cat_id: i for i, cat_id in enumerate(cat_ids)}
            self.index_to_cat_id = {i: cat_id for cat_id, i in self.cat_id_to_index.items()}
        else:
            self.cat_id_to_index = None
            self.index_to_cat_id = None

        self.size_buckets = self._build_size_buckets()
        self.batches = self._build_batches()

    @staticmethod
    def round_up_to_multiple(value: int, multiple: int) -> int:
        return int(math.ceil(value / multiple) * multiple)

    def _resized_shape_from_info(self, width: int, height: int) -> Tuple[int, int]:
        new_h = self.round_up_to_multiple(height, self.multiple)
        new_w = self.round_up_to_multiple(width, self.multiple)
        return new_h, new_w

    def _build_size_buckets(self) -> Dict[Tuple[int, int], List[int]]:
        buckets: Dict[Tuple[int, int], List[int]] = {}

        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            resized_shape = self._resized_shape_from_info(img_info["width"], img_info["height"])
            buckets.setdefault(resized_shape, []).append(img_id)

        # Shuffle each bucket deterministically
        for shape in buckets:
            self.rng.shuffle(buckets[shape])

        return buckets

    def _build_batches(self) -> List[List[int]]:
        batches: List[List[int]] = []

        for shape, bucket_img_ids in self.size_buckets.items():
            for i in range(0, len(bucket_img_ids), self.batch_size):
                batches.append(bucket_img_ids[i:i + self.batch_size])

        self.rng.shuffle(batches)
        return batches

    def __len__(self) -> int:
        return len(self.image_ids)

    def num_batches(self) -> int:
        return len(self.batches)

    def _load_annotations(self, img_id: int) -> Tuple[np.ndarray, np.ndarray]:
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            if self.skip_crowd and ann.get("iscrowd", 0) == 1:
                continue

            x, y, w, h = ann["bbox"]

            # Convert COCO [x_min, y_min, w, h] -> [center_x, center_y, w, h]
            center_x = x + w / 2.0
            center_y = y + h / 2.0

            boxes.append([center_x, center_y, w, h])

            category_id = ann["category_id"]
            if self.remap_classes:
                labels.append(self.cat_id_to_index[category_id])
            else:
                labels.append(category_id)

        if boxes:
            boxes_array = np.array(boxes, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int64)
        else:
            boxes_array = np.zeros((0, 4), dtype=np.float32)
            labels_array = np.zeros((0,), dtype=np.int64)

        return boxes_array, labels_array

    def _resize_image_and_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        new_h: int,
        new_w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        old_h, old_w = image.shape[:2]

        if old_h == new_h and old_w == new_w:
            return image, boxes

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)

        if boxes.shape[0] == 0:
            return resized_image, boxes

        scale_x = new_w / old_w
        scale_y = new_h / old_h

        resized_boxes = boxes.copy()
        resized_boxes[:, 0] *= scale_x  # center_x
        resized_boxes[:, 1] *= scale_y  # center_y
        resized_boxes[:, 2] *= scale_x  # width
        resized_boxes[:, 3] *= scale_y  # height

        return resized_image, resized_boxes

    def get_sample(self, img_id: int) -> dict:
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels = self._load_annotations(img_id)

        new_h, new_w = self._resized_shape_from_info(img_info["width"], img_info["height"])
        image, boxes = self._resize_image_and_boxes(image, boxes, new_h, new_w)

        return {
            "image": image,
            "boxes": boxes,          # shape (N, 4), format [center_x, center_y, width, height]
            "labels": labels,        # shape (N,)
            "image_id": img_id,
            "file_name": img_info["file_name"],
            "original_size": (img_info["height"], img_info["width"]),
            "resized_size": (new_h, new_w),
        }

    def iter_samples(self) -> Iterator[dict]:
        for img_id in self.image_ids:
            yield self.get_sample(img_id)

    def get_batch(self, batch_index: int) -> dict:
        img_ids = self.batches[batch_index]
        samples = [self.get_sample(img_id) for img_id in img_ids]

        images = np.stack([sample["image"] for sample in samples], axis=0)
        boxes = [sample["boxes"] for sample in samples]
        labels = [sample["labels"] for sample in samples]

        return {
            "images": images,                # shape (B, H, W, C)
            "boxes": boxes,                  # list of length B, each element shape (N_i, 4)
            "labels": labels,                # list of length B, each element shape (N_i,)
            "image_ids": [sample["image_id"] for sample in samples],
            "file_names": [sample["file_name"] for sample in samples],
            "original_sizes": [sample["original_size"] for sample in samples],
            "resized_size": samples[0]["resized_size"],  # all same within batch
        }

    def iter_batches(self) -> Iterator[dict]:
        for batch_index in range(len(self.batches)):
            yield self.get_batch(batch_index)

