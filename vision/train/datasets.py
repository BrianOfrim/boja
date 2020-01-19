import os
from typing import List

from PIL import Image
import torch

from .pascal_voc_parser import read_content


INVALID_ANNOTATION_FILE_IDENTIFIER = "invalid"


class BojaDataSet(object):
    def __init__(
        self,
        image_dir_path: str,
        annotation_dir_path: str,
        manifest_file_path: str,
        transforms,
        labels: List[str],
    ):
        self.image_dir_path = image_dir_path
        self.annotation_dir_path = annotation_dir_path

        self.transforms = transforms

        self.labels = labels
        manifest_items = [
            item.strip() for item in open(manifest_file_path).read().splitlines()
        ]
        # Filter out Invalid images
        manifest_items = [
            item
            for item in manifest_items
            if item.split(",")[1].lower() != INVALID_ANNOTATION_FILE_IDENTIFIER
        ]

        self.images = [
            os.path.join(self.image_dir_path, item.split(",")[0])
            for item in manifest_items
        ]
        self.annotations = [
            os.path.join(self.annotation_dir_path, item.split(",")[1])
            for item in manifest_items
        ]

    def __getitem__(self, idx):
        # load images ad masks
        img = Image.open(self.images[idx]).convert("RGB")
        _, annotation_boxes = read_content(self.annotations[idx])

        num_objs = len(annotation_boxes)
        boxes = [[b.xmin, b.ymin, b.xmax, b.ymax] for b in annotation_boxes]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class

        labels = [self.labels.index(b.label) for b in annotation_boxes]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])  # pylint: disable=not-callable

        area = [b.get_area() for b in annotation_boxes]
        area = torch.as_tensor(area, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)
