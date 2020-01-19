import os
import re
import time
from typing import List

from absl import app, flags
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from .pascal_voc_parser import read_content
from .transforms import ToTensor, RandomHorizontalFlip, Compose

matplotlib.use("TKAgg")

IMAGE_DIR_NAME = "images"
ANNOTATION_DIR_NAME = "annotations"
MANIFEST_DIR_NAME = "manifests"
MODEL_STATE_ROOT_DIR = "modelState"

IMAGE_FILE_TYPE = "jpg"
ANNOTATION_FILE_TYPE = "xml"
MANIFEST_FILE_TYPE = "txt"
MODEL_STATE_FILE_NAME = "modelState.pt"

INVALID_ANNOTATION_FILE_IDENTIFIER = "invalid"

flags.DEFINE_string(
    "local_data_dir",
    os.path.join(os.path.expanduser("~"), "boja", "data"),
    "Local data directory.",
)

flags.DEFINE_string(
    "label_file_path",
    os.path.join(os.path.expanduser("~"), "boja", "data", "labels.txt"),
    "Path to the file containing the category labels.",
)

flags.DEFINE_string(
    "manifest_path", None, "The manifest file to load images from. Default is newest."
)

flags.DEFINE_string("model_path", None, "The model to load. Default is newest.")

flags.DEFINE_float(
    "threshold", 0.5, "The threshold above which to display predicted bounding boxes"
)


class ODDataSet(object):
    def __init__(self, data_root, transforms, labels, manifest_file_path: str):

        self.data_root = data_root
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
            os.path.join(self.data_root, IMAGE_DIR_NAME, item.split(",")[0])
            for item in manifest_items
        ]
        self.annotations = [
            os.path.join(self.data_root, ANNOTATION_DIR_NAME, item.split(",")[1])
            for item in manifest_items
        ]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.data_root, IMAGE_DIR_NAME, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        annotation_path = os.path.join(
            self.data_root, ANNOTATION_DIR_NAME, self.annotations[idx]
        )
        _, annotation_boxes = read_content(annotation_path)

        num_objs = len(annotation_boxes)
        boxes = [[b.xmin, b.ymin, b.xmax, b.ymax] for b in annotation_boxes]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class

        labels = [self.labels.index(b.label) for b in annotation_boxes]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])

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


def get_model_instance_detection(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_model_instance_detection_mobilenet(num_classes):

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    return model


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def get_files_from_dir(dir_path: str, file_type: str = None) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    file_paths = [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]
    if file_type is not None:
        file_paths = [f for f in file_paths if f.lower().endswith(file_type.lower())]
    return file_paths


def int_string_sort(manifest_file) -> int:
    match = re.match("[0-9]+", manifest_file)
    if not match:
        return 0
    return int(match[0])


def get_newest_manifest_path(manifest_dir_path: str) -> str:
    manifest_files = get_files_from_dir(manifest_dir_path)
    manifest_files = [
        f for f in manifest_files if f.lower().endswith(MANIFEST_FILE_TYPE)
    ]
    if len(manifest_files) == 0:
        return None
    newest_manifest_file = sorted(manifest_files, key=int_string_sort, reverse=True)[0]
    return os.path.join(
        flags.FLAGS.local_data_dir, MANIFEST_DIR_NAME, newest_manifest_file
    )


def get_newest_saved_model_path(model_dir_path: str) -> str:
    _, model_storage_dirs, _ = next(os.walk(model_dir_path))
    if len(model_storage_dirs) == 0:
        return None
    model_storage_dirs = sorted(model_storage_dirs, key=int_string_sort, reverse=True)
    model_file_path = os.path.join(
        model_dir_path, model_storage_dirs[0], MODEL_STATE_FILE_NAME
    )
    if not os.path.isfile(model_file_path):
        return None
    return model_file_path


def draw_bboxes(
    ax, bboxes, label_indices, label_names, label_colors, label_scores=None
):
    for box_index, (box, label_index) in enumerate(zip(bboxes, label_indices)):
        height = box[3] - box[1]
        width = box[2] - box[0]
        lower_left = (box[0], box[1])
        rect = patches.Rectangle(
            lower_left,
            width,
            height,
            linewidth=2,
            edgecolor=label_colors[label_index],
            facecolor="none",
        )
        ax.add_patch(rect)
        label_string = ""
        if label_scores is None:
            label_string = label_names[label_index]
        else:
            label_string = "%s [%.2f]" % (
                label_names[label_index],
                label_scores[box_index],
            )
        ax.text(
            box[0],
            box[1] - 10,
            label_string,
            bbox=dict(
                facecolor=label_colors[label_index],
                alpha=0.5,
                pad=1,
                edgecolor=label_colors[label_index],
            ),
            fontsize=10,
            color="white",
        )


def main(unused_argv):

    if not os.path.isfile(flags.FLAGS.label_file_path):
        print("Invalid category labels path.")
        return

    labels = [
        label.strip() for label in open(flags.FLAGS.label_file_path).read().splitlines()
    ]

    if len(labels) == 0:
        print("No labels are present in %s" % flags.FLAGS.label_file_path)
        return

    # Add the background as the first class
    labels.insert(0, "background")

    print("Labels found:")
    print(labels)

    manifest_file_path = (
        flags.FLAGS.manifest_path
        if flags.FLAGS.manifest_path is not None
        else get_newest_manifest_path(
            os.path.join(flags.FLAGS.local_data_dir, MANIFEST_DIR_NAME)
        )
    )

    if manifest_file_path is None:
        print("No manifest file found")
        return

    saved_model_file_path = (
        flags.FLAGS.model_path
        if flags.FLAGS.model_path is not None
        else get_newest_saved_model_path(
            os.path.join(flags.FLAGS.local_data_dir, MODEL_STATE_ROOT_DIR)
        )
    )

    if saved_model_file_path is None:
        print("No saved model state found")
        return

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Using device: ", device)

    # Add one class for the background
    num_classes = len(labels)
    # use our dataset and defined transformations
    dataset = ODDataSet(
        flags.FLAGS.local_data_dir,
        get_transform(train=False),
        labels,
        manifest_file_path,
    )

    # get the model using our helper function
    model = get_model_instance_detection_mobilenet(num_classes)

    print("Loading model state from: %s" % saved_model_file_path)

    model.load_state_dict(torch.load(saved_model_file_path))

    print("Model state loaded")

    model.eval()

    # move model to the right device
    model.to(device)

    # create plots
    fig, (ground_truth_ax, inference_ax) = plt.subplots(1, 2)

    label_colors = plt.get_cmap("hsv")(np.linspace(0, 0.9, len(labels)))

    with torch.no_grad():
        for data in dataset:
            image, target = data
            model_time = time.time()
            outputs = model([image])
            outputs = [
                {k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs
            ]
            model_time = time.time() - model_time
            print("Inference time = ", model_time)

            display_image_base = F.to_pil_image(image)

            ground_truth_ax.clear()
            inference_ax.clear()

            ground_truth_ax.set_title("Ground Truth")
            inference_ax.set_title("Inference")

            ground_truth_ax.imshow(display_image_base)
            inference_ax.imshow(display_image_base)

            draw_bboxes(
                ground_truth_ax, target["boxes"], target["labels"], labels, label_colors
            )

            # filter out the background labels and scores bellow threshold
            filtered_output = [
                (
                    outputs[0]["boxes"][j],
                    outputs[0]["labels"][j],
                    outputs[0]["scores"][j],
                )
                for j in range(len(outputs[0]["boxes"]))
                if outputs[0]["scores"][j] > flags.FLAGS.threshold
                and outputs[0]["labels"][j] > 0
            ]

            inference_boxes, inference_labels, inference_scores = (
                zip(*filtered_output) if len(filtered_output) > 0 else ([], [], [])
            )

            draw_bboxes(
                inference_ax,
                inference_boxes,
                inference_labels,
                labels,
                label_colors,
                inference_scores,
            )

            plt.pause(0.001)

    # evaluate on the test dataset
    #    evaluate(model, data_loader, device=device)

    print("Visualization complete")


if __name__ == "__main__":
    app.run(main)
