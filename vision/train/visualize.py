import os
import re
import time
from typing import List

from absl import app, flags
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision.transforms.functional as F

from .datasets import BojaDataSet
from .. import _models
from .transforms import ToTensor, RandomHorizontalFlip, Compose

matplotlib.use("TKAgg")

from .._settings import (
    IMAGE_DIR_NAME,
    ANNOTATION_DIR_NAME,
    MANIFEST_DIR_NAME,
    MODEL_STATE_DIR_NAME,
    IMAGE_FILE_TYPE,
    ANNOTATION_FILE_TYPE,
    MANIFEST_FILE_TYPE,
    MODEL_STATE_FILE_TYPE,
    NETWORKS,
)

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

flags.DEFINE_enum(
    "network", NETWORKS[0], NETWORKS, "The neural network to use for object detection",
)


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


def int_string_sort(file_name) -> int:
    match = re.match("[0-9]+", file_name)
    if not match:
        return 0
    return int(match[0])


def get_highest_numbered_file(
    dir_path: str, file_extention: str = None, filter_keyword=None
) -> str:
    file_names = get_files_from_dir(dir_path)
    if file_extention is not None:
        file_names = [
            file_name
            for file_name in file_names
            if file_name.lower().endswith(file_extention.lower())
        ]
    if filter_keyword is not None:
        file_names = [
            file_name
            for file_name in file_names
            if filter_keyword.lower() in file_name.lower()
        ]
    if len(file_names) == 0:
        return None
    highest_numbered_file = sorted(file_names, key=int_string_sort, reverse=True)[0]
    return os.path.join(dir_path, highest_numbered_file)


def get_newest_manifest_path(manifest_dir_path: str) -> str:
    return get_highest_numbered_file(manifest_dir_path, MANIFEST_FILE_TYPE)


def get_newest_saved_model_path(model_dir_path: str, filter_keyword=None) -> str:
    return get_highest_numbered_file(
        model_dir_path, MODEL_STATE_FILE_TYPE, filter_keyword
    )


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
            os.path.join(flags.FLAGS.local_data_dir, MODEL_STATE_DIR_NAME),
            flags.FLAGS.network,
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
    dataset = BojaDataSet(
        os.path.join(flags.FLAGS.local_data_dir, IMAGE_DIR_NAME),
        os.path.join(flags.FLAGS.local_data_dir, ANNOTATION_DIR_NAME),
        manifest_file_path,
        get_transform(train=False),
        labels,
    )

    # get the model using our helper function
    model = _models.__dict__[flags.FLAGS.network](num_classes)

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
            # make a copy of the image for display before sending to device
            display_image_base = F.to_pil_image(image)
            image = image.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            model_time = time.time()
            outputs = model([image])
            outputs = [
                {k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs
            ]
            model_time = time.time() - model_time
            print("Inference time = ", model_time)

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
