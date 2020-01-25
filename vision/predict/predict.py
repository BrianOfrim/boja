import copy
import os
import time
from typing import List

from absl import app, flags
from cv2 import cv2
from genicam.gentl import TimeoutException
from harvesters.core import Harvester

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision.transforms.functional as F

from .._file_utils import get_files_from_dir, get_highest_numbered_file
from .._image_utils import RGB8Image
from .. import _models
from .._settings import MODEL_STATE_DIR_NAME, MODEL_STATE_FILE_TYPE, NETWORKS


matplotlib.use("TKAgg")

INFERENCE_WINDOW_NAME = "Inference"


flags.DEFINE_string(
    "gentl_producer_path",
    "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti",
    "Path to the GenTL producer .cti file to use.",
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

flags.DEFINE_float(
    "threshold", 0.5, "The threshold above which to display predicted bounding boxes"
)

flags.DEFINE_string("model_path", None, "The model to load. Default is newest.")


flags.DEFINE_integer("frame_rate", 30, "Frame rate to acquire images at.")


flags.DEFINE_enum(
    "network", NETWORKS[0], NETWORKS, "The neural network to use for object detection",
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


def get_newest_saved_model_path(model_dir_path: str, filter_keyword=None) -> str:
    return get_highest_numbered_file(
        model_dir_path, MODEL_STATE_FILE_TYPE, filter_keyword
    )


def get_newest_image(cam):
    try:
        retrieved_image = None
        with cam.fetch_buffer() as buffer:
            component = buffer.payload.components[0]
            retrieved_image = RGB8Image(
                component.width,
                component.height,
                component.data_format,
                component.data.copy(),
            )
        return retrieved_image
    except TimeoutException:
        print("Timeout ocurred waiting for image.")
        return None
    except ValueError as err:
        print(err)
        return None


def key_press(event, continue_streaming):

    if event.key == "escape":
        continue_streaming[0] = False


def display_images(cam, labels, saved_model_file_path) -> None:
    try:

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # get the model using our helper function
        model = _models.__dict__[flags.FLAGS.network](len(labels))

        print("Loading model state from: %s" % saved_model_file_path)

        model.load_state_dict(torch.load(saved_model_file_path))

        # move model to the right device
        model.to(device)

        model.eval()

        # create plots
        fig, (im_ax, inference_ax) = plt.subplots(1, 2)

        continue_streaming = [True]

        fig.canvas.mpl_connect(
            "key_press_event", lambda event: key_press(event, continue_streaming)
        )

        print("Model state loaded")

        label_colors = plt.get_cmap("hsv")(np.linspace(0, 0.9, len(labels)))

        print("Starting inference")

        print("Starting live stream.")
        cam.start_image_acquisition()

        with torch.no_grad():
            while continue_streaming[0]:
                retrieved_image = get_newest_image(cam)

                if retrieved_image is None:
                    break

                image_data = RGB8Image.to_bgr(retrieved_image.get_data())

                tensor_image = F.to_tensor(image_data)
                tensor_image = tensor_image.to(device)

                outputs = model([tensor_image])
                outputs = [
                    {k: v.to(torch.device("cpu")) for k, v in t.items()}
                    for t in outputs
                ]

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

                inference_ax.clear()
                im_ax.clear()

                inference_ax.imshow(image_data)
                im_ax.imshow(image_data)

                draw_bboxes(
                    inference_ax,
                    inference_boxes,
                    inference_labels,
                    labels,
                    label_colors,
                    inference_scores,
                )

                plt.pause(0.001)

    finally:
        print("Ending live stream")
        cam.stop_image_acquisition()


def apply_camera_settings(cam) -> None:
    # Configure newest only buffer handling
    cam.keep_latest = True
    cam.num_filled_buffers_to_hold = 1

    # Configure frame rate
    cam.remote_device.node_map.AcquisitionFrameRateEnable.value = True
    cam.remote_device.node_map.AcquisitionFrameRate.value = min(
        flags.FLAGS.frame_rate, cam.remote_device.node_map.AcquisitionFrameRate.max
    )
    print(
        "Acquisition frame rate set to: %3.1f"
        % cam.remote_device.node_map.AcquisitionFrameRate.value
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

    h = Harvester()
    h.add_cti_file(flags.FLAGS.gentl_producer_path)
    if len(h.cti_files) == 0:
        print("No valid cti file found at %s" % flags.FLAGS.gentl_producer_path)
        h.reset()
        return
    print("Currently available genTL Producer CTI files: ", h.cti_files)

    h.update_device_info_list()
    if len(h.device_info_list) == 0:
        print("No compatible devices detected.")
        h.reset()
        return

    print("Available devices List: ", h.device_info_list)
    print("Using device: ", h.device_info_list[0])

    cam = h.create_image_acquirer(list_index=0)

    apply_camera_settings(cam)

    display_images(cam, labels, saved_model_file_path)

    # clean up
    cam.destroy()
    h.reset()

    print("Exiting.")


if __name__ == "__main__":
    app.run(main)
