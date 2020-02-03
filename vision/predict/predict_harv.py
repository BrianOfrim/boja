import os

from genicam.gentl import TimeoutException
from harvesters.core import Harvester

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F

from .._file_utils import get_highest_numbered_file
from .._image_utils import RGB8Image, draw_bboxes
from .. import _models
from .._s3_utils import s3_bucket_exists, s3_download_highest_numbered_file
from .._settings import (
    DEFAULT_LOCAL_DATA_DIR,
    DEFAULT_S3_DATA_DIR,
    DEFAULT_GENTL_PRODUCER_PATH,
    LABEL_FILE_NAME,
    MODEL_STATE_DIR_NAME,
    MODEL_STATE_FILE_TYPE,
    NETWORKS,
)


matplotlib.use("TKAgg")

INFERENCE_WINDOW_NAME = "Inference"


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


def display_images(
    cam, labels, network_type, saved_model_file_path, threshold=0.5
) -> None:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # get the model using our helper function
    model = _models.__dict__[network_type](
        len(labels),
        box_score_thresh=threshold,
        min_size=600,
        max_size=800,
        box_nms_thresh=0.3,
    )

    print("Loading model state from: %s" % saved_model_file_path)
    checkpoint = torch.load(saved_model_file_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    # move model to the right device
    model.to(device)

    model.eval()

    # create plots
    fig, inference_ax = plt.subplots()

    fig.canvas.set_window_title("Predict")

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
                {k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs
            ]

            # filter out the background labels and scores bellow threshold
            filtered_output = [
                (
                    outputs[0]["boxes"][j],
                    outputs[0]["labels"][j],
                    outputs[0]["scores"][j],
                )
                for j in range(len(outputs[0]["boxes"]))
                if outputs[0]["scores"][j] > threshold and outputs[0]["labels"][j] > 0
            ]

            inference_boxes, inference_labels, inference_scores = (
                zip(*filtered_output) if len(filtered_output) > 0 else ([], [], [])
            )

            inference_ax.clear()

            inference_ax.imshow(image_data)

            draw_bboxes(
                inference_ax,
                inference_boxes,
                inference_labels,
                labels,
                label_colors,
                inference_scores,
            )

            plt.pause(0.001)

    print("Ending live stream")
    cam.stop_image_acquisition()


def apply_camera_settings(cam, framerate=30.0) -> None:
    # Configure newest only buffer handling
    cam.keep_latest = True
    cam.num_filled_buffers_to_hold = 1

    # Configure frame rate
    cam.remote_device.node_map.AcquisitionFrameRateEnable.value = True
    cam.remote_device.node_map.AcquisitionFrameRate.value = min(
        framerate, cam.remote_device.node_map.AcquisitionFrameRate.max
    )
    print(
        "Acquisition frame rate set to: %3.1f"
        % cam.remote_device.node_map.AcquisitionFrameRate.value
    )


def main(args):

    use_s3 = True if args.s3_bucket_name is not None else False

    if use_s3:
        if not s3_bucket_exists(args.s3_bucket_name):
            use_s3 = False
            print(
                "Bucket: %s either does not exist or you do not have access to it"
                % args.s3_bucket_name
            )
        else:
            print("Bucket: %s exists and you have access to it" % args.s3_bucket_name)

    if use_s3:
        # Get the newest model
        s3_download_highest_numbered_file(
            args.s3_bucket_name,
            "/".join([args.s3_data_dir, MODEL_STATE_DIR_NAME]),
            os.path.join(args.local_data_dir, MODEL_STATE_DIR_NAME),
            MODEL_STATE_FILE_TYPE,
            args.network,
        )

    label_file_path = os.path.join(args.local_data_dir, LABEL_FILE_NAME)
    if not os.path.isfile(label_file_path):
        print("Missing file %s" % label_file_path)
        return

    # read in the category labels
    labels = open(label_file_path).read().splitlines()

    if len(labels) == 0:
        print("No label categories found in %s" % label_file_path)
        return

    # Add the background as the first class
    labels.insert(0, "background")

    print("Labels found:")
    print(labels)

    saved_model_file_path = (
        args.model_path
        if args.model_path is not None
        else get_newest_saved_model_path(
            os.path.join(args.local_data_dir, MODEL_STATE_DIR_NAME), args.network,
        )
    )

    if saved_model_file_path is None:
        print("No saved model state found")
        return

    h = Harvester()
    h.add_cti_file(args.gentl_producer_path)
    if len(h.cti_files) == 0:
        print("No valid cti file found at %s" % args.gentl_producer_path)
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

    display_images(cam, labels, args.network, saved_model_file_path, args.threshold)

    # clean up
    cam.destroy()
    h.reset()

    print("Exiting.")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gentl_producer_path",
        type=str,
        default=DEFAULT_GENTL_PRODUCER_PATH,
        help="Path to the GenTL producer .cti file to use",
    )
    parser.add_argument("--s3_bucket_name", type=str)
    parser.add_argument(
        "--s3_data_dir",
        type=str,
        default=DEFAULT_S3_DATA_DIR,
        help="Prefix of the s3 data objects",
    )
    parser.add_argument(
        "--local_data_dir", type=str, default=DEFAULT_LOCAL_DATA_DIR,
    )
    parser.add_argument("--model_path", type=str, help="The model to load")
    parser.add_argument(
        "--network",
        type=str,
        choices=NETWORKS,
        default=NETWORKS[0],
        help="The neural network to use for object detection",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="The threshold above which to display predicted bounding boxes",
    )
    parser.add_argument(
        "--frame_rate", type=float, default=30.0,
    )

    args = parser.parse_args()

    main(args)

