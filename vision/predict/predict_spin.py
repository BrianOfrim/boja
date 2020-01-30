import os

from absl import app, flags
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PySpin
import torch
import torchvision.transforms.functional as F

from .._file_utils import get_highest_numbered_file
from .._image_utils import RGB8Image, draw_bboxes
from .. import _models
from .._s3_utils import s3_bucket_exists, s3_download_highest_numbered_file
from .._settings import (
    DEFAULT_LOCAL_DATA_DIR,
    DEFAULT_S3_DATA_DIR,
    LABEL_FILE_NAME,
    MODEL_STATE_DIR_NAME,
    MODEL_STATE_FILE_TYPE,
    NETWORKS,
)


matplotlib.use("TKAgg")

INFERENCE_WINDOW_NAME = "Inference"


flags.DEFINE_string(
    "s3_bucket_name", None, "S3 bucket to retrieve images from and upload manifest to."
)

flags.DEFINE_string(
    "s3_data_dir", DEFAULT_S3_DATA_DIR, "Prefix of the s3 data objects."
)


flags.DEFINE_string(
    "local_data_dir", DEFAULT_LOCAL_DATA_DIR, "Local data directory.",
)

flags.DEFINE_float(
    "threshold", 0.5, "The threshold above which to display predicted bounding boxes"
)

flags.DEFINE_string("model_path", None, "The model to load. Default is newest.")


flags.DEFINE_integer("frame_rate", 30, "Frame rate to acquire images at.")


flags.DEFINE_enum(
    "network", NETWORKS[0], NETWORKS, "The neural network to use for object detection",
)


def get_newest_saved_model_path(model_dir_path: str, filter_keyword=None) -> str:
    return get_highest_numbered_file(
        model_dir_path, MODEL_STATE_FILE_TYPE, filter_keyword
    )


def get_newest_image(cam, pixel_format):
    try:
        spinnaker_image = cam.GetNextImage()
        retrieved_image = RGB8Image(
            spinnaker_image.GetWidth(),
            spinnaker_image.GetHeight(),
            pixel_format,
            spinnaker_image.GetData().copy(),
        )
        spinnaker_image.Release()
        return retrieved_image
    except ValueError as err:
        print(err)
        return None


def key_press(event, continue_streaming):

    if event.key == "escape":
        continue_streaming[0] = False


def display_images(cam, labels, saved_model_file_path) -> None:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # get the model using our helper function
    model = _models.__dict__[flags.FLAGS.network](
        len(labels),
        box_score_thresh=flags.FLAGS.threshold,
        min_size=600,
        max_size=800,
        box_nms_thresh=0.3,
    )

    print("Loading model state from: %s" % saved_model_file_path)

    model.load_state_dict(torch.load(saved_model_file_path, map_location=device))

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
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam.BeginAcquisition()

    pixel_format = cam.PixelFormat.GetCurrentEntry().GetSymbolic()

    with torch.no_grad():
        while continue_streaming[0]:
            retrieved_image = get_newest_image(cam, pixel_format)

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
                if outputs[0]["scores"][j] > flags.FLAGS.threshold
                and outputs[0]["labels"][j] > 0
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
    cam.EndAcquisition()


def apply_camera_settings(cam) -> None:
    # Configure newest only buffer handling
    s_node_map = cam.GetTLStreamNodeMap()

    # Retrieve Buffer Handling Mode Information
    handling_mode = PySpin.CEnumerationPtr(
        s_node_map.GetNode("StreamBufferHandlingMode")
    )
    handling_mode_entry = handling_mode.GetEntryByName("NewestOnly")
    handling_mode.SetIntValue(handling_mode_entry.GetValue())

    # Set stream buffer Count Mode to manual
    stream_buffer_count_mode = PySpin.CEnumerationPtr(
        s_node_map.GetNode("StreamBufferCountMode")
    )
    stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(
        stream_buffer_count_mode.GetEntryByName("Manual")
    )
    stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())

    # Retrieve and modify Stream Buffer Count
    buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode("StreamBufferCountManual"))

    buffer_count.SetValue(3)

    # Display Buffer Info
    print("Buffer Handling Mode: %s" % handling_mode_entry.GetDisplayName())
    print("Buffer Count: %d" % buffer_count.GetValue())
    print("Maximum Buffer Count: %d" % buffer_count.GetMax())

    # Configure frame rate
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(
        min(flags.FLAGS.frame_rate, cam.AcquisitionFrameRate.GetMax())
    )
    print("Acquisition frame rate set to: %3.1f" % cam.AcquisitionFrameRate.GetValue())


def main(unused_argv):

    use_s3 = True if flags.FLAGS.s3_bucket_name is not None else False

    if use_s3:
        if not s3_bucket_exists(flags.FLAGS.s3_bucket_name):
            use_s3 = False
            print(
                "Bucket: %s either does not exist or you do not have access to it"
                % flags.FLAGS.s3_bucket_name
            )
        else:
            print(
                "Bucket: %s exists and you have access to it"
                % flags.FLAGS.s3_bucket_name
            )

    if use_s3:
        # Get the newest model
        s3_download_highest_numbered_file(
            flags.FLAGS.s3_bucket_name,
            "/".join([flags.FLAGS.s3_data_dir, MODEL_STATE_DIR_NAME]),
            os.path.join(flags.FLAGS.local_data_dir, MODEL_STATE_DIR_NAME),
            MODEL_STATE_FILE_TYPE,
            flags.FLAGS.network,
        )

    label_file_path = os.path.join(flags.FLAGS.local_data_dir, LABEL_FILE_NAME)
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

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print("Number of cameras detected: %d" % num_cameras)
    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print("Not enough cameras!")
        input("Done! Press Enter to exit...")
        return

    cam = cam_list.GetByIndex(0)

    cam.Init()

    apply_camera_settings(cam)

    display_images(cam, labels, saved_model_file_path)

    cam.DeInit()

    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    print("Exiting.")


if __name__ == "__main__":
    app.run(main)
