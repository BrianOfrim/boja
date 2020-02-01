import copy
import os
import time
import threading
import typing
import queue

from absl import app, flags
from cv2 import cv2
from genicam.gentl import TimeoutException
from harvesters.core import Harvester
import numpy as np

from .._file_utils import create_output_dir
from .._image_utils import RGB8Image
from .._s3_utils import s3_upload_files, s3_bucket_exists

from .._settings import (
    DEFAULT_LOCAL_DATA_DIR,
    DEFAULT_S3_DATA_DIR,
    DEFAULT_GENTL_PRODUCER_PATH,
    IMAGE_DIR_NAME,
    IMAGE_FILE_TYPE,
    NETWORKS,
)

WINDOW_NAME = "Capture"

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "gentl_producer_path",
    DEFAULT_GENTL_PRODUCER_PATH,
    "Path to the GenTL producer .cti file to use.",
)

flags.DEFINE_integer(
    "display_width", 1080, "Target image width for the display window.",
)

flags.DEFINE_integer("frame_rate", 30, "Frame rate to acquire images at.")

flags.DEFINE_string(
    "local_data_dir",
    DEFAULT_LOCAL_DATA_DIR,
    "Local parent directory of the image directory to store images.",
)

flags.DEFINE_string("s3_bucket_name", None, "S3 bucket to send images to.")

flags.DEFINE_string(
    "s3_data_dir", DEFAULT_S3_DATA_DIR, "Prefix of the s3 data objects."
)

flags.DEFINE_enum(
    "network", NETWORKS[0], NETWORKS, "The neural network to use for object detection",
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


def acquire_images(cam, save_queue: queue.Queue) -> None:
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)

    cam.start_image_acquisition()
    print("Acquisition started.")
    print("Press enter to save images. Press escape to exit.")
    while True:
        retrieved_image = get_newest_image(cam)
        if retrieved_image is None:
            break

        cv2.imshow(
            WINDOW_NAME, retrieved_image.get_resized_image(FLAGS.display_width),
        )
        keypress = cv2.waitKey(1)

        if keypress == 27:
            # escape key pressed
            break
        elif cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            # x button clicked
            break
        elif keypress == 13:
            # Enter key pressed
            cv2.imshow(
                WINDOW_NAME, retrieved_image.get_highlighted_image(FLAGS.display_width),
            )
            save_queue.put(retrieved_image)
            cv2.waitKey(500)

    save_queue.put(None)
    cv2.destroyWindow(WINDOW_NAME)
    cam.stop_image_acquisition()
    print("Acquisition Ended.")


def save_images(save_queue: queue.Queue, use_s3: bool) -> None:

    while True:
        image = save_queue.get(block=True)
        if image is None:
            break
        file_path = os.path.join(
            FLAGS.local_data_dir,
            IMAGE_DIR_NAME,
            "%i.%s" % (time.time(), IMAGE_FILE_TYPE),
        )
        save_successfull = image.save(file_path)
        print("Image saved at: %s" % file_path)

        if use_s3 and save_successfull:
            s3_upload_files(
                FLAGS.s3_bucket_name,
                [file_path],
                "/".join([FLAGS.s3_data_dir, IMAGE_DIR_NAME]),
            )

    print("Saving complete.")


def apply_camera_settings(cam) -> None:
    # Configure newest only buffer handling
    cam.keep_latest = True
    cam.num_filled_buffers_to_hold = 1

    # Configure frame rate
    cam.remote_device.node_map.AcquisitionFrameRateEnable.value = True
    cam.remote_device.node_map.AcquisitionFrameRate.value = min(
        FLAGS.frame_rate, cam.remote_device.node_map.AcquisitionFrameRate.max
    )
    print(
        "Acquisition frame rate set to: %3.1f"
        % cam.remote_device.node_map.AcquisitionFrameRate.value
    )


def main(unused_argv):

    if not create_output_dir(os.path.join(FLAGS.local_data_dir, IMAGE_DIR_NAME)):
        print("Cannot create output annotations directory.")
        return

    use_s3 = True if FLAGS.s3_bucket_name is not None else False

    if use_s3:
        if not s3_bucket_exists(FLAGS.s3_bucket_name):
            use_s3 = False
            print(
                "Bucket: %s either does not exist or you do not have access to it"
                % FLAGS.s3_bucket_name
            )
        else:
            print("Bucket: %s exists and you have access to it" % FLAGS.s3_bucket_name)

    h = Harvester()
    h.add_cti_file(FLAGS.gentl_producer_path)
    if len(h.cti_files) == 0:
        print("No valid cti file found at %s" % FLAGS.gentl_producer_path)
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

    save_queue = queue.Queue()

    save_thread = threading.Thread(target=save_images, args=(save_queue, use_s3,))

    save_thread.start()

    acquire_images(cam, save_queue)

    save_thread.join()

    # clean up
    cam.destroy()
    h.reset()

    print("Exiting.")


if __name__ == "__main__":
    app.run(main)
