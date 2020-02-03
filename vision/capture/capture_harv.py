import copy
import os
import time
import threading
import typing
import queue

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
            WINDOW_NAME, retrieved_image.get_resized_image(args.display_width),
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
                WINDOW_NAME, retrieved_image.get_highlighted_image(args.display_width),
            )
            save_queue.put(retrieved_image)
            cv2.waitKey(500)

    save_queue.put(None)
    cv2.destroyWindow(WINDOW_NAME)
    cam.stop_image_acquisition()
    print("Acquisition Ended.")


def save_images(
    save_queue: queue.Queue,
    local_data_dir,
    use_s3: bool = False,
    s3_bucket_name: str = None,
    s3_data_dir: str = None,
) -> None:

    while True:
        image = save_queue.get(block=True)
        if image is None:
            break
        file_path = os.path.join(
            local_data_dir,
            IMAGE_DIR_NAME,
            "%i.%s" % ((time.time() * 1000), IMAGE_FILE_TYPE),
        )
        save_successfull = image.save(file_path)
        print("Image saved at: %s" % file_path)

        if use_s3 and save_successfull:
            s3_upload_files(
                s3_bucket_name, [file_path], "/".join([s3_data_dir, IMAGE_DIR_NAME]),
            )

    print("Saving complete.")


def apply_camera_settings(cam, framerate: float = 30.0) -> None:
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

    if not create_output_dir(os.path.join(args.local_data_dir, IMAGE_DIR_NAME)):
        print("Cannot create output annotations directory.")
        return

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

    apply_camera_settings(cam, args.framerate)

    save_queue = queue.Queue()

    save_thread = threading.Thread(
        target=save_images,
        args=(save_queue, use_s3, args.s3_bucket_name, args.s3_data_dir,),
    )

    save_thread.start()

    acquire_images(cam, save_queue)

    save_thread.join()

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
    parser.add_argument(
        "--frame_rate", type=float, default=30.0,
    )
    parser.add_argument(
        "--display_width", type=int, default=1080,
    )

    args = parser.parse_args()

    main(args)

