import copy
import os
import time
import threading
import typing
import queue

from absl import app, flags

import cv2
import numpy as np
import PySpin
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


def acquire_images(cam, save_queue: queue.Queue) -> None:
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam.BeginAcquisition()
    print("Acquisition started.")
    print("Press enter to save images. Press escape to exit.")

    pixel_format = cam.PixelFormat.GetCurrentEntry().GetSymbolic()

    while True:
        retrieved_image = get_newest_image(cam, pixel_format)
        if retrieved_image is None:
            break

        cv2.imshow(
            WINDOW_NAME, retrieved_image.get_resized_image(FLAGS.display_width),
        )
        keypress = cv2.waitKey(1)
        if keypress == 27:
            # escape key pressed
            break
        #        elif cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        #            # x button clicked
        #            break
        elif keypress == 13:
            # Enter key pressed
            cv2.imshow(
                WINDOW_NAME, retrieved_image.get_highlighted_image(FLAGS.display_width),
            )
            save_queue.put(retrieved_image)
            cv2.waitKey(500)

    save_queue.put(None)
    cv2.destroyWindow(WINDOW_NAME)
    cam.EndAcquisition()
    print("Acquisition Ended.")


def save_images(save_queue: queue.Queue, use_s3: bool) -> None:

    while True:
        image = save_queue.get(block=True)
        if image is None:
            break
        file_path = os.path.join(
            FLAGS.local_data_dir,
            IMAGE_DIR_NAME,
            "%i.%s" % ((time.time() * 1000), IMAGE_FILE_TYPE),
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
        min(FLAGS.frame_rate, cam.AcquisitionFrameRate.GetMax())
    )
    print("Acquisition frame rate set to: %3.1f" % cam.AcquisitionFrameRate.GetValue())


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

    save_queue = queue.Queue()

    save_thread = threading.Thread(target=save_images, args=(save_queue, use_s3,))

    save_thread.start()

    acquire_images(cam, save_queue)

    save_thread.join()

    cam.DeInit()

    del cam
    cam_list.Clear()
    system.ReleaseInstance()

    print("Exiting.")


if __name__ == "__main__":
    app.run(main)
