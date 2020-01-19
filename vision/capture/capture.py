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

from .._s3_utils import s3_upload_files, s3_bucket_exists

WINDOW_NAME = "Capture"

IMAGE_DIR_NAME = "images"
IMAGE_FILE_TYPE = "jpg"

flags.DEFINE_string(
    "gentl_producer_path",
    "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti",
    "Path to the GenTL producer .cti file to use.",
)

flags.DEFINE_integer(
    "display_width", 1080, "Target image width for the display window.",
)

flags.DEFINE_integer("frame_rate", 30, "Frame rate to acquire images at.")

flags.DEFINE_string(
    "local_data_dir",
    os.path.join(os.path.expanduser("~"), "boja", "data"),
    "Local parent directory of the image directory to store images.",
)

flags.DEFINE_string("s3_bucket_name", None, "S3 bucket to send images to.")

flags.DEFINE_string("s3_data_dir", "data", "Prefix of the s3 data objects.")


class RGB8Image:
    BORDER_COLOR = (3, 252, 53)
    BORDER_WIDTH = 10

    def __init__(
        self, width: int, height: int, data_format: str, image_data: np.ndarray
    ):
        self.image_data: np.ndarray = self._process_image(
            image_data, data_format, width, height
        )

    def get_height(self):
        return self.image_data.shape[0]

    def get_width(self):
        return self.image_data.shape[1]

    def get_channels(self):
        if len(self.image_data.shape) < 3:
            return 1
        return self.image_data.shape[2]

    def get_data(self) -> np.ndarray:
        return self.image_data

    def _process_image(self, image_data, data_format, width, height) -> np.ndarray:
        if data_format == "Mono8":
            return cv2.cvtColor(image_data.reshape(height, width), cv2.COLOR_GRAY2RGB)
        elif data_format == "BayerRG8":
            return cv2.cvtColor(
                image_data.reshape(height, width), cv2.COLOR_BayerRG2RGB
            )
        elif data_format == "BayerGR8":
            return cv2.cvtColor(
                image_data.reshape(height, width), cv2.COLOR_BayerGR2RGB
            )
        elif data_format == "BayerGB8":
            return cv2.cvtColor(
                image_data.reshape(height, width), cv2.COLOR_BayerGB2RGB
            )
        elif data_format == "BayerBG8":
            return cv2.cvtColor(
                image_data.reshape(height, width), cv2.COLOR_BayerBG2RGB
            )
        elif data_format == "RGB8":
            return image_data.reshape(height, width, 3)
        elif data_format == "BGR8":
            return cv2.cvtColor(image_data.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
        else:
            print("Unsupported pixel format: %s" % data_format)
            raise ValueError("Unsupported pixel format: %s" % data_format)

    def get_resized_image(self, target_width: int) -> np.ndarray:
        resize_ratio = float(target_width / self.get_width())
        return cv2.resize(self.image_data, (0, 0), fx=resize_ratio, fy=resize_ratio)

    def get_highlighted_image(self, target_width: int = None) -> np.ndarray:
        return cv2.copyMakeBorder(
            (
                self.get_resized_image(target_width)
                if target_width is not None
                else self.get_data()
            )[
                self.BORDER_WIDTH : -self.BORDER_WIDTH,
                self.BORDER_WIDTH : -self.BORDER_WIDTH,
            ],
            top=self.BORDER_WIDTH,
            bottom=self.BORDER_WIDTH,
            left=self.BORDER_WIDTH,
            right=self.BORDER_WIDTH,
            borderType=cv2.BORDER_ISOLATED,
            value=self.BORDER_COLOR,
        )

    def save(self, file_path: str) -> bool:
        try:
            cv2.imwrite(file_path, self.get_data())
        except FileExistsError:
            return False
        return True


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
    try:
        cam.start_image_acquisition()
        print("Acquisition started.")
        while True:
            retrieved_image = get_newest_image(cam)
            if retrieved_image is None:
                break

            cv2.imshow(
                WINDOW_NAME,
                retrieved_image.get_resized_image(flags.FLAGS.display_width),
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
                    WINDOW_NAME,
                    retrieved_image.get_highlighted_image(flags.FLAGS.display_width),
                )
                save_queue.put(retrieved_image)
                cv2.waitKey(500)
    finally:
        save_queue.put(None)
        cv2.destroyWindow(WINDOW_NAME)
        cam.stop_image_acquisition()
        print("Acquisition Ended.")


def save_images(save_queue: queue.Queue, use_s3: bool) -> None:
    try:
        while True:
            image = save_queue.get(block=True)
            if image is None:
                break
            file_path = os.path.join(
                flags.FLAGS.local_data_dir,
                IMAGE_DIR_NAME,
                "%i.%s" % (time.time(), IMAGE_FILE_TYPE),
            )
            save_successfull = image.save(file_path)
            print("Image saved at: %s" % file_path)

            if use_s3 and save_successfull:
                s3_upload_files(
                    flags.FLAGS.s3_bucket_name,
                    [file_path],
                    "/".join([flags.FLAGS.s3_data_dir, IMAGE_DIR_NAME]),
                )
    finally:
        print("Saving complete.")


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


def create_output_dir(dir_name) -> bool:
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        print("Creating output directory: %s" % dir_name)
        try:
            os.makedirs(dir_name)
        except OSError:
            print("Creation of the directory %s failed" % dir_name)
            return False
        else:
            print("Successfully created the directory %s " % dir_name)
            return True
    else:
        print("Output directory exists.")
        return True


def main(unused_argv):

    if not create_output_dir(os.path.join(flags.FLAGS.local_data_dir, IMAGE_DIR_NAME)):
        print("Cannot create output annotations directory.")
        return

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
