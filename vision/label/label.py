import os
import re
import time
from typing import List
import shutil

from absl import app, flags
import numpy as np
import matplotlib.pyplot as plt

from .._file_utils import (
    create_output_dir,
    get_files_from_dir,
    get_highest_numbered_file,
)
from .gui import GUI, AnnotatedImage, Category
from .._s3_utils import (
    s3_upload_files,
    s3_bucket_exists,
    s3_download_dir,
)
from .._settings import (
    DEFAULT_LOCAL_DATA_DIR,
    LABEL_FILE_NAME,
    IMAGE_DIR_NAME,
    ANNOTATION_DIR_NAME,
    MANIFEST_DIR_NAME,
    IMAGE_FILE_TYPE,
    ANNOTATION_FILE_TYPE,
    MANIFEST_FILE_TYPE,
)

flags.DEFINE_string(
    "local_data_dir", DEFAULT_LOCAL_DATA_DIR, "Local directory of the data to label.",
)

flags.DEFINE_string(
    "s3_bucket_name", None, "S3 bucket to retrieve images from and upload manifest to."
)

flags.DEFINE_string("s3_data_dir", "data", "Prefix of the s3 data objects.")


def get_newest_manifest_path(manifest_dir_path: str) -> str:
    return get_highest_numbered_file(manifest_dir_path, MANIFEST_FILE_TYPE)


def save_outputs(
    annotatedImages: List[AnnotatedImage],
    previous_manifest_path: str,
    start_time: int,
    use_s3: bool,
) -> None:
    # create a new manifest file
    new_manifest_path = os.path.join(
        flags.FLAGS.local_data_dir,
        MANIFEST_DIR_NAME,
        "%i-manifest.%s" % (start_time, MANIFEST_FILE_TYPE),
    )
    if previous_manifest_path is not None:
        shutil.copyfile(previous_manifest_path, new_manifest_path)
    else:
        open(new_manifest_path, "a").close()

    new_annotation_filepaths = []
    with open(new_manifest_path, "a") as manifest:
        for image in annotatedImages:
            annotation_filepath = image.write_to_pascal_voc()
            image_filename = os.path.basename(image.image_path)
            annotation_filename = (
                os.path.basename(annotation_filepath)
                if annotation_filepath is not None
                else "Invalid"
            )
            if annotation_filepath is not None:
                new_annotation_filepaths.append(annotation_filepath)
            manifest.write("%s,%s\n" % (image_filename, annotation_filename,))
    if use_s3:
        s3_upload_files(
            flags.FLAGS.s3_bucket_name,
            new_annotation_filepaths,
            flags.FLAGS.s3_data_dir + "/" + ANNOTATION_DIR_NAME,
        )
        s3_upload_files(
            flags.FLAGS.s3_bucket_name,
            [new_manifest_path],
            flags.FLAGS.s3_data_dir + "/" + MANIFEST_DIR_NAME,
        )
        # ensure that all images have been uploaded
        s3_upload_files(
            flags.FLAGS.s3_bucket_name,
            [image.image_path for image in annotatedImages],
            flags.FLAGS.s3_data_dir + "/" + IMAGE_DIR_NAME,
        )


def main(unused_argv):

    start_time = time.time()

    fig = plt.figure()
    gui = GUI(fig)

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
        # Download new images from s3
        s3_download_dir(
            flags.FLAGS.s3_bucket_name,
            "/".join([flags.FLAGS.s3_data_dir, IMAGE_DIR_NAME]),
            os.path.join(flags.FLAGS.local_data_dir, IMAGE_DIR_NAME),
            IMAGE_FILE_TYPE,
        )

        # Download any nest annotation files from s3
        s3_download_dir(
            flags.FLAGS.s3_bucket_name,
            "/".join([flags.FLAGS.s3_data_dir, ANNOTATION_DIR_NAME]),
            os.path.join(flags.FLAGS.local_data_dir, ANNOTATION_DIR_NAME),
            ANNOTATION_FILE_TYPE,
        )

        # Download any new manifests files from s3
        s3_download_dir(
            flags.FLAGS.s3_bucket_name,
            "/".join([flags.FLAGS.s3_data_dir, MANIFEST_DIR_NAME]),
            os.path.join(flags.FLAGS.local_data_dir, MANIFEST_DIR_NAME),
            MANIFEST_FILE_TYPE,
        )

    label_file_path = os.path.join(flags.FLAGS.local_data_dir, LABEL_FILE_NAME)
    if not os.path.isfile(label_file_path):
        print("Missing file %s" % label_file_path)
        return

    # read in the category labels
    category_labels = open(label_file_path).read().splitlines()

    if len(category_labels) == 0:
        print("No label categories found in %s" % label_file_path)
        return

    category_colors = plt.get_cmap("hsv")(np.linspace(0, 0.9, len(category_labels)))

    for index, (name, color) in enumerate(zip(category_labels, category_colors)):
        gui.add_category(Category(name, tuple(color), str(index)))

    if not os.path.isdir(os.path.join(flags.FLAGS.local_data_dir, IMAGE_DIR_NAME)):
        print("Invalid input image directory")
        return

    previous_manifest_file = get_newest_manifest_path(
        os.path.join(flags.FLAGS.local_data_dir, MANIFEST_DIR_NAME)
    )
    manifest_images = set()
    if previous_manifest_file is not None:
        with open(previous_manifest_file, "r") as manifest:
            for line in manifest:
                manifest_images.add(line.split(",")[0].rstrip())

    # read in the names of the images to label
    for image_file in os.listdir(
        os.path.join(flags.FLAGS.local_data_dir, IMAGE_DIR_NAME)
    ):
        if (
            image_file.endswith(IMAGE_FILE_TYPE)
            and os.path.basename(image_file) not in manifest_images
        ):
            gui.add_image(
                AnnotatedImage(
                    os.path.join(
                        flags.FLAGS.local_data_dir, IMAGE_DIR_NAME, image_file
                    ),
                    os.path.join(flags.FLAGS.local_data_dir, ANNOTATION_DIR_NAME),
                )
            )

    if len(gui.images) == 0:
        print("No input images found")
        return

    if not create_output_dir(
        os.path.join(flags.FLAGS.local_data_dir, ANNOTATION_DIR_NAME)
    ):
        print("Cannot create output annotations directory.")
        return

    if not create_output_dir(
        os.path.join(flags.FLAGS.local_data_dir, MANIFEST_DIR_NAME)
    ):
        print("Cannot create output manifests directory")
        return

    annotated_images = gui.show()
    save_outputs(annotated_images, previous_manifest_file, start_time, use_s3)


if __name__ == "__main__":
    app.run(main)
