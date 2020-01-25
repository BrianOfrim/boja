from dataclasses import dataclass
import os
import re
from typing import List

from absl import app, flags

from ._file_utils import create_output_dir, get_files_from_dir
from ._s3_utils import (
    s3_bucket_exists,
    s3_download_files,
    s3_get_object_names_from_dir,
    s3_upload_files,
    s3_file_exists,
)
from ._settings import (
    IMAGE_DIR_NAME,
    ANNOTATION_DIR_NAME,
    MANIFEST_DIR_NAME,
    MODEL_STATE_DIR_NAME,
    LOGS_DIR_NAME,
    IMAGE_FILE_TYPE,
    ANNOTATION_FILE_TYPE,
    MANIFEST_FILE_TYPE,
    MODEL_STATE_FILE_TYPE,
    LABEL_FILE_NAME,
)

flags.DEFINE_string(
    "local_data_dir",
    os.path.join(os.path.expanduser("~"), "boja", "data"),
    "Local data directory.",
)

flags.DEFINE_string(
    "s3_bucket_name", None, "S3 bucket to retrieve images from and upload manifest to."
)

flags.DEFINE_string("s3_data_dir", "data", "Prefix of the s3 data objects.")


@dataclass
class DataSubDir:
    name: str
    file_type: str
    max_to_download: int


DATA_SUB_DIRS = [
    DataSubDir(IMAGE_DIR_NAME, IMAGE_FILE_TYPE, None),
    DataSubDir(ANNOTATION_DIR_NAME, ANNOTATION_FILE_TYPE, None),
    DataSubDir(MANIFEST_DIR_NAME, MANIFEST_FILE_TYPE, None),
    DataSubDir(MODEL_STATE_DIR_NAME, MODEL_STATE_FILE_TYPE, 5),
    DataSubDir(LOGS_DIR_NAME, None, None),
]


def int_string_sort(file_name) -> int:
    match = re.match("[0-9]+", os.path.basename(file_name))
    if not match:
        return 0
    return int(match[0])


def sync_s3_and_local_dir(s3_bucket_name, s3_dir, local_dir, file_type, n_newest=None):
    # get a set of data files in s3 dir
    s3_files = s3_get_object_names_from_dir(s3_bucket_name, s3_dir, file_type)

    s3_files = {os.path.basename(s3_file) for s3_file in s3_files}

    # get a set of data files from local data dir
    local_files = get_files_from_dir(local_dir, file_type)

    local_files = {os.path.basename(local_file) for local_file in local_files}

    # Determine which files to download and upload
    files_to_download = s3_files.difference(local_files)
    files_to_upload = local_files.difference(s3_files)

    if n_newest is not None:
        # only download the numerically newest
        files_to_download = sorted(files_to_download, key=int_string_sort, reverse=True)
        files_to_download = files_to_download[: min(len(files_to_download), n_newest)]

    s3_download_files(
        s3_bucket_name,
        ["/".join([s3_dir, file_name]) for file_name in files_to_download],
        local_dir,
    )
    s3_upload_files(
        s3_bucket_name,
        [os.path.join(local_dir, file_name) for file_name in files_to_upload],
        s3_dir,
    )


def main(unused_argv):
    # create local data directory if it does not exits
    if not create_output_dir(flags.FLAGS.local_data_dir):
        print("Error creating local data directory %s" % flags.FLAGS.local_data_dir)
        return
    # create local data sub directories
    for data_sub_dir in DATA_SUB_DIRS:
        if not create_output_dir(
            os.path.join(flags.FLAGS.local_data_dir, data_sub_dir.name)
        ):
            print(
                "Error creating local data directory %s"
                % os.path.join(flags.FLAGS.local_data_dir, data_sub_dir.name)
            )
            return

    use_s3 = True if flags.FLAGS.s3_bucket_name is not None else False

    if not use_s3:
        print("Local data directories created")
        return

    if not s3_bucket_exists(flags.FLAGS.s3_bucket_name):
        print(
            "Bucket: %s either does not exist or you do not have access to it"
            % flags.FLAGS.s3_bucket_name
        )
        return

    print("Bucket: %s exists and you have access to it" % flags.FLAGS.s3_bucket_name)

    local_label_file_exists = os.path.isfile(
        os.path.join(flags.FLAGS.local_data_dir, LABEL_FILE_NAME)
    )

    s3_label_file_exists = s3_file_exists(
        flags.FLAGS.s3_bucket_name, "/".join([flags.FLAGS.s3_data_dir, LABEL_FILE_NAME])
    )

    if local_label_file_exists and not s3_label_file_exists:
        s3_upload_files(
            flags.FLAGS.s3_bucket_name,
            [os.path.join(flags.FLAGS.local_data_dir, LABEL_FILE_NAME)],
            flags.FLAGS.s3_data_dir,
        )
    elif s3_label_file_exists and not local_label_file_exists:
        s3_download_files(
            flags.FLAGS.s3_bucket_name,
            ["/".join([flags.FLAGS.s3_data_dir, LABEL_FILE_NAME])],
            flags.FLAGS.local_data_dir,
        )

    for data_sub_dir in DATA_SUB_DIRS:
        print(
            "Syncing files of type: %s from folder: %s between s3 and local"
            % (
                "all" if data_sub_dir.file_type is None else data_sub_dir.file_type,
                data_sub_dir.name,
            )
        )
        sync_s3_and_local_dir(
            flags.FLAGS.s3_bucket_name,
            "/".join([flags.FLAGS.s3_data_dir, data_sub_dir.name]),
            os.path.join(flags.FLAGS.local_data_dir, data_sub_dir.name),
            data_sub_dir.file_type,
            data_sub_dir.max_to_download,
        )


if __name__ == "__main__":
    app.run(main)
