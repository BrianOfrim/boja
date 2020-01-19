import os
import pathlib
from typing import List

import boto3
import botocore


def s3_bucket_exists(name: str) -> bool:
    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=name)
    except botocore.exceptions.ClientError as e:
        print(e)
        return False
    return True


def file_exists(bucket_name: str, s3_object_path: str) -> None:
    s3 = boto3.resource("s3")
    try:
        s3.Object(bucket_name, s3_object_path).load()  # pylint: disable=no-member
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise
    else:
        return True


def s3_get_object_names_from_dir(
    bucket_name: str, dir_name: str, file_type: str = None
) -> List[str]:
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)  # pylint: disable=no-member
    object_names = [
        object_summary.key for object_summary in bucket.objects.filter(Prefix=dir_name)
    ]

    if file_type is not None:
        object_names = [
            object_name
            for object_name in object_names
            if object_name.lower().endswith(file_type.lower())
        ]
    return object_names


def s3_download_files(
    bucket_name: str,
    s3_object_paths: List[str],
    destination_dir: str,
    notify_if_exists: bool = False,
) -> None:
    s3_client = boto3.client("s3")
    s3_resource = boto3.resource("s3")
    object_summary_list = [
        s3_resource.ObjectSummary(  # pylint: disable=no-member
            bucket_name, s3_object_path
        )
        for s3_object_path in s3_object_paths
    ]

    if not os.path.isdir(destination_dir):
        pathlib.Path(destination_dir).mkdir(parents=True, exist_ok=True)

    for object_index, object_summary in enumerate(object_summary_list):
        destination_file_path = os.path.join(
            destination_dir, os.path.basename(object_summary.key)
        )
        if not os.path.isfile(destination_file_path):
            print(
                "Downloading file from %s:%s, %i/%i"
                % (
                    object_summary.bucket_name,
                    object_summary.key,
                    object_index + 1,
                    len(object_summary_list),
                )
            )
            try:
                s3_client.download_file(  # pylint: disable=no-member
                    object_summary.bucket_name,
                    object_summary.key,
                    destination_file_path,
                )
            except botocore.exceptions.ClientError as e:
                print(e)
       else:
            if notify_if_exists:
                print(
                    "File already downloaded: %s:%s, %i/%i"
                    % (
                        object_summary.bucket_name,
                        object_summary.key,
                        object_index + 1,
                        len(object_summary_list),
                    )
                )


def s3_upload_files(
    bucket_name,
    files_to_send: List[str],
    s3_destination_object_dir: str,
    notify_if_exists: bool = False,
) -> None:
    s3 = boto3.client("s3")
    for file_index, file_to_send in enumerate(files_to_send):
        s3_destination_object_path = "/".join(
            [s3_destination_object_dir, os.path.basename(file_to_send)]
        )
        try:
            if file_exists(bucket_name, s3_destination_object_path):
                if notify_if_exists:
                    print(
                        "S3 object already exists %s:%s, %i/%i"
                        % (
                            bucket_name,
                            s3_destination_object_dir,
                            file_index + 1,
                            len(files_to_send),
                        )
                    )
                continue
            s3.upload_file(file_to_send, bucket_name, s3_destination_object_path)
        except botocore.exceptions.ClientError as e:
            print(e)
            continue
        print(
            "Uploading file to %s:%s, %i/%i"
            % (
                bucket_name,
                s3_destination_object_path,
                file_index + 1,
                len(files_to_send),
            )
        )
