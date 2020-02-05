# Based on sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import time
from typing import List
import re

import matplotlib
import matplotlib.pyplot as plt
import torch

from .datasets import BojaDataSet
from .engine import train_one_epoch, evaluate
from .._file_utils import create_output_dir, get_highest_numbered_file
from . import _hparams
from .. import _models
from .._s3_utils import (
    s3_bucket_exists,
    s3_upload_files,
    s3_download_dir,
)
from . import train
from .train_utils import collate_fn
from .._settings import (
    DEFAULT_LOCAL_DATA_DIR,
    DEFAULT_S3_DATA_DIR,
    IMAGE_DIR_NAME,
    ANNOTATION_DIR_NAME,
    MANIFEST_DIR_NAME,
    MODEL_STATE_DIR_NAME,
    IMAGE_FILE_TYPE,
    ANNOTATION_FILE_TYPE,
    MANIFEST_FILE_TYPE,
    MODEL_STATE_FILE_TYPE,
    LABEL_FILE_NAME,
    LOGS_DIR_NAME,
    INVALID_ANNOTATION_FILE_IDENTIFIER,
    NETWORKS,
    AVERAGE_PRECISION_STAT_INDEX,
    AVERAGE_PRECISION_STAT_LABEL,
    AVERAGE_RECALL_STAT_INDEX,
    AVERAGE_RECALL_STAT_LABEL,
)

matplotlib.use("Agg")


def get_newest_manifest_path(manifest_dir_path: str) -> str:
    return get_highest_numbered_file(manifest_dir_path, MANIFEST_FILE_TYPE)


def main(args):

    start_time = int(time.time())

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
        train.sync_s3(args)

    label_file_path = os.path.join(args.local_data_dir, LABEL_FILE_NAME)

    labels = train.get_labels(label_file_path)

    dataset, dataset_test = train.get_datasets(labels, args)

    print(
        "Training dataset size: %d, Testing dataset size: %d"
        % (len(dataset), len(dataset_test))
    )

    num_classes = len(labels)

    hparam = {
        "optimizer": {
            "name": "SGD",
            "args": {"lr": 0.005, "momentum": 0.9, "weight_decay": 0.0005},
        },
        "lr_scheduler": {"name": "StepLR", "args": {"step_size": 3, "gamma": 0.1}},
    }

    # get the model using our helper function
    model = _models.__dict__[args.network](num_classes)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.__dict__[hparam["optimizer"]["name"]](
        params, **hparam["optimizer"]["args"]
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.__dict__[hparam["lr_scheduler"]["name"]](
        optimizer, **hparam["lr_scheduler"]["args"]
    )

    model_state, metrics = train.train_model(
        model,
        dataset,
        dataset_test,
        lr_scheduler,
        optimizer,
        num_epochs=args.num_epochs,
    )

    model_state_local_dir = os.path.join(args.local_data_dir, MODEL_STATE_DIR_NAME)
    # Create model state directory if it does not exist yet
    create_output_dir(model_state_local_dir)
    run_name = "%s-%s" % (str(start_time), args.network)

    model_state_file_path = os.path.join(
        model_state_local_dir, "%s.%s" % (run_name, MODEL_STATE_FILE_TYPE),
    )

    # Save the model state to a file
    torch.save(model_state, model_state_file_path)

    log_file_path = train.plot_metrics(run_name, metrics)

    print("Model state saved at: %s" % model_state_file_path)

    if use_s3:
        # Send the saved model and logs to S3
        s3_upload_files(
            args.s3_bucket_name,
            [model_state_file_path],
            "/".join([args.s3_data_dir, MODEL_STATE_DIR_NAME]),
        )
        s3_upload_files(
            args.s3_bucket_name,
            [log_file_path],
            "/".join([args.s3_data_dir, LOGS_DIR_NAME]),
        )
    print("Training complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_data_dir",
        type=str,
        default=DEFAULT_LOCAL_DATA_DIR,
        help="Local data directory.",
    )
    parser.add_argument(
        "--s3_bucket_name", type=str,
    )
    parser.add_argument(
        "--s3_data_dir",
        type=str,
        default=DEFAULT_S3_DATA_DIR,
        help="Prefix of the s3 data objects.",
    )
    parser.add_argument(
        "--network",
        type=str,
        choices=NETWORKS,
        default=NETWORKS[0],
        help="The neural network to use for object detection",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10,
    )

    args = parser.parse_args()

    main(args)

