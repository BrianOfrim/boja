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

    session_name = "%s-%s" % (start_time, args.network)

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

    optimizer_choices = _hparams.RandomHPChoices(
        [
            _hparams.Optimizer(
                name="SGD",
                options={
                    "lr": _hparams.RandomUniform(min_val=0.001, max_val=0.01),
                    "momentum": _hparams.RandomNormal(
                        mean=0.5, std=0.25, min_val=0.0, max_val=1.0
                    ),
                    "weight_decay": _hparams.RandomUniform(
                        min_val=0.0001, max_val=0.001
                    ),
                },
            ),
            _hparams.Optimizer(
                name="Adam",
                options={"lr": _hparams.RandomUniform(min_val=0.001, max_val=0.01),},
            ),
        ]
    )

    lr_scheduler_choices = _hparams.RandomHPChoices(
        [
            _hparams.LRScheduler(
                name="StepLR",
                options={
                    "step_size": _hparams.RandomUniform(1, 4),
                    "gamma": _hparams.RandomUniform(0.05, 0.3),
                },
            )
        ]
    )

    log_file = open(
        os.path.join(
            args.local_data_dir, LOGS_DIR_NAME, "%s-hp_search_log.txt" % session_name
        ),
        "w",
    )

    stat_totals = {AVERAGE_PRECISION_STAT_LABEL: {}, AVERAGE_RECALL_STAT_LABEL: {}}

    for i in range(args.num_trials):

        run_time = int(time.time())

        log_file.write("[%i]Run:%d, start time:%d\n" % (i, i, run_time))
        # get the model using our helper function
        model = _models.__dict__[args.network](num_classes)
        log_file.write("[%i]Model: %s\n" % (i, args.network))

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]

        optimizer_choice = optimizer_choices.get_next()
        optimizer_choice.set_params(params)
        optimizer = optimizer_choice.get_next()
        log_file.write("[%i]Optimizer choice: %s\n" % (i, optimizer_choice.name))
        log_file.write("[%i]Optimizer defaults: %s\n" % (i, optimizer.defaults))

        lr_scheduler_choice = lr_scheduler_choices.get_next()
        lr_scheduler_choice.set_optimizer(optimizer)
        lr_scheduler = lr_scheduler_choice.get_next()
        log_file.write("[%i]LR Scheduler choice: %s\n" % (i, lr_scheduler_choice.name))
        # log_file.write("[%i]LR Scheduler defaults: %s\n" % (i, lr_scheduler.defaults))

        model_state, metrics = train.train_model(
            model,
            dataset,
            dataset_test,
            lr_scheduler,
            optimizer,
            num_epochs=args.num_epochs,
        )

        log_file.write(
            "[%i]%s: %s\n"
            % (i, AVERAGE_PRECISION_STAT_LABEL, metrics[AVERAGE_PRECISION_STAT_LABEL])
        )
        log_file.write(
            "[%i]%s: %s\n"
            % (i, AVERAGE_RECALL_STAT_LABEL, metrics[AVERAGE_RECALL_STAT_LABEL])
        )

        stat_totals[AVERAGE_PRECISION_STAT_LABEL][i] = metrics[
            AVERAGE_PRECISION_STAT_LABEL
        ]
        stat_totals[AVERAGE_RECALL_STAT_LABEL][i] = metrics[AVERAGE_RECALL_STAT_LABEL]

    print("Training complete")

    log_image_local_dir = os.path.join(args.local_data_dir, LOGS_DIR_NAME)
    create_output_dir(log_image_local_dir)

    # create the AP plot
    for k, v in stat_totals[AVERAGE_PRECISION_STAT_LABEL].items():
        plt.plot(v, label=str(k))

    plt.legend(loc="lower right")
    plt.title("%s from session %s" % (AVERAGE_PRECISION_STAT_LABEL, session_name))
    plt.xlabel("Epoch")

    log_file_path_AP = os.path.join(
        log_image_local_dir, "%s-AP.jpg" % str(session_name)
    )
    plt.savefig(log_file_path_AP)

    plt.clf()

    # create the AR plot
    for k, v in stat_totals[AVERAGE_RECALL_STAT_LABEL].items():
        plt.plot(v, label=str(k))

    plt.legend(loc="lower right")
    plt.title("%s from session %s" % (AVERAGE_RECALL_STAT_LABEL, session_name))
    plt.xlabel("Epoch")

    log_file_path_AR = os.path.join(
        log_image_local_dir, "%s-AR.jpg" % str(session_name)
    )
    plt.savefig(log_file_path)

    log_file.close()
    plt.close()

    if use_s3:
        s3_upload_files(
            args.s3_bucket_name,
            [log_file_path_AP.log_file_path_AR],
            "/".join([args.s3_data_dir, LOGS_DIR_NAME]),
        )


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
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of random trials to run in search.",
    )

    args = parser.parse_args()

    main(args)

