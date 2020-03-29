# Based on sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import time
import sys
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
from .transforms import ToTensor, RandomHorizontalFlip, Compose
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


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def get_newest_manifest_path(manifest_dir_path: str) -> str:
    print("Searching for manifest in: %s " % manifest_dir_path)
    return get_highest_numbered_file(manifest_dir_path, MANIFEST_FILE_TYPE)


def get_datasets(labels, args):
    newest_manifest_file = get_newest_manifest_path(
        os.path.join(args.local_data_dir, MANIFEST_DIR_NAME)
    )

    if newest_manifest_file is None:
        raise FileNotFoundError(
            "Cannot find a manifest file in: %s"
            % (os.path.join(args.local_data_dir, MANIFEST_DIR_NAME))
        )

    # use our dataset and defined transformations
    dataset = BojaDataSet(
        os.path.join(args.local_data_dir, IMAGE_DIR_NAME),
        os.path.join(args.local_data_dir, ANNOTATION_DIR_NAME),
        newest_manifest_file,
        labels,
        training=True,
    )

    dataset_test = BojaDataSet(
        os.path.join(args.local_data_dir, IMAGE_DIR_NAME),
        os.path.join(args.local_data_dir, ANNOTATION_DIR_NAME),
        newest_manifest_file,
        labels,
        training=False,
    )

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    # use 20 percent of the dataset for testing
    num_test = int(0.2 * len(dataset))

    dataset = torch.utils.data.Subset(dataset, indices[: -1 * num_test])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-1 * num_test :])

    return dataset, dataset_test


def get_labels(label_file_path):
    if not os.path.isfile(label_file_path):
        raise FileNotFoundError("Missing file %s" % label_file_path)

    # read in the category labels
    labels = open(label_file_path).read().splitlines()

    if len(labels) == 0:
        raise ValueError("No label categories found in %s" % label_file_path)

    # add the background class
    labels.insert(0, "background")

    return labels


def train_model(
    model,
    dataset,
    dataset_test,
    optimizer,
    lr_scheduler=None,
    num_epochs=10,
    batch_size=1,
    num_workers=1,
):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Using device: ", device)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), " GPUs")
        model = torch.nn.DataParallel(model)
    # move model to the right device
    model.to(device)

    print("Training for %d epochs" % num_epochs)

    evaluation_metrics = {
        AVERAGE_PRECISION_STAT_LABEL: [],
        AVERAGE_RECALL_STAT_LABEL: [],
    }

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        if lr_scheduler is not None:
            # update the learning rate
            lr_scheduler.step()
        # evaluate on the test dataset
        eval_data = evaluate(model, data_loader_test, device=device)

        stats = eval_data.coco_eval["bbox"].stats
        print("Epoch: %d, AP: %f" % (epoch, stats[AVERAGE_PRECISION_STAT_INDEX]))
        print("Epoch: %d, AR: %f" % (epoch, stats[AVERAGE_RECALL_STAT_INDEX]))
        evaluation_metrics[AVERAGE_PRECISION_STAT_LABEL].append(
            stats[AVERAGE_PRECISION_STAT_INDEX]
        )
        evaluation_metrics[AVERAGE_RECALL_STAT_LABEL].append(
            stats[AVERAGE_RECALL_STAT_INDEX]
        )

    state = {"model": model.state_dict()}

    return state, evaluation_metrics


def plot_metrics(run_name, metrics):
    plt.plot(metrics[AVERAGE_PRECISION_STAT_LABEL], label=AVERAGE_PRECISION_STAT_LABEL)
    plt.plot(metrics[AVERAGE_RECALL_STAT_LABEL], label=AVERAGE_RECALL_STAT_LABEL)

    plt.legend(loc="lower right")
    plt.title("Evaluation data from %s" % run_name)
    plt.xlabel("Epoch")

    # Create log file directory if it does not exist yet
    log_image_local_dir = os.path.join(args.local_data_dir, LOGS_DIR_NAME)
    create_output_dir(log_image_local_dir)

    log_file_name = "%s.jpg" % run_name
    log_file_path = os.path.join(log_image_local_dir, log_file_name)
    plt.savefig(log_file_path)

    print("Log file saved at: %s" % log_file_path)
    return log_file_path


def sync_s3(args):
    # Download any new images from s3
    s3_download_dir(
        args.s3_bucket_name,
        "/".join([args.s3_data_dir, IMAGE_DIR_NAME]),
        os.path.join(args.local_data_dir, IMAGE_DIR_NAME),
        IMAGE_FILE_TYPE,
    )

    # Download any new annotation files from s3
    s3_download_dir(
        args.s3_bucket_name,
        "/".join([args.s3_data_dir, ANNOTATION_DIR_NAME]),
        os.path.join(args.local_data_dir, ANNOTATION_DIR_NAME),
        ANNOTATION_FILE_TYPE,
    )

    # Download any new manifests files from s3
    s3_download_dir(
        args.s3_bucket_name,
        "/".join([args.s3_data_dir, MANIFEST_DIR_NAME]),
        os.path.join(args.local_data_dir, MANIFEST_DIR_NAME),
        MANIFEST_FILE_TYPE,
    )


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
        sync_s3(args)

    label_file_path = os.path.join(args.local_data_dir, LABEL_FILE_NAME)

    labels = get_labels(label_file_path)

    dataset, dataset_test = get_datasets(labels, args)

    print(
        "Training dataset size: %d, Testing dataset size: %d"
        % (len(dataset), len(dataset_test))
    )

    num_classes = len(labels)

    # get the model using our helper function
    model = _models.__dict__[args.network](num_classes)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    lr_scheduler = None
    batch_size = 1 if torch.cuda.device_count() == 0 else 2 * torch.cuda.device_count()

    try:
        model_state, metrics = train_model(
            model,
            dataset,
            dataset_test,
            optimizer,
            lr_scheduler,
            num_epochs=args.num_epochs,
            batch_size=batch_size,
            num_workers=args.num_data_workers,
        )
    except RuntimeError as err:
        print("Error: %s" % err)
        sys.exit(1)

    model_state_local_dir = os.path.join(args.local_data_dir, MODEL_STATE_DIR_NAME)
    # Create model state directory if it does not exist yet
    create_output_dir(model_state_local_dir)
    run_name = "%s-%s" % (str(start_time), args.network)

    model_state_file_path = os.path.join(
        model_state_local_dir, "%s.%s" % (run_name, MODEL_STATE_FILE_TYPE),
    )

    # Save the model state to a file
    torch.save(model_state, model_state_file_path)

    print("Model state saved at: %s" % model_state_file_path)

    log_file_path = plot_metrics(run_name, metrics)

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

    parser.add_argument("--num_data_workers", type=int, default=1)

    parser.add_argument(
        "--num_epochs", type=int, default=10,
    )

    args = parser.parse_args()

    main(args)
