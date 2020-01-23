# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import time
from typing import List
import re

from absl import app, flags
import torch

from .datasets import BojaDataSet
from .engine import train_one_epoch, evaluate
from .._models import (
    get_fasterrcnn_resnet50,
    get_fasterrcnn_mobilenet_v2,
    get_fasterrcnn_resnet34,
)
from .._s3_utils import (
    s3_bucket_exists,
    s3_upload_files,
    s3_download_dir,
)
from .transforms import ToTensor, RandomHorizontalFlip, Compose
from .train_utils import collate_fn
from .._settings import (
    IMAGE_DIR_NAME,
    ANNOTATION_DIR_NAME,
    MANIFEST_DIR_NAME,
    MODEL_STATE_DIR_NAME,
    IMAGE_FILE_TYPE,
    ANNOTATION_FILE_TYPE,
    MANIFEST_FILE_TYPE,
    MODEL_STATE_FILE_TYPE,
    LABEL_FILE_NAME,
    INVALID_ANNOTATION_FILE_IDENTIFIER,
)

flags.DEFINE_string(
    "local_data_dir",
    os.path.join(os.path.expanduser("~"), "boja", "data"),
    "Local data directory.",
)

flags.DEFINE_string(
    "label_file_path",
    os.path.join(os.path.expanduser("~"), "boja", "data", "labels.txt"),
    "Path to the file containing the category labels.",
)

flags.DEFINE_string(
    "s3_bucket_name", None, "S3 bucket to retrieve images from and upload manifest to."
)

flags.DEFINE_string("s3_data_dir", "data", "Prefix of the s3 data objects.")

# Hyperparameters

flags.DEFINE_integer("num_epochs", 10, "The number of epochs to train the model for.")


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def get_files_from_dir(dir_path: str, file_type: str = None) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    file_paths = [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]
    if file_type is not None:
        file_paths = [f for f in file_paths if f.lower().endswith(file_type.lower())]
    return file_paths


def manifest_file_sort(manifest_file) -> int:
    match = re.match("[0-9]+", manifest_file)
    if not match:
        return 0
    return int(match[0])


def get_newest_manifest_path(manifest_dir_path: str) -> str:
    manifest_files = get_files_from_dir(manifest_dir_path)
    manifest_files = [
        f for f in manifest_files if f.lower().endswith(MANIFEST_FILE_TYPE)
    ]
    if len(manifest_files) == 0:
        return None
    newest_manifest_file = sorted(manifest_files, key=manifest_file_sort, reverse=True)[
        0
    ]
    return os.path.join(
        flags.FLAGS.local_data_dir, MANIFEST_DIR_NAME, newest_manifest_file
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
        return True


def main(unused_argv):

    start_time = int(time.time())

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
        # Download any new images from s3
        s3_download_dir(
            flags.FLAGS.s3_bucket_name,
            "/".join([flags.FLAGS.s3_data_dir, IMAGE_DIR_NAME]),
            os.path.join(flags.FLAGS.local_data_dir, IMAGE_DIR_NAME),
            IMAGE_FILE_TYPE,
        )

        # Download any new annotation files from s3
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

    if not os.path.isfile(flags.FLAGS.label_file_path):
        print("Invalid category labels path.")
        return

    labels = [
        label.strip() for label in open(flags.FLAGS.label_file_path).read().splitlines()
    ]

    if len(labels) == 0:
        print("No labels are present in %s" % flags.FLAGS.label_file_path)
        return

    # add the background class
    labels.insert(0, "background")

    newest_manifest_file = get_newest_manifest_path(
        os.path.join(flags.FLAGS.local_data_dir, MANIFEST_DIR_NAME)
    )

    if newest_manifest_file is None:
        print(
            "Cannot find a manifest file in: %s"
            % (os.path.join(flags.FLAGS.local_data_dir, MANIFEST_DIR_NAME))
        )

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Using device: ", device)

    num_classes = len(labels)
    # use our dataset and defined transformations

    dataset = BojaDataSet(
        os.path.join(flags.FLAGS.local_data_dir, IMAGE_DIR_NAME),
        os.path.join(flags.FLAGS.local_data_dir, ANNOTATION_DIR_NAME),
        newest_manifest_file,
        get_transform(train=True),
        labels,
    )

    dataset_test = BojaDataSet(
        os.path.join(flags.FLAGS.local_data_dir, IMAGE_DIR_NAME),
        os.path.join(flags.FLAGS.local_data_dir, ANNOTATION_DIR_NAME),
        newest_manifest_file,
        get_transform(train=False),
        labels,
    )

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()

    # use 20 percent of the dataset for testing
    num_test = int(0.2 * len(dataset))

    dataset = torch.utils.data.Subset(dataset, indices[: -1 * num_test])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-1 * num_test :])

    print(
        "Training dataset size: %d, Testing dataset size: %d"
        % (len(dataset), len(dataset_test))
    )

    # define training and validation data loaders
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    # )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn,
    )

    # get the model using our helper function
    model = get_fasterrcnn_resnet34(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = flags.FLAGS.num_epochs

    print("Training for %d epochs" % num_epochs)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    model_state_local_dir = os.path.join(
        flags.FLAGS.local_data_dir, MODEL_STATE_DIR_NAME
    )
    # Create model state directory if it does not exist yet
    create_output_dir(model_state_local_dir)

    model_state_file_path = os.path.join(
        model_state_local_dir, "%s.%s" % (str(start_time), MODEL_STATE_FILE_TYPE)
    )

    # Save the model state to a file
    torch.save(model.state_dict(), model_state_file_path)

    print("Model state saved at: %s" % model_state_file_path)

    if use_s3:
        # Send the saved model to S3
        s3_upload_files(
            flags.FLAGS.s3_bucket_name,
            [model_state_file_path],
            "/".join([flags.FLAGS.s3_data_dir, MODEL_STATE_DIR_NAME]),
        )

    print("Training complete")


if __name__ == "__main__":
    app.run(main)
