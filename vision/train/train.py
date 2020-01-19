# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import time
from typing import List
import re

from absl import app, flags
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# import transforms as T
import pascal_voc_parser

from .engine import train_one_epoch, evaluate
from .._s3_utils import (
    s3_bucket_exists,
    s3_upload_files,
    s3_get_object_names_from_dir,
    s3_download_files,
)
from .transforms import ToTensor, RandomHorizontalFlip, Compose
from .train_utils import collate_fn

IMAGE_DIR_NAME = "images"
ANNOTATION_DIR_NAME = "annotations"
MANIFEST_DIR_NAME = "manifests"
MODEL_STATE_ROOT_DIR = "modelState"

IMAGE_FILE_TYPE = "jpg"
ANNOTATION_FILE_TYPE = "xml"
MANIFEST_FILE_TYPE = "txt"
MODEL_STATE_FILE_NAME = "modelState.pt"

INVALID_ANNOTATION_FILE_IDENTIFIER = "invalid"

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


class ODDataSet(object):
    def __init__(self, data_root, transforms, labels, manifest_file_path: str):

        self.data_root = data_root
        self.transforms = transforms

        self.labels = labels
        manifest_items = [
            item.strip() for item in open(manifest_file_path).read().splitlines()
        ]
        # Filter out Invalid images
        manifest_items = [
            item
            for item in manifest_items
            if item.split(",")[1].lower() != INVALID_ANNOTATION_FILE_IDENTIFIER
        ]

        self.images = [
            os.path.join(self.data_root, IMAGE_DIR_NAME, item.split(",")[0])
            for item in manifest_items
        ]
        self.annotations = [
            os.path.join(self.data_root, ANNOTATION_DIR_NAME, item.split(",")[1])
            for item in manifest_items
        ]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.data_root, IMAGE_DIR_NAME, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        annotation_path = os.path.join(
            self.data_root, ANNOTATION_DIR_NAME, self.annotations[idx]
        )
        _, annotation_boxes = pascal_voc_parser.read_content(annotation_path)

        num_objs = len(annotation_boxes)
        boxes = [[b.xmin, b.ymin, b.xmax, b.ymax] for b in annotation_boxes]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class

        labels = [self.labels.index(b.label) for b in annotation_boxes]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])

        area = [b.get_area() for b in annotation_boxes]
        area = torch.as_tensor(area, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_model_instance_detection(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_model_instance_detection_mobilenet(num_classes):

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    return model


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
        # Download new images from s3
        s3_images = s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name,
            flags.FLAGS.s3_data_dir + "/" + IMAGE_DIR_NAME,
            IMAGE_FILE_TYPE,
        )
        s3_download_files(
            flags.FLAGS.s3_bucket_name,
            s3_images,
            os.path.join(flags.FLAGS.local_data_dir, IMAGE_DIR_NAME),
        )

        # Download any nest annotation files from s3
        s3_annotations = s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name,
            flags.FLAGS.s3_data_dir + "/" + ANNOTATION_DIR_NAME,
            ANNOTATION_FILE_TYPE,
        )

        s3_download_files(
            flags.FLAGS.s3_bucket_name,
            s3_annotations,
            os.path.join(flags.FLAGS.local_data_dir, ANNOTATION_DIR_NAME),
        )

        # Download any new manifests files from s3
        s3_manifests = s3_get_object_names_from_dir(
            flags.FLAGS.s3_bucket_name,
            flags.FLAGS.s3_data_dir + "/" + MANIFEST_DIR_NAME,
            MANIFEST_FILE_TYPE,
        )

        s3_download_files(
            flags.FLAGS.s3_bucket_name,
            s3_manifests,
            os.path.join(flags.FLAGS.local_data_dir, MANIFEST_DIR_NAME),
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
    dataset = ODDataSet(
        flags.FLAGS.local_data_dir,
        get_transform(train=True),
        labels,
        newest_manifest_file,
    )
    dataset_test = ODDataSet(
        flags.FLAGS.local_data_dir,
        get_transform(train=False),
        labels,
        newest_manifest_file,
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
    model = get_model_instance_detection_mobilenet(num_classes)

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

    model_state_dir = os.path.join(
        flags.FLAGS.local_data_dir, MODEL_STATE_ROOT_DIR, str(start_time)
    )

    # Create model state directory if it does not exist yet
    create_output_dir(model_state_dir)

    model_state_file_path = os.path.join(model_state_dir, MODEL_STATE_FILE_NAME)

    # Save the model state to a file
    torch.save(model.state_dict(), model_state_file_path)

    print("Model state saved at: %s" % model_state_file_path)

    if use_s3:
        # Send the saved model to S3
        s3_upload_files(
            flags.FLAGS.s3_bucket_name,
            [model_state_file_path],
            "/".join([flags.FLAGS.s3_data_dir, MODEL_STATE_ROOT_DIR, str(start_time)]),
        )

    print("Training complete")


if __name__ == "__main__":
    app.run(main)
