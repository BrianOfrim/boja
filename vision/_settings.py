import os

# Constants that are shared throughout the package
DEFAULT_LOCAL_DATA_DIR = os.path.join(os.path.expanduser("~"), "boja", "data")
DEFAULT_S3_DATA_DIR = "data"

IMAGE_DIR_NAME = "images"
ANNOTATION_DIR_NAME = "annotations"
MANIFEST_DIR_NAME = "manifests"
MODEL_STATE_DIR_NAME = "modelstates"
LOGS_DIR_NAME = "logs"

IMAGE_FILE_TYPE = "jpg"
ANNOTATION_FILE_TYPE = "xml"
MANIFEST_FILE_TYPE = "txt"
MODEL_STATE_FILE_TYPE = "pt"

LABEL_FILE_NAME = "labels.txt"

INVALID_ANNOTATION_FILE_IDENTIFIER = "invalid"

NETWORKS = [
    "fasterrcnn_resnet50",
    "fasterrcnn_resnet34",
    "fasterrcnn_mobilenetv2",
]

DEFAULT_GENTL_PRODUCER_PATH = "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"
