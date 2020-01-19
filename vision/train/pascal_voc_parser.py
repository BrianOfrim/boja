from dataclasses import dataclass
import os
import sys
from typing import List
import xml.etree.ElementTree as ET

ANNOTATION_FILE_TYPE = "xml"


@dataclass
class BBox:
    def __init__(self, label: str, xmin: int, ymin: int, xmax: int, ymax: int):
        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def get_area(self) -> int:
        return abs(self.xmax - self.xmin) * abs(self.ymin - self.ymax)


def read_content(xml_file: str) -> (str, List[BBox]):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_filename = root.find("filename").text
    bboxes: List[BBox] = []

    for obj in root.iter("object"):

        bndbox = obj.find("bndbox")
        bboxes.append(
            BBox(
                obj.find("name").text,
                xmin=int(bndbox.find("xmin").text),
                ymin=int(bndbox.find("ymin").text),
                xmax=int(bndbox.find("xmax").text),
                ymax=int(bndbox.find("ymax").text),
            )
        )
    return image_filename, bboxes


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No annotation file specified.")
    elif not os.path.isfile(
        str(sys.argv[1]) or not str(sys.argv[1]).lower().endswith(ANNOTATION_FILE_TYPE)
    ):
        print("Invalid annotation file specified.")
    else:
        image_file_name, bboxes = read_content(str(sys.argv[1]))
        print(
            "Annotation file: %s, Image file: %s" % (str(sys.argv[1]), image_file_name)
        )
        print("Bounding boxes:")
        for bbox in bboxes:
            print(bbox.label)
            print(
                "\txmin: %d, ymin: %d, xmax: %d, ymax: %d, Area: %d"
                % (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.get_area())
            )

