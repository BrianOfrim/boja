from cv2 import cv2
import numpy as np
import matplotlib
import matplotlib.patches as patches


class RGB8Image:
    BORDER_COLOR = (3, 252, 53)
    BORDER_WIDTH = 10

    def __init__(
        self, width: int, height: int, data_format: str, image_data: np.ndarray
    ):
        self.image_data: np.ndarray = self._process_image(
            image_data, data_format, width, height
        )

    def get_height(self):
        return self.image_data.shape[0]

    def get_width(self):
        return self.image_data.shape[1]

    def get_channels(self):
        if len(self.image_data.shape) < 3:
            return 1
        return self.image_data.shape[2]

    def get_data(self) -> np.ndarray:
        return self.image_data

    def _process_image(self, image_data, data_format, width, height) -> np.ndarray:
        if data_format == "Mono8":
            return cv2.cvtColor(image_data.reshape(height, width), cv2.COLOR_GRAY2RGB)
        elif data_format == "BayerRG8":
            return cv2.cvtColor(
                image_data.reshape(height, width), cv2.COLOR_BayerRG2RGB
            )
        elif data_format == "BayerGR8":
            return cv2.cvtColor(
                image_data.reshape(height, width), cv2.COLOR_BayerGR2RGB
            )
        elif data_format == "BayerGB8":
            return cv2.cvtColor(
                image_data.reshape(height, width), cv2.COLOR_BayerGB2RGB
            )
        elif data_format == "BayerBG8":
            return cv2.cvtColor(
                image_data.reshape(height, width), cv2.COLOR_BayerBG2RGB
            )
        elif data_format == "RGB8":
            return image_data.reshape(height, width, 3)
        elif data_format == "BGR8":
            return cv2.cvtColor(image_data.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
        else:
            print("Unsupported pixel format: %s" % data_format)
            raise ValueError("Unsupported pixel format: %s" % data_format)

    def get_resized_image(self, target_width: int) -> np.ndarray:
        resize_ratio = float(target_width / self.get_width())
        return cv2.resize(self.image_data, (0, 0), fx=resize_ratio, fy=resize_ratio)

    def get_highlighted_image(self, target_width: int = None) -> np.ndarray:
        return cv2.copyMakeBorder(
            (
                self.get_resized_image(target_width)
                if target_width is not None
                else self.get_data()
            )[
                self.BORDER_WIDTH : -self.BORDER_WIDTH,
                self.BORDER_WIDTH : -self.BORDER_WIDTH,
            ],
            top=self.BORDER_WIDTH,
            bottom=self.BORDER_WIDTH,
            left=self.BORDER_WIDTH,
            right=self.BORDER_WIDTH,
            borderType=cv2.BORDER_ISOLATED,
            value=self.BORDER_COLOR,
        )

    def save(self, file_path: str) -> bool:
        try:
            cv2.imwrite(file_path, self.get_data())
        except FileExistsError:
            return False
        return True

    @staticmethod
    def to_bgr(image_data: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)


def draw_bboxes(
    ax, bboxes, label_indices, label_names, label_colors, label_scores=None
):
    for box_index, (box, label_index) in enumerate(zip(bboxes, label_indices)):
        height = box[3] - box[1]
        width = box[2] - box[0]
        lower_left = (box[0], box[1])
        rect = patches.Rectangle(
            lower_left,
            width,
            height,
            linewidth=2,
            edgecolor=label_colors[label_index],
            facecolor="none",
        )
        ax.add_patch(rect)
        label_string = ""
        if label_scores is None:
            label_string = label_names[label_index]
        else:
            label_string = "%s [%.2f]" % (
                label_names[label_index],
                label_scores[box_index],
            )
        ax.text(
            box[0],
            box[1] - 10,
            label_string,
            bbox=dict(
                facecolor=label_colors[label_index],
                alpha=0.5,
                pad=1,
                edgecolor=label_colors[label_index],
            ),
            fontsize=10,
            color="white",
        )

