import math
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from pascal_voc_writer import Writer
from PIL import Image

matplotlib.use("TKAgg")


@dataclass
class BBoxCorner:
    x: int
    y: int


@dataclass
class BBox:
    corner1: BBoxCorner
    corner2: BBoxCorner
    category: str


class AnnotatedImage:
    def __init__(self, image_path: str, annotation_base_dir: str):
        self.image_path = image_path
        self.annotation_base_dir = annotation_base_dir
        self.bboxes: List[BBox] = []
        self.valid = True

    # use the base image filename for the output annotation xml file
    def _get_pascal_voc_filename(self) -> str:
        if not self.valid:
            return "Invalid"
        else:
            return os.path.splitext(os.path.basename(self.image_path))[0] + ".xml"

    def remove_incomplete_boxes(self) -> None:
        for bbox in self.bboxes:
            if bbox.corner2 is None:
                self.bboxes.remove(bbox)

    def write_to_pascal_voc(self) -> str:
        if len(self.bboxes) == 0 or not self.valid:
            return None
        width, height = Image.open(self.image_path).size
        writer = Writer(self.image_path, width, height)
        annotation_path = os.path.join(
            self.annotation_base_dir, self._get_pascal_voc_filename()
        )
        self.remove_incomplete_boxes()
        for bbox in self.bboxes:
            writer.addObject(
                bbox.category,
                bbox.corner1.x,
                bbox.corner1.y,
                bbox.corner2.x,
                bbox.corner2.y,
            )
        writer.save(annotation_path)
        return annotation_path


class Category:
    BOX_SIDES = ["left", "right", "top", "bottom"]
    SELECTED_CATEGORY_COLOR = "#1bbf3c"
    SELECTED_CATEGORY_BORDER_WIDTH = 3

    def __init__(self, name, color, keyboard_string):
        self.name = name
        self.color = color
        self.keyboard_string = keyboard_string
        self.ax = None
        self.button = None

    def select(self):
        if self.ax is not None:
            for side in self.BOX_SIDES:
                self.ax.spines[side].set_linewidth(self.SELECTED_CATEGORY_BORDER_WIDTH)
                self.ax.spines[side].set_color(self.SELECTED_CATEGORY_COLOR)

    def deselect(self):
        if self.ax is not None:
            for side in self.BOX_SIDES:
                self.ax.spines[side].set_linewidth(None)
                self.ax.spines[side].set_color("black")


class GUI:
    INVALID_IMAGE_COLOR = "#f58d42"
    INVALID_IMAGE_BORDER_WIDTH = 5

    CATEGORY_BUTTON_BOTTOM = 0.1
    CATEGORY_BUTTON_HEIGHT = 0.075
    CATEGORY_BUTTON_WIDTH = 0.11
    CATEGORY_BUTTON_LEFT_MARGIN = 0.025
    CATEGORY_BUTTON_SPACING = 0.12

    UTILITY_BUTTON_BOTTOM = 0.01
    UTILITY_BUTTON_HEIGHT = 0.075
    UTILITY_BUTTON_WIDTH = 0.11
    UTILITY_BUTTON_RIGHT_MARGIN = 0.975
    UTILITY_BUTTON_SPACING = 0.12

    IMAGE_BOX_BOTTOM = 0.25
    IMAGE_BOX_LEFT = 0.075
    IMAGE_BOX_HEIGHT = 0.65
    IMAGE_BOX_WIDTH = 0.85

    BOX_SIDES = ["left", "right", "top", "bottom"]

    def __init__(self, fig):
        self.fig = fig
        self.categories: Dict[str, Category] = dict()
        self.current_category: str = None
        self.images: List[AnnotatedImage] = []
        self.image_index: int = 0
        self.fig.canvas.set_window_title("Label")
        self.image_ax = self.fig.add_axes(
            [
                self.IMAGE_BOX_LEFT,
                self.IMAGE_BOX_BOTTOM,
                self.IMAGE_BOX_WIDTH,
                self.IMAGE_BOX_HEIGHT,
            ]
        )
        self.corner1_vline = self.image_ax.axvline(linestyle="dashed", visible=False)
        self.corner1_hline = self.image_ax.axhline(linestyle="dashed", visible=False)
        self.corner2_vline = self.image_ax.axvline(linestyle="dashed", visible=False)
        self.corner2_hline = self.image_ax.axhline(linestyle="dashed", visible=False)

        self.display_image = None

        self.undo_ax = self.fig.add_axes(self._get_utility_ax_rect(3))
        self.invalid_ax = self.fig.add_axes(self._get_utility_ax_rect(2))
        self.prev_ax = self.fig.add_axes(self._get_utility_ax_rect(1))
        self.next_ax = self.fig.add_axes(self._get_utility_ax_rect(0))
        self.invalid_button = Button(
            self.invalid_ax, "Invalid", color=self.INVALID_IMAGE_COLOR
        )
        self.undo_button = Button(self.undo_ax, "Undo")
        self.prev_button = Button(self.prev_ax, "Prev")
        self.next_button = Button(self.next_ax, "Next")

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_keypress)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)

    def show(self) -> None:
        # Display the first image

        self.display_image = self.image_ax.imshow(
            Image.open(self.images[self.image_index].image_path)
        )

        self._update_title()
        # Select the first category as default
        self.current_category = next(iter(self.categories))
        self.categories[self.current_category].select()
        plt.show()

        print("Closed window")
        return self._get_annotated_images()

    def _get_utility_ax_rect(self, utility_index) -> List[int]:
        return [
            self.UTILITY_BUTTON_RIGHT_MARGIN
            - ((utility_index + 1) * self.UTILITY_BUTTON_SPACING),
            self.UTILITY_BUTTON_BOTTOM,
            self.UTILITY_BUTTON_WIDTH,
            self.UTILITY_BUTTON_HEIGHT,
        ]

    def _get_category_ax_rect(self, category_index: int) -> List[int]:
        return [
            (category_index * self.CATEGORY_BUTTON_SPACING)
            + self.CATEGORY_BUTTON_LEFT_MARGIN,
            self.CATEGORY_BUTTON_BOTTOM,
            self.CATEGORY_BUTTON_WIDTH,
            self.CATEGORY_BUTTON_HEIGHT,
        ]

    def add_category(self, category: Category) -> None:
        category.ax = self.fig.add_axes(
            self._get_category_ax_rect(len(self.categories))
        )
        category.button = Button(category.ax, category.name, color=category.color)
        self.categories[category.name] = category

    def add_image(self, image: AnnotatedImage) -> None:
        self.images.append(image)

    def _get_annotated_images(self) -> List[AnnotatedImage]:
        return [
            image
            for image in self.images[: self.image_index + 1]
            if len(image.bboxes) > 0 or not image.valid
        ]

    def _update_title(self) -> None:
        self.image_ax.set_title(
            "%s [%i/%i]"
            % (
                self.images[self.image_index].image_path.split("/")[-1],
                self.image_index + 1,
                len(self.images),
            )
        )

    def _reset_extent(self, data):
        ax = self.image_ax
        dataShape = data.size
        im = self.display_image

        if im.origin == "upper":
            im.set_extent((-0.5, dataShape[0] - 0.5, dataShape[1] - 0.5, -0.5))
            ax.set_xlim((-0.5, dataShape[0] - 0.5))
            ax.set_ylim((dataShape[1] - 0.5, -0.5))
        else:
            im.set_extent((-0.5, dataShape[0] - 0.5, -0.5, dataShape[1] - 0.5))
            ax.set_xlim((-0.5, dataShape[0] - 0.5))
            ax.set_ylim((-0.5, dataShape[1] - 0.5))

    def _display_image(self) -> None:
        img = Image.open(self.images[self.image_index].image_path)
        self.display_image.set_data(img)
        self._reset_extent(img)
        self._update_title()
        self._refresh()

    def _next_image(self) -> None:
        self.images[self.image_index].remove_incomplete_boxes()
        self._clear_all_lines()
        if self.image_index == len(self.images) - 1:
            plt.close()
        else:
            self.image_index += 1
            self._display_image()
            self._draw_bounding_boxes()
            self._draw_image_border()

    def _prev_image(self) -> None:
        self._clear_all_lines()
        if self.image_index != 0:
            self.images[self.image_index].remove_incomplete_boxes()
            self.image_index -= 1
            self._display_image()
            self._draw_bounding_boxes()
            self._draw_image_border()

    def _format_corners(self, bbox) -> None:
        x_min = min(bbox.corner1.x, bbox.corner2.x)
        y_min = min(bbox.corner1.y, bbox.corner2.y)
        x_max = max(bbox.corner1.x, bbox.corner2.x)
        y_max = max(bbox.corner1.y, bbox.corner2.y)
        bbox.corner1.x = x_min
        bbox.corner1.y = y_min
        bbox.corner2.x = x_max
        bbox.corner2.y = y_max

    def _clear_corner1_lines(self) -> None:
        self.corner1_hline.set_visible(False)
        self.corner1_vline.set_visible(False)

    def _clear_corner2_lines(self) -> None:
        self.corner2_hline.set_visible(False)
        self.corner2_vline.set_visible(False)

    def _clear_all_lines(self) -> None:
        self._clear_corner1_lines()
        self._clear_corner2_lines()

    def _draw_corner_1_lines(self, x: int, y: int) -> None:
        self.corner1_hline.set_ydata(y)
        self.corner1_vline.set_xdata(x)

        self.corner1_hline.set_visible(True)
        self.corner1_vline.set_visible(True)

    def _draw_corner_2_lines(self, x: int, y: int) -> None:
        self.corner2_hline.set_ydata(y)
        self.corner2_vline.set_xdata(x)

        self.corner2_hline.set_visible(True)
        self.corner2_vline.set_visible(True)

    def _draw_bounding_boxes(self) -> None:
        # clear all current boxes
        [p.remove() for p in reversed(self.image_ax.patches)]
        # redraw the boxes
        for bbox in self.images[self.image_index].bboxes:
            if bbox.corner2 is None:
                continue
            height = bbox.corner2.y - bbox.corner1.y
            width = bbox.corner2.x - bbox.corner1.x
            lower_left = (bbox.corner1.x, bbox.corner1.y)
            color = self.categories[bbox.category].color
            rect = patches.Rectangle(
                lower_left,
                width,
                height,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            self.image_ax.add_patch(rect)
        self._refresh()

    def _handle_bbox_entry(self, event) -> None:
        if not self.images[self.image_index].valid:
            print("Image marked as invalid. Cannot draw bounding box")
            return
        if (
            len(self.images[self.image_index].bboxes) > 0
            and self.images[self.image_index].bboxes[-1].corner2 is None
        ):
            self._clear_all_lines()
            self.images[self.image_index].bboxes[-1].corner2 = BBoxCorner(
                math.floor(event.xdata), math.floor(event.ydata)
            )
            self._format_corners(self.images[self.image_index].bboxes[-1])
            self._draw_bounding_boxes()
        else:
            self.images[self.image_index].bboxes.append(
                BBox(
                    BBoxCorner(math.floor(event.xdata), math.floor(event.ydata)),
                    None,
                    self.current_category,
                )
            )
            self._draw_corner_1_lines(
                self.images[self.image_index].bboxes[-1].corner1.x,
                self.images[self.image_index].bboxes[-1].corner1.y,
            )

    def _draw_invalid_image_border(self) -> None:
        for side in self.BOX_SIDES:
            self.image_ax.spines[side].set_linewidth(self.INVALID_IMAGE_BORDER_WIDTH)
            self.image_ax.spines[side].set_color(self.INVALID_IMAGE_COLOR)

    def _draw_valid_image_border(self) -> None:
        for side in self.BOX_SIDES:
            self.image_ax.spines[side].set_linewidth(None)
            self.image_ax.spines[side].set_color("black")

    def _draw_image_border(self):
        if not self.images[self.image_index].valid:
            self._draw_invalid_image_border()
        else:
            self._draw_valid_image_border()

    def _toggle_image_validation(self) -> None:
        self.images[self.image_index].valid = not self.images[self.image_index].valid
        self.images[self.image_index].bboxes.clear()
        self._clear_all_lines()
        self._draw_bounding_boxes()
        self._draw_image_border()

    def _undo_latest(self) -> None:
        self._clear_all_lines()
        if len(self.images[self.image_index].bboxes) == 0:
            print("No more bounding boxes to clear")
        elif self.images[self.image_index].bboxes[-1].corner2 is None:
            self.images[self.image_index].remove_incomplete_boxes()
        else:
            # Edit corner 2 of newest bbox
            self.images[self.image_index].bboxes[-1].corner2 = None
            self._draw_corner_1_lines(
                self.images[self.image_index].bboxes[-1].corner1.x,
                self.images[self.image_index].bboxes[-1].corner1.y,
            )
        self._draw_bounding_boxes()

    def _refresh(self) -> None:
        if plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw()

    def _on_click(self, event) -> None:
        # verify that the click was inbounds for an axes
        if event.xdata is None or event.ydata is None or event.inaxes is None:
            return
        elif event.inaxes == self.image_ax:
            self._handle_bbox_entry(event)
        elif event.inaxes == self.next_ax:
            self._next_image()
        elif event.inaxes == self.prev_ax:
            self._prev_image()
        elif event.inaxes == self.invalid_ax:
            self._toggle_image_validation()
        elif event.inaxes == self.undo_ax:
            self._undo_latest()
        else:
            for category_name, category in self.categories.items():
                if event.inaxes == category.ax:
                    self.current_category = category_name
                    [c.deselect() for _, c in self.categories.items()]
                    category.select()
                    break
        self._refresh()

    def _on_keypress(self, event) -> None:
        if event.key == "d":
            self._next_image()
        elif event.key == "a":
            self._prev_image()
        elif event.key == "w" or event.key == "escape":
            self._undo_latest()
        for category_name, category in self.categories.items():
            if event.key == category.keyboard_string:
                self.current_category = category_name
                [c.deselect() for _, c in self.categories.items()]
                category.select()
        self._refresh()

    def _on_mouse_motion(self, event) -> None:
        if event.inaxes is None or event.inaxes != self.image_ax:
            if (
                len(self.images[self.image_index].bboxes) == 0
                or not self.images[self.image_index].valid
                or self.images[self.image_index].bboxes[-1].corner2 is not None
            ):
                self._clear_corner1_lines()
            elif self.images[self.image_index].bboxes[-1].corner2 is None:
                self._clear_corner2_lines()
        else:
            if (
                len(self.images[self.image_index].bboxes) == 0
                or self.images[self.image_index].bboxes[-1].corner2 is not None
            ):
                self._draw_corner_1_lines(event.xdata, event.ydata)
            elif self.images[self.image_index].bboxes[-1].corner2 is None:
                self._draw_corner_2_lines(event.xdata, event.ydata)
        self._refresh()
