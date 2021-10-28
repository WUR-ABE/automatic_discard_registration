from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any, List, Optional, Tuple, Union


class BoundingBoxMixin(object):
    def __init__(self, *args: Any, box_x1y1wh: Optional[Union[List[int], np.ndarray]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # type: ignore

        if box_x1y1wh is not None:
            if isinstance(box_x1y1wh, list):
                box_x1y1wh = np.array(box_x1y1wh)

            # Set the property and recalculate all other properties
            self.box_x1y1wh = box_x1y1wh

    @property
    def box_x1y1wh(self) -> np.ndarray:
        return self.__box_x1y1wh

    @box_x1y1wh.setter
    def box_x1y1wh(self, value: Union[List[int], np.ndarray]) -> None:
        """
        Converts the bounding box in the form [x, y, w, h] to the form [x1, y1, x2, y2] and [x, y, s, r] where x1 and
        y1 are the top-left coordinates, x2, y2 the bottom-right coordinates, x, y the center coordinates, s the
        bounding box area ans r the ratio between the box width and length. The values are set in the box_x1y1wh,
        box_xysr and box_x1y1x2y2 properties of this base class.

        (x1,y1)           w
               x --------------------- x
               |           |           |
               |                       |       s = area = w * h
               |           | (x, y)    |       r = ratio = w / h
               | - - - - - x - - - - - | h
               |           |           |
               |                       |
               |           |           |
               x --------------------- x
                                        (x2, y2)

        """
        self.__box_x1y1wh = np.array(value).reshape((4, 1)) if isinstance(value, list) else value

        # Calculate xysr box
        x = value[0] + value[2] / 2  # tlx + w / 2
        y = value[1] + value[3] / 2  # tly + h / 2
        s = value[2] * value[3]  # scale = area
        r = value[2] / float(value[3])  # width / height
        self.__box_xysr = np.array([x, y, s, r]).reshape((4, 1))

        # Calculate x1y1x2y2 box
        x1 = value[0]  # x
        y1 = value[1]  # y
        x2 = value[0] + value[2]  # x + w
        y2 = value[1] + value[3]  # y + h
        self.__box_x1y1x2y2 = np.array([x1, y1, x2, y2]).reshape((4, 1))

        # Calculate area
        self.__area = (value[2] + 1) * (value[3] + 1)

    @box_x1y1wh.deleter
    def box_x1y1wh(self) -> None:
        # Delete all attributes since these are correlated
        del self.__box_x1y1wh, self.__box_x1y1x2y2, self.__box_xysr, self.__area

    @property
    def box_xysr(self) -> np.ndarray:
        return self.__box_xysr

    @box_xysr.setter
    def box_xysr(self, value: Union[List[int], np.ndarray]) -> None:
        """
        Converts the bounding box in the form [x, y, s, r] to the form [x1, y1, x2, y2] and [x1, y1, w, h] where x1 and
        y1 are the top-left coordinates, x2, y2 the bottom-right coordinates, x, y the center coordinates, s the
        bounding box area ans r the ratio between the box width and length. The values are set in the box_x1y1wh,
        box_xysr and box_x1y1x2y2 properties of this base class.

        (x1,y1)           w
               x --------------------- x
               |           |           |
               |                       |       s = area = w * h
               |           | (x, y)    |       r = ratio = w / h
               | - - - - - x - - - - - | h
               |           |           |
               |                       |
               |           |           |
               x --------------------- x
                                        (x2, y2)

        """
        self.__box_xysr = np.array(value).reshape((4, 1)) if isinstance(value, list) else value

        # Get width and height
        w = np.sqrt(abs(value[2] * value[3]))  # width = sqrt(scale * ratio) > area cannot be negative
        h = value[2] / w  # height = scale / width

        # Calculate x1y1x2y2
        x1 = value[0] - w / 2  # x1 = x_center - width / 2
        y1 = value[1] - h / 2  # y1 = y_center - height / 2
        x2 = x1 + w  # x2 = x1 + width
        y2 = y1 + h  # y2 = y1 + width
        self.__box_x1y1x2y2 = np.array([x1, y1, x2, y2]).reshape((4, 1))

        # Calculate x1y1wh box
        self.__box_x1y1wh = np.array([x1, y1, w, h]).reshape((4, 1))

        # Calculate area
        self.__area = (w + 1) * (h + 1)

    @box_xysr.deleter
    def box_xysr(self) -> None:
        # Delete all attributes since these are correlated
        del self.__box_x1y1wh, self.__box_x1y1x2y2, self.__box_xysr, self.__area

    @property
    def box_x1y1x2y2(self) -> np.ndarray:
        return self.__box_x1y1x2y2

    @box_x1y1x2y2.setter
    def box_x1y1x2y2(self, value: Union[List[int], np.ndarray]) -> None:
        """
        Converts the bounding box in the form [x1, y1, x2, y2] to the form [x, y, s, r] and [x1, y1, w, h] where x1 and
        y1 are the top-left coordinates, x2, y2 the bottom-right coordinates, x, y the center coordinates, s the
        bounding box area ans r the ratio between the box width and length. The values are set in the box_x1y1wh,
        box_xysr and box_x1y1x2y2 properties of this base class.

        (x1,y1)           w
               x --------------------- x
               |           |           |
               |                       |       s = area = w * h
               |           | (x, y)    |       r = ratio = w / h
               | - - - - - x - - - - - | h
               |           |           |
               |                       |
               |           |           |
               x --------------------- x
                                        (x2, y2)

        """
        self.__box_x1y1x2y2 = np.array(value).reshape((4, 1)) if isinstance(value, list) else value

        # Get width and height
        w = abs(value[2] - value[0])  # width = x2 - x1
        h = abs(value[3] - value[1])  # heigth = y2 - y1

        # Calculate x1y1wh
        self.__box_x1y1wh = np.array([value[0], value[1], w, h]).reshape((4, 1))

        # Calculate xysr
        x = value[0] + w / 2  # tlx + w / 2
        y = value[1] + h / 2  # tly + h / 2
        s = w * h  # scale = area
        r = w / float(h)  # width / height
        self.__box_xysr = np.array([x, y, s, r]).reshape((4, 1))

        # Calculate area
        self.__area = (w + 1) * (h + 1)

    @box_x1y1x2y2.deleter
    def box_x1y1x2y2(self) -> None:
        # Delete all attributes since these are correlated
        del self.__box_x1y1wh, self.__box_x1y1x2y2, self.__box_xysr, self.__area

    @property
    def area(self) -> float:
        return float(self.__area)

    def calculate_iou_with(self, other_detection: BoundingBoxMixin) -> float:
        """
        Calculates the IoU between the this detection and another one.
         _______________________________________________________
        |                                                       |
        |     _______________________                           |
        |    |(x11,y11)              |                          |
        |    |             __________|___________               |
        |    |            |(x21,y21) |           |              |
        |    |            |(xa1,ya1) |           |              |
        |    |____________|__________|(x12,y12)  |              |
        |                 |           (xa2,ya2)  |              |
        |                 |______________________|(x22,y22)     |
        |_______________________________________________________|

        We have to add 1 to the area calculation since we are working in pixel coordinates:
        https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

        :param other_detection: The other detection or tracker.
        :return: The calculated IoU.
        """
        if not self.box_intersect_with(other_detection):
            return -1

        # Intersection area
        x1_i = max(self.box_x1y1x2y2[0], other_detection.box_x1y1x2y2[0])
        y1_i = max(self.box_x1y1x2y2[1], other_detection.box_x1y1x2y2[1])
        x2_i = min(self.box_x1y1x2y2[2], other_detection.box_x1y1x2y2[2])
        y2_i = min(self.box_x1y1x2y2[3], other_detection.box_x1y1x2y2[3])
        intersection = (x2_i - x1_i + 1) * (y2_i - y1_i + 1)

        # Get union area
        union = self.area + other_detection.area - intersection

        # Calculate iou
        iou = float(intersection / union)

        return 0.0 if np.any(np.isnan(iou)) else iou

    def box_intersect_with(self, other_detection: BoundingBoxMixin) -> bool:
        # Box 1 is right of box 2
        if self.box_x1y1x2y2[0] > other_detection.box_x1y1x2y2[2]:
            return False

        # Box 1 is below box 2
        if self.box_x1y1x2y2[1] > other_detection.box_x1y1x2y2[3]:
            return False

        # Box 1 is left of  box 2
        if other_detection.box_x1y1x2y2[0] > self.box_x1y1x2y2[2]:
            return False

        # Box 1 is above box 2
        if other_detection.box_x1y1x2y2[1] > self.box_x1y1x2y2[3]:
            return False

        return True

    def in_image(self, image_size: Tuple[int, int], margin: int = 10) -> bool:
        if (
            self.box_x1y1x2y2[0] > image_size[0] - margin
            or self.box_x1y1x2y2[2] < margin
            or self.box_x1y1x2y2[1] > image_size[1] - margin
            or self.box_x1y1x2y2[3] < margin
        ):
            return False
        return True
