from __future__ import annotations

from typing import TYPE_CHECKING

from tracking.metrics import MetricBase

if TYPE_CHECKING:
    from typing import Union

    from common.detection import Detection
    from tracking.tracker import Tracker


class IoUMetric(MetricBase):
    maximize = True
    threshold = 0.5

    @staticmethod
    def calculate(detection: Detection, tracker: Tracker) -> float:
        """
        Calculates the IoU between the detection bounding box and the tracker bounding box.

        :param detection: The detection.
        :param tracker: The tracker.
        :return: The IoU.
        """
        return detection.calculate_iou_with(tracker)

    @staticmethod
    def is_valid(value: Union[int, float]) -> bool:
        return value > IoUMetric.threshold
