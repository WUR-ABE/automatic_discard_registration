from __future__ import annotations

from typing import TYPE_CHECKING

from tracking.metrics import MetricBase

if TYPE_CHECKING:
    from typing import Union

    from common.detection import Detection
    from tracking.tracker import Tracker


class HorizontalDisplacementMetric(MetricBase):
    threshold = 500

    @staticmethod
    def calculate(detection: Detection, tracker: Tracker) -> int:
        """
        Calculates the horizontal displacement between the detection and the tracker.

        :param detection: The detection.
        :param tracker: The tracker.
        :return: The horizontal displacement.
        """
        return int(abs(detection.box_xysr[0] - tracker.box_xysr[0]))

    @staticmethod
    def is_valid(value: Union[int, float]) -> bool:
        return value < HorizontalDisplacementMetric.threshold


class WeightedHorizontalDisplacementMetric(MetricBase):
    threshold = 500

    @staticmethod
    def calculate(detection: Detection, tracker: Tracker, wv: float = 3) -> float:
        """
        Distance measure between the detection and the tracker:

        weighted_displacement = horizontal_displacement + wv * vertical_displacement

        :param detection: The detection.
        :param tracker: The tracker.
        :param wv: The weight for vertical displacement.
        :return: The weighted horizontal displacement between the tracker and detection.
        """
        return float(
            abs(detection.box_xysr[0] - tracker.box_xysr[0]) + wv * abs(detection.box_xysr[1] - tracker.box_xysr[1])
        )

    @staticmethod
    def is_valid(value: Union[int, float]) -> bool:
        return value < WeightedHorizontalDisplacementMetric.threshold
