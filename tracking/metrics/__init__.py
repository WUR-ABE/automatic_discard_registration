from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union

    from common.detection import Detection
    from tracking.tracker import Tracker


class MetricBase(metaclass=ABCMeta):
    """
    Base class to calculate the metric of the match between the detection and the tracker.
    """

    maximize: bool = False
    threshold: Union[int, float] = 0

    @staticmethod
    @abstractmethod
    def calculate(detection: Detection, tracker: Tracker) -> Union[int, float]:
        """
        Calculate the metric between the detection and the tracker.

        :param detection: The detection.
        :param tracker: The tracker.
        :return: The metric score, the lower, the better.
        """
        pass

    @staticmethod
    @abstractmethod
    def is_valid(value: Union[int, float]) -> bool:
        """
        Checks whether the calculated value is valid or not.

        :param value: The calculated metric value.
        :return: True if the value is valid, False otherwise.
        """
        pass
