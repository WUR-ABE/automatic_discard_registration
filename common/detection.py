from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from common.bounding_box import BoundingBoxMixin

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Union


class Detection(BoundingBoxMixin):
    def __init__(
        self,
        image_name: str,
        class_probabilities: Dict[str, float],
        box_x1y1wh: Union[List[int], np.ndarray],
        objectness: Optional[float] = None,
        detection_id: Optional[str] = None,
        visibility: Optional[str] = None,
        orientation: Optional[str] = None,
        create_id_if_not_specified: bool = False,
    ) -> None:
        # Required parameters
        super().__init__(box_x1y1wh=box_x1y1wh)
        self.image_name = image_name
        self.class_probabilities = class_probabilities

        # Optional parameters
        self.objectness = objectness
        self.detection_id = detection_id
        self.visibility = visibility
        self.orientation = orientation

        if self.detection_id is None and create_id_if_not_specified:
            self.detection_id = str(uuid4())

        # Set private parameters
        self.__best_class_name = max(self.class_probabilities.items(), key=itemgetter(1))[
            0
        ]  # Class with highest confidence

    def is_certain(self, min_objectness_threshold: float) -> bool:
        """
        Function that checks if the detection is certain enough to be included in the analysis.

        :param min_objectness_threshold: Minimum objectness threshold.
        :return: True if the detection is certain, False otherwise.
        """
        return self.objectness is None or self.objectness >= min_objectness_threshold

    @property
    def best_class_name(self) -> str:
        return self.__best_class_name

    def __repr__(self) -> str:
        """
        Method for printing this object (easier debugging).
        """
        return f"Detection at { id(self) }" if self.detection_id is None else f"Detection { self.detection_id }"
