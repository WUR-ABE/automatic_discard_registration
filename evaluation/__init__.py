from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import detection.setup_paths
from detection.yolov3.utils.utils import load_classes

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Dict, List, Optional, Tuple

    from common.detection import Detection


class EvaluationResult(NamedTuple):
    match: Tuple[Optional[Detection], Optional[Detection]]
    y_true: List[int]
    y_pred: List[int]
    classes: List[str]


class EvaluatorBase(metaclass=ABCMeta):
    def __init__(
        self,
        gt_dict: Dict[str, List[Detection]],
        results_dict: Dict[str, List[Detection]],
        path_dict: Dict[str, Path],
        min_objectness_threshold: float = 0.4,
        min_iou_threshold: float = 0.5,
        skip_classes: List[str] = [],
    ) -> None:
        # Fix keys by removing the _RGB part
        self.gt_dict = {}
        for k in gt_dict.keys():
            self.gt_dict[k.replace("_RGB", "")] = gt_dict[k]
        self.result_dict = {}
        for k in results_dict.keys():
            self.result_dict[k.replace("_RGB", "")] = results_dict[k]

        # Load the class names
        self.classes = load_classes(path_dict["names_file"])

        # Remove classes from list
        for skc in skip_classes:
            self.classes.remove(skc)

        self.min_objectness_threshold = min_objectness_threshold
        self.min_iou_threshold = min_iou_threshold

    @abstractmethod
    def associate_results_with_gt(
        self, visibility_class: Optional[str] = None, orientation_class: Optional[str] = None
    ) -> EvaluationResult:
        pass

    def match(
        self, gt_in_image_list: List[Detection], detection_in_image_list: List[Detection]
    ) -> Tuple[List[int], List[int]]:
        """
        Function that associates the trackers with the ground truth values.

        :param gt_in_image_list: List of ground truth detections in the image.
        :param detection_in_image_list: List of detections in the image.
        :returns: Tuple of matched ground truth and detection ids.
        """
        # Calculate IoU for all the combinations
        iou_matrix = np.zeros((len(gt_in_image_list), len(detection_in_image_list)), dtype=np.float)
        for gt_i, gt_in_image in enumerate(gt_in_image_list):
            for detection_i, detection_in_image in enumerate(detection_in_image_list):
                iou_matrix[gt_i, detection_i] = gt_in_image.calculate_iou_with(detection_in_image)

        # Set low IoU values to zero
        iou_matrix[iou_matrix < self.min_iou_threshold] = 0

        # Get highest value for each column and make all other values zero
        _column_mask = np.zeros_like(iou_matrix)
        for dt_i in range(len(detection_in_image_list)):
            # Get the highest value indexes for each column
            gt_i = iou_matrix[:, dt_i].argmax(0)

            # Fill the zero matrix with the highest value of each column
            _column_mask[gt_i, dt_i] = np.max(iou_matrix[:, dt_i], axis=0)

        # Get the highest value for each row and make all other values zero
        _row_mask = np.zeros_like(iou_matrix)
        for gt_i in range(len(gt_in_image_list)):
            # Get the highest value indexes for each row
            dt_i = _column_mask[
                gt_i,
            ].argmax(0)

            # Fill the zero matrix with the highest
            _row_mask[gt_i, dt_i] = np.max(_column_mask[gt_i, :], axis=0)

        # Get the best combination
        return np.where(_row_mask != 0)  # type: ignore
