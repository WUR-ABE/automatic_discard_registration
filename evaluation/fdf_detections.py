from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tqdm.notebook import tqdm

from evaluation import EvaluationResult, EvaluatorBase

if TYPE_CHECKING:
    from typing import Optional


class FDFDetectionEvaluator(EvaluatorBase):
    def associate_results_with_gt(
        self, visibility_class: Optional[str] = None, orientation_class: Optional[str] = None
    ) -> EvaluationResult:
        # Get all image names
        all_image_names = set([*self.gt_dict] + [*self.result_dict])

        # Add the background class to the labels
        self.classes.insert(0, "background") if "background" not in self.classes else self.classes

        # Define the detection gt dictionary
        match_list = []

        # Define variables for confusion matrix gneration in Sklearn
        y_true_list = []
        y_pred_list = []

        # Loop over the images
        for image_name in tqdm(sorted(list(all_image_names)), desc="Associate"):
            # Get all the detections and ground truth in the image
            gt_in_image_list = self.gt_dict.get(image_name, [])  # type: ignore
            detection_in_image_list = self.result_dict.get(image_name, [])  # type: ignore

            # Define initial empty lists
            matched_gt_i = np.array([], dtype=np.uint8)
            matched_detection_i = np.array([], dtype=np.uint8)
            unmatched_gt_i = np.array([], dtype=np.uint8)
            unmatched_detection_i = np.array([], dtype=np.uint8)

            # Skip detections with low confidence
            detection_in_image_list = [
                dt for dt in detection_in_image_list if dt.is_certain(self.min_objectness_threshold)
            ]

            # If there are no ground truth bounding boxes, every detection is a false positive
            if len(gt_in_image_list) == 0 and len(detection_in_image_list) > 0:
                for dt_i, detection in enumerate(detection_in_image_list):
                    unmatched_detection_i = np.append(unmatched_detection_i, dt_i)

                    y_true_list.append("background")
                    y_pred_list.append(detection.best_class_name)
                    match_list.append((None, detection))

            # If there are no detections, every ground truth is a false negative
            elif len(detection_in_image_list) == 0 and len(gt_in_image_list) > 0:
                for gt_i, gt in enumerate(gt_in_image_list):
                    if (visibility_class is not None and gt.visibility != visibility_class) or (
                        orientation_class is not None and gt.orientation != orientation_class
                    ):
                        continue
                    unmatched_gt_i = np.append(unmatched_gt_i, gt_i)

                    y_true_list.append(gt.best_class_name)
                    y_pred_list.append("background")
                    match_list.append((gt, None))

            # If there are detections and ground truth bounding boxes
            elif len(detection_in_image_list) > 0 and len(gt_in_image_list) > 0:
                # Associate the gt and detections
                matched_gt_i, matched_detection_i = self.match(gt_in_image_list, detection_in_image_list)

                # Get all unmatched ground truth bounding boxes and if they have no corresponding detection, mark as
                # false negative
                for gt_i, gt in enumerate(gt_in_image_list):
                    if (visibility_class is not None and gt.visibility != visibility_class) or (
                        orientation_class is not None and gt.orientation != orientation_class
                    ):
                        continue

                    if gt_i not in matched_gt_i:
                        gt = gt_in_image_list[gt_i]

                        # if not np.all(iou_matrix[gt_i, :] == 0, axis=0):
                        unmatched_gt_i = np.append(unmatched_gt_i, gt_i)

                        y_true_list.append(gt.best_class_name)
                        y_pred_list.append("background")
                        match_list.append((gt, None))

                # Get all unmatched detections and if they have no corresponding ground truth, mark as false positive
                for detection_i, detection in enumerate(detection_in_image_list):
                    if detection_i not in matched_detection_i:
                        # if not np.all(iou_matrix[:, detection_i] == 0, axis=0):
                        unmatched_detection_i = np.append(unmatched_detection_i, detection_i)

                        y_true_list.append("background")
                        y_pred_list.append(detection.best_class_name)
                        match_list.append((None, detection))

                for gt_i, detection_i in zip(matched_gt_i, matched_detection_i):
                    gt = gt_in_image_list[gt_i]
                    detection = detection_in_image_list[detection_i]

                    if (visibility_class is not None and gt.visibility != visibility_class) or (
                        orientation_class is not None and gt.orientation != orientation_class
                    ):
                        continue

                    y_true_list.append(gt.best_class_name)
                    y_pred_list.append(detection.best_class_name)
                    match_list.append((gt, detection))

            # fmt: off
            # Check all the results
            if not (visibility_class is not None or orientation_class is not None):
                assert len(unmatched_gt_i) + len(matched_gt_i) == len(gt_in_image_list), \
                    f"\nunmatched gt: { len(unmatched_gt_i) }\t\tunmatched dt: { len(unmatched_detection_i) }" \
                    f"\nmatched gt: { len(matched_gt_i) }\t\tmatched dt: { len(matched_detection_i) }" \
                    f"\ntotal gt: { len(gt_in_image_list) }\t\ttotal dt: { len(detection_in_image_list) }"
                assert len(unmatched_detection_i) + len(matched_detection_i) == len(detection_in_image_list), \
                    f"\nunmatched gt: {len(unmatched_gt_i)}\t\tunmatched dt: {len(unmatched_detection_i)}" \
                    f"\nmatched gt: {len(matched_gt_i)}\t\tmatched dt: {len(matched_detection_i)}" \
                    f"\ntotal gt: {len(gt_in_image_list)}\t\ttotal dt: {len(detection_in_image_list)}"
            # fmt: on

        assert len(y_true_list) == len(y_pred_list) == len(match_list)

        return EvaluationResult(match_list, y_true_list, y_pred_list, self.classes)
