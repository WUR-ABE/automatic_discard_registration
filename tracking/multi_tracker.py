from __future__ import annotations

from json import decoder, load
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.io import to_image
from tracking.metrics.iou import IoUMetric
from tracking.tracker import Tracker
from tracking.translation_estimator import ORBTranslationEstimator

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple, Type, Union

    from common.detection import Detection
    from tracking.metrics import MetricBase

    ASSOCIATE_TYPE = Tuple[np.array, np.array, np.array]


log = getLogger("SortFish")


class MultiTracker(object):
    def __init__(
        self,
        first_image: Union[Path, np.ndarray],
        metric: Type[MetricBase] = IoUMetric,
        min_create_objectness_threshold: Optional[float] = None,
        max_age: int = 3,
        n_init: int = 2,
    ) -> None:
        """
        Tracking module that can hold and update multiple trackers.

        :param first_image_path: The image path of the first image, needed to initialise the velocity estimation.
        :param metric: The metric to evaluate the association of tracker and detection.
        :param min_create_confidence_threshold: An optional confidence threshold that the detections should suppress in
                                                order to create a new tracker.
        :param max_age: Maximum age of a tracker.
        :param n_init: Minimim initialise images.
        """
        # Create list of all and the active trackers (active trackers are a subset of all trackers)
        self._trackers: List[Tracker] = []
        self._active_trackers: List[Tracker] = []

        self._next_id = 1
        self._metric = metric
        self._min_create_objectness_threshold = min_create_objectness_threshold
        self._max_age = max_age
        self._n_init = n_init

        # Load the first
        first_image = to_image(first_image)
        self._velocity_estimator = ORBTranslationEstimator(first_image)

    def predict(self, image_name: str) -> None:
        """
        Predicts the position of the bounding box based on the velocity in the Kalman filter.

        :param image_name: The name of the current image (needed to store the predicted position).
        """
        for tracker in self._active_trackers:
            tracker.predict(image_name)

        # Create new list of active trackers, since some trackers can be out of image
        self._active_trackers = [tracker for tracker in self._trackers if tracker.is_active()]

    def update(self, detections: List[Detection], image: Union[Path, np.ndarray], image_name: str) -> None:
        """
        Function that matches the trackers with the detections and updates the corresponding Kalman filters with the new
        tracker position.

        :param detections: A list of detections in the current image.
        :param image: The image (or path), needed to estimate the velocity between this image and the previous image.
        :param image_name: The current image name, needed for the initialisation of the new trackers (so we can remember
                           the first tracker position = detection position).
        """
        # Estimate the velocity between the image
        image = to_image(image)

        # Update the estimator
        self._velocity_estimator.update(image)

        # Estimate the velocity
        velocity_estimate = self._velocity_estimator.predict()

        # Associate the current detections with the trackers
        (
            matched_trackers_ind,
            unmatched_detections_ind,
            unmatched_trackers_ind,
        ) = self._associate(detections, self._metric)

        log.debug(f"Matched { len(matched_trackers_ind) } trackers")

        # Update the matched trackers
        for match in matched_trackers_ind:
            self._active_trackers[match[1]].update(detections[match[0]], velocity_estimate, image_name)

        # Mark the missed trackers
        for ind in unmatched_trackers_ind:
            self._active_trackers[ind].mark_missed()

        # Create new trackers for unmatched detections
        for ind in unmatched_detections_ind:
            self._initiate_tracker(detections[ind], velocity_estimate)

        # Create a new list with active trackers
        self._active_trackers = [tracker for tracker in self._trackers if tracker.is_active()]

    def get_trackers(self) -> List[Tracker]:
        """
        Returns all the trackers.

        :return: A list of all the trackers (active, invalid etc.).
        """
        return self._trackers

    def archive_active_trackers(self) -> None:
        """
        Archives current active trackers by setting all confirmed trackers to OutOfImage.
        """
        for tracker in self._trackers:
            tracker.archive_tracker()

        # Create a new list with active trackers
        self._active_trackers = [tracker for tracker in self._trackers if tracker.is_active()]

        log.info("Archived all active trackers...")

    def _associate(self, detections: List[Detection], metric: Type[MetricBase]) -> ASSOCIATE_TYPE:
        """
        Associates the current detections with the trackers.

        Adapted from https://github.com/abewley/sort/blob/master/sort.py

        :param detections: The detections in the current image.
        :return: The associated trackers with the current detections.
        """
        # Return if there are no trackers yet
        if len(self._active_trackers) == 0:
            log.debug("No active trackers to match")
            unmatched_detection_ind = np.arange(0, len(detections))
            return np.array([]), unmatched_detection_ind, np.array([])

        # Return if there are no detections
        if len(detections) == 0:
            log.debug("No detections to match")
            unmatched_tracker_ind = np.arange(0, len(self._active_trackers))
            return np.array([]), np.array([]), unmatched_tracker_ind

        # Create a cost matrix (the costs of matching a specific tracker with a specific detection) and fill the cost
        # metric based on the metric calculation
        cost_matrix = np.zeros((len(detections), len(self._active_trackers)), dtype=np.float)

        for d, detection in enumerate(detections):
            for t, tracker in enumerate(self._active_trackers):
                cost_matrix[d, t] = metric.calculate(detection, tracker)

        # Uncomment for printing the cost matrix:
        # with np.printoptions(precision=3, suppress=True):
        #     print("\n")
        #     print(cost_matrix)

        # Filter out the too values that aren't valid
        if metric.maximize:
            cost_matrix[cost_matrix < metric.threshold] = metric.threshold
        else:
            cost_matrix[cost_matrix > metric.threshold] = metric.threshold

        # Apply Hungarian algorithm to associate detections (workers) with the trackers (jobs) based on combination
        # with the lowest cost
        matched_detection_ind_list, matched_tracker_ind_list = linear_sum_assignment(
            cost_matrix, maximize=metric.maximize
        )
        matched_indices = np.array(list(zip(matched_detection_ind_list, matched_tracker_ind_list)))

        # Get the unmatched detections and trackers
        unmatched_detection_ind = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
        unmatched_tracker_ind = [t for t in range(len(self._active_trackers)) if t not in matched_indices[:, 1]]

        # Filter out matches with a cost that is too high
        matched_tracker_ind = []
        for match_ind in matched_indices:
            if metric.is_valid(cost_matrix[match_ind[0], match_ind[1]]):
                matched_tracker_ind.append(match_ind.reshape(1, 2))
            else:
                unmatched_detection_ind.append(match_ind[0])
                unmatched_tracker_ind.append(match_ind[1])

        if len(matched_tracker_ind) == 0:
            matched_tracker_ind = np.empty((0, 2), dtype=int)
        else:
            matched_tracker_ind = np.concatenate(matched_tracker_ind, axis=0)

        return (
            matched_tracker_ind,
            unmatched_detection_ind,
            unmatched_tracker_ind,
        )

    def _initiate_tracker(self, detection: Detection, velocity_estimate: np.array) -> None:
        """
        Function that creates a tracker when the detection confidence is higher than the specified value.

        :param detection: The detection to create a tracker.
        :param velocity_estimate: The velocity estimate of the image in which the detection is defined.
        """
        if self._min_create_objectness_threshold is None or detection.is_certain(self._min_create_objectness_threshold):
            tracker = Tracker(
                detection,
                velocity_estimate,
                self._next_id,
                max_age=self._max_age,
                n_init=self._n_init,
            )
            self._trackers.append(tracker)
            self._next_id += 1
