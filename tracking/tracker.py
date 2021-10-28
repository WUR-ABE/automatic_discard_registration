from __future__ import annotations

from collections import OrderedDict, defaultdict
from enum import Enum
from logging import getLogger
from operator import itemgetter
from typing import TYPE_CHECKING

import numpy as np
from filterpy.kalman import KalmanFilter

from common.bounding_box import BoundingBoxMixin

if TYPE_CHECKING:
    from typing import Dict, List, Tuple

    from common.detection import Detection

log = getLogger(__name__)


class TrackerState(Enum):
    Tentative = 0
    Confirmed = 1
    OutOfImage = 2  # Tracker is outside the image regions
    LostTracking = 3  # Tracking is lost after being confirmed
    Deleted = 4  # Prediction has not enough hits to be confirmed
    Invalid = 5


class Tracker(BoundingBoxMixin, KalmanFilter):
    def __init__(
        self,
        detection: Detection,
        velocity_estimate: np.ndarray,
        track_id: int,
        max_age: int = 3,
        n_init: int = 2,
        image_size: Tuple[int, int] = (1600, 1200),
    ) -> None:
        """
        Tracker that tracks an bounding box through several images.

        State space formulation:
        X* = A(t)x(t) + B(t)u(t)
        Y* = C(t)x(t) + D(t)u(t)


        :param detection: The detected bounding box.
        :param velocity_estimate: 2x1 vector that estimates vx, vy for the image.
        :param track_id: The ID of this tracker.
        :param max_age: The max age of the tracker, when tracker isn't seen for this amount of hits, it will be deleted.
        :param n_init: The number of hits before confirming a tracker.
        """
        # Setup the 8-dimensions Kalman filter:
        # x, y, s, r, vx, vy, vs, vr
        super().__init__(box_x1y1wh=detection.box_x1y1wh, dim_x=8, dim_z=6)

        # Set track ID
        self.track_id = track_id
        log.debug(f"Create tracker { self.track_id }")

        # State transition matrix according to our state space system (measurements + their derivatives):
        self.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Measurement function: the 6 measurements are directly corresponding with x, y, s, r, vx, vy
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )

        # Uncertainty on the measurements. Give higher uncertainty on the bounding  box dimensions because we don't
        # measure them that well and the dimensions are dependent on the quality of the prediction.
        self.R = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 10, 0, 0, 0],  # Higher uncertainty on s
                [0, 0, 0, 10, 0, 0],  # Higher uncertainty on r
                [0, 0, 0, 0, 10, 0],  # Higher uncertainty on vx
                [0, 0, 0, 0, 0, 10],  # Higher uncertainty on vy
            ]
        )

        # Covariance matrix for the starting state of the bounding box. High uncertainty on the current bounding box
        # position and lower on the dimensions. High uncertainty on the side-wards movement and a smaller uncertainty
        # on the up-down movement (since we know that the bounding boxes are moving side-wards). Also, give small
        # uncertainty on the change in size of the bounding box.
        self.P = np.array(
            [
                [30, 0, 0, 0, 0, 0, 0, 0],
                [0, 30, 0, 0, 0, 0, 0, 0],
                [0, 0, 10, 0, 0, 0, 0, 0],
                [0, 0, 0, 10, 0, 0, 0, 0],
                [0, 0, 0, 0, 30, 0, 0, 0],  # High uncertainty on vx
                [0, 0, 0, 0, 0, 30, 0, 0],  # Lower uncertainty on vy
                [0, 0, 0, 0, 0, 0, 10, 0],  # Low uncertainty on vs
                [0, 0, 0, 0, 0, 0, 0, 10],  # Low uncertainty on vr
            ]
        )

        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        # Assign the initial values for x, y, s, r, vx, vy from the first measurement
        self.x[:6] = np.concatenate((detection.box_xysr, velocity_estimate), axis=0)

        # fmt: on

        # Create parameters to keep track of the age of the tracker and its state
        self._times_predicted = 1
        self._times_updated = 1
        self._time_since_last_update = 0
        self._state = TrackerState.Tentative
        self._max_age = max_age
        self._n_init = n_init
        self._image_size = image_size

        # Keep the tracker position
        self.tracker_positions: OrderedDict[str, np.ndarray] = OrderedDict()
        self.tracker_positions[detection.image_name] = detection.box_x1y1wh

        # Keep track of the class distribution
        self.class_observations: List[Dict[str, float]] = [
            detection.class_probabilities,
        ]

        # Save detection ids' that where tracked using this tracker
        self._tracked_ids: List[str] = []
        if detection.detection_id:
            self._tracked_ids.append(detection.detection_id)

    def predict(self, image_name: str) -> None:
        """
        Performs a new Kalman prediction.

        :param image_name: The name of the current image (needed to store the predicted position).
        """
        super().predict()

        # Update the associated bounding box
        self.box_xysr = self.x[:4].reshape((4, 1))

        # Check if the prediction is in the image, if not set to out of image
        if not self.in_image(self._image_size):
            self.set_state(TrackerState.OutOfImage)
            return

        # Update tracker position
        self.tracker_positions[image_name] = self.box_x1y1wh
        self._times_predicted += 1
        self._time_since_last_update += 1

        # Mark as deleted when it contains NaN values and was tentative
        if np.any(np.isnan(self.box_xysr)) and self._state == TrackerState.Tentative:
            self.set_state(TrackerState.Deleted)
            return

        # Mark as invalid when it contains NaN values and was confirmed
        if np.any(np.isnan(self.box_xysr)) and self._state == TrackerState.Confirmed:
            self.set_state(TrackerState.Invalid)
            return

        # Restore invalid confirmed tracker
        if not self.is_valid():
            self.set_state(TrackerState.Confirmed)
            return

    def update(self, detection: Detection, velocity_estimate: np.ndarray, image_name) -> None:
        """
        Update the current tracker Kalman filter with an associated detection.

        :param detection: The associated detection.
        :param velocity_estimate: The velocity estimate.
        :param image_name: The image name needed to store teh tracker positions.
        """
        super().update(np.concatenate((detection.box_xysr, velocity_estimate), axis=0))

        # Store the updated state
        self.box_xysr = self.x[:4].reshape((4, 1))

        # Overwrite the tracker position when we have an update for this image name
        self.tracker_positions[image_name] = self.box_x1y1wh

        # Add the class probabilities
        self.class_observations.append(detection.class_probabilities)

        # Make state confirmed when updated successfully after x hits
        self._time_since_last_update = 0
        self._times_updated += 1

        # Mark tracker confirmed if allowed
        if self._state == TrackerState.Tentative and self._times_updated >= self._n_init:
            self.set_state(TrackerState.Confirmed)

        # Update the tracked ids
        if detection.detection_id:
            self._tracked_ids.append(detection.detection_id)

    def mark_missed(self) -> None:
        """
        Mark this track as missed (no association at the current time step).
        """
        # Delete tentative trackers
        if self._state == TrackerState.Tentative:
            self.set_state(TrackerState.Deleted)
            return

        # Set tracking lost for confirmed trackers
        if (
            self._state == TrackerState.Confirmed or self._state == TrackerState.Invalid
        ) and self._time_since_last_update > self._max_age:
            self.set_state(TrackerState.LostTracking)

            # Remove last positions from tracking history since it's lost
            log.info(f"Removing last { self._max_age + 1 } positions for tracker { self.track_id }")
            for _ in range(self._max_age + 1):
                self.tracker_positions.popitem(last=True)
            return

        log.debug(f"Mark missed tracker {self.track_id} ({self._time_since_last_update}/{self._max_age})")

    def set_state(self, new_state: TrackerState):
        log.debug(f"Mark tracker { self.track_id } as { new_state.name }, was { self._state.name }")
        self._state = new_state

    def is_valid(self) -> bool:
        """
        :return: True if state is valid, Else otherwise.
        """
        return self._state != TrackerState.Invalid

    def is_tentative(self) -> bool:
        """
        :return: True if state is tentative, False otherwise.
        """
        return self._state == TrackerState.Tentative

    def is_confirmed(self) -> bool:
        """
        :return: True if state is confirmed, False otherwise.
        """
        return self._state == TrackerState.Confirmed

    def is_active(self) -> bool:
        """
        :return: True if the tracker is active, False otherwise.
        """
        return (
            self._state == TrackerState.Tentative
            or self._state == TrackerState.Confirmed
            or self._state == TrackerState.Invalid
        )

    def was_confirmed(self) -> bool:
        """
        :return: True if the tracker was confirmed sometime, False otherwise.
        """
        return (
            self._state == TrackerState.Confirmed
            or self._state == TrackerState.OutOfImage
            or self._state == TrackerState.LostTracking
        )

    def basian_class_probabilities(self) -> Dict[str, float]:
        """
        Returns the basian class probabilities for the given observations.

        Method from https://jonathanweisberg.org/vip/multiple-conditions.html part 2: multiple witnesses

                                        Pr(class_name)Pr(observations | class_name)
        Pr(class_name | observations) = -------------------------------------------
                                                       Pr(observations)

        where:

        Pr(observations | class_name) = Pr(observation_1 | class_name)...Pr(observation_n | class_name)

        Pr(observations) = Pr(observations | class_name)Pr(class_name) + Pr(observations | ~class_name)Pr(~class_name)

        :return: Dictionary with probabilities for each class.
        """
        classes = self.class_observations[0].keys()

        # Pr(observations | class_name)Pr(class_name) > for each class_name
        pr_observations_given_class_name: Dict[str, float] = defaultdict(lambda: 1 / len(classes))

        for class_name in classes:
            for obs in self.class_observations:
                pr_observations_given_class_name[class_name] *= obs[class_name]

        # Pr(observations)
        pr_observations = sum(pr_observations_given_class_name.values())

        # Pr(class_name | observations) > for each class_name
        class_probabilities = {}
        for class_name in classes:
            class_probabilities[class_name] = pr_observations_given_class_name[class_name] / pr_observations
        return class_probabilities

    def popular_vote_class_probabilities(self) -> Dict[str, float]:
        class_probabilities: Dict[str, float] = {}
        # Summize list of dicts
        for single_probability in self.class_observations:
            class_probabilities = {
                k: class_probabilities.get(k, 0) + single_probability.get(k, 0)
                for k in set(class_probabilities) | set(single_probability)
            }

        # Normalize dict
        factor = 1.0 / sum(class_probabilities.values())
        return {key: value * factor for key, value in class_probabilities.items()}

    def get_most_probable_class_name(self, method: str = "basian") -> str:
        if method == "basian":
            class_probabilities = self.basian_class_probabilities()
        elif method == "popular_vote":
            class_probabilities = self.popular_vote_class_probabilities()
        else:
            raise NotImplementedError

        return max(class_probabilities.items(), key=itemgetter(1))[0]  # type: ignore
