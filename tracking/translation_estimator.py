from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from typing import Optional

log = getLogger(__name__)


class ORBTranslationEstimatorError(Exception):
    pass


class ORBTranslationEstimator(object):
    def __init__(
        self,
        source_image: np.array,
        n_features: int = 500,
        min_inliers: int = 150,
        max_distance: int = 100,
        ransac_reprojection_threshold: float = 5.0,
    ) -> None:
        # Initialize ORB matcher
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        # Copy the variables
        self._min_inliers = min_inliers
        self._max_distance = max_distance
        self._ransac_reprojection_threshold = ransac_reprojection_threshold

        # Remember the previous translation vector
        self._previous_translation_vector: Optional[np.ndarray] = None

        # Save the features from the first image
        self._kp_source, self._des_source = self._orb.detectAndCompute(source_image, None)
        self._kp_target, self._des_target = None, None

    def update(self, target_image: np.array) -> None:
        # If there is a target image already, use it to overwrite the source image, because previous image is set as
        # reference image for this match. Should not be done the first time.
        if self._kp_target is not None and self._des_target is not None:
            self._kp_source, self._des_source = (
                self._kp_target.copy(),
                self._des_target.copy(),
            )

        # Calculate ORB features of target image
        self._kp_target, self._des_target = self._orb.detectAndCompute(target_image, None)

    def predict(self) -> np.array:
        """
        Predicts the velocity (vx, vy) between the two images.

        :return: Numpy array with velocity (vx, vy).
        """
        # Do all checks
        if self._kp_target is None or self._des_target is None:
            raise ORBTranslationEstimatorError("Features of target image are not set, run update() first!")

        # Match the source and target image features
        matches = self._bf.match(self._des_source, self._des_target)

        # Store all matches with distance lower than the threshold
        good_matches = []
        for m in matches:
            if m.distance < self._max_distance:
                good_matches.append(m)

        # Check if we have enough inliers
        if len(good_matches) < self._min_inliers:
            log.warning(
                f"Could not estimate the velocity between this and the previous image, not enough inliers: "
                f"{ len(good_matches ) } / { self._min_inliers }. Returning old translation vector..."
            )
            return (
                np.array([0, 0]).reshape((2, 1))
                if self._previous_translation_vector is None
                else self._previous_translation_vector
            )

        # Calculate the affine transform matrix between the source and target ORB features
        source_pts = np.float32([self._kp_source[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        target_pts = np.float32([self._kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        affine_transform, inliers = cv2.estimateAffine2D(source_pts, target_pts)

        # Affine transformation matrix:
        #
        # | cos(t) -sin(t) vx |
        # | sin(t)  cos(t) vy |
        # | 0       0      1  |
        translation_vector = affine_transform[0:2, 2].reshape((2, 1))
        self._previous_translation_vector = translation_vector
        return translation_vector
