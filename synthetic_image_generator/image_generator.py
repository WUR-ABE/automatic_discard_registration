from __future__ import annotations

from collections import defaultdict
from itertools import chain
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from zipfile import ZipFile

import cv2
import numpy as np

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

    from annotation_generator import AnnotationGenerator


log = getLogger(__name__)


class BoundingBox(NamedTuple):
    x_min: int
    x_max: int
    y_min: int
    y_max: int


class RGBD(NamedTuple):
    rgb: Path
    depth: Path


class ImageGenerator(object):
    IMAGE_FOLDER = Path(__file__).parent / ".images/"
    BACKGROUND_FOLDER = IMAGE_FOLDER / "background"
    CLASSES_FOLDER = IMAGE_FOLDER / "classes"
    DEBRIS_FOLDER = IMAGE_FOLDER / "debris"

    def __init__(self, annotation_generator: Optional[AnnotationGenerator] = None) -> None:
        self._annotation_generator = annotation_generator

        # Create variables
        self._backgrounds: List[RGBD] = []
        self._classes: Dict[str, List[RGBD]] = defaultdict(list)
        self._debris: Dict[str, List[RGBD]] = defaultdict(list)

        # Some maximum dimensions
        self._top_offset = 200
        self._bottom_offset = 100
        self._conveyer_depth = 10000

        # Define settings
        self._alpha_border_width = 10

        # Define random gains
        self._scale_gain = 0.15
        self._h_gain = 0.2
        self._s_gain = 0.1
        self._v_gain = 0.2

        # Initialise generator
        self._initialise()

    def _initialise(self) -> None:
        log.debug("Initialising image generator..")

        # Extract ZIP file when needed
        if not all([self.BACKGROUND_FOLDER.is_dir(), self.CLASSES_FOLDER.is_dir(), self.DEBRIS_FOLDER.is_dir()]):
            self._extract_images()

        # Initialise backgrounds
        for rgb in self.BACKGROUND_FOLDER.glob("*_rgb.tiff"):
            rgbd = self._bgrd_from_bgr(rgb)
            self._backgrounds.append(rgbd)

        # Initialise classes
        for class_folder in self.CLASSES_FOLDER.iterdir():
            for rgb in class_folder.glob("*_rgb.tiff"):
                rgbd = self._bgrd_from_bgr(rgb)
                self._classes[class_folder.stem].append(rgbd)

        # Initialise debris
        for debris_folder in self.DEBRIS_FOLDER.iterdir():
            for rgb in debris_folder.glob("*_rgb.tiff"):
                rgbd = self._bgrd_from_bgr(rgb)
                self._debris[debris_folder.stem].append(rgbd)

        log.info(f"Initialised { len(self._backgrounds) } background images...")
        log.info(
            f"Initialised { len(self._classes) } classes with total of "
            f"{ len(list(chain.from_iterable(self._classes.values()))) } images..."
        )
        log.info(
            f"Initialised { len(self._debris) } debris classes with total of "
            f"{ len(list(chain.from_iterable(self._debris.values()))) } images..."
        )

    def _extract_images(self) -> None:
        log.info("Extracting image files...")

        zip_path = Path(__file__).parent / "images.zip"
        if not zip_path.is_file():
            log.error("The images archive (images.zip) could not be found!")
            exit(1)

        with ZipFile(zip_path, "r") as zip_file_handler:
            zip_file_handler.extractall(self.IMAGE_FOLDER)
        log.info("Images are extracted...")

    def generate_image(self, counts: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        # Clear the annotations
        if self._annotation_generator:
            self._annotation_generator.clear()

        # Get the background image
        background_i = np.random.randint(0, len(self._backgrounds), dtype=int)
        background_bgrd = self._backgrounds[background_i]
        background_image_bgr = cv2.imread(str(background_bgrd.rgb), cv2.IMREAD_UNCHANGED)
        background_image_depth = cv2.imread(str(background_bgrd.depth), cv2.IMREAD_ANYDEPTH)

        # Calculate the conveyer belt depth (assume median depth = conveyer belt depth)
        self._conveyer_depth = np.median(background_image_depth)

        # Add classes at random position and orientation
        for class_name, count in counts.items():
            for i in range(count):
                background_image_bgr, background_image_depth, annotation = self.add_fish(
                    background_image_bgr, background_image_depth, class_name
                )
                if self._annotation_generator:
                    self._annotation_generator.add_annotation(annotation, class_name)

        # Add some other stuff
        for d_name, d_items in self._debris.items():
            d_i = np.random.randint(-1, len(d_items))
            if d_i >= 0:
                background_image_bgr, background_image_depth = self.add_debris(
                    background_image_bgr, background_image_depth, d_name, i=d_i
                )

        return background_image_bgr, background_image_depth

    def add_fish(
        self, background_image_bgr: np.ndarray, background_image_depth: np.ndarray, class_name: str
    ) -> Tuple[np.ndarray, np.ndarray, BoundingBox]:
        class_i = np.random.randint(0, len(self._classes[class_name]))
        class_bgrd = self._classes[class_name][class_i]
        class_image_bgr = cv2.imread(str(class_bgrd.rgb), cv2.IMREAD_UNCHANGED)
        class_image_depth = cv2.imread(str(class_bgrd.depth), cv2.IMREAD_ANYDEPTH)

        # Augment the color image
        class_image_bgr = self._augment_hsv(class_image_bgr)

        # Translate and rotate the images and merge with background image
        return self._augment_spatial(background_image_bgr, background_image_depth, class_image_bgr, class_image_depth)

    def add_debris(
        self, background_image_bgr: np.ndarray, background_image_depth: np.ndarray, debris_name: str, i: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        d_bgrd = self._debris[debris_name][i]
        d_image_bgr = cv2.imread(str(d_bgrd.rgb), cv2.IMREAD_UNCHANGED)
        d_image_depth = cv2.imread(str(d_bgrd.depth), cv2.IMREAD_ANYDEPTH)

        # Augment the color image
        d_image_bgr = self._augment_hsv(d_image_bgr)

        # Translate and rotate the images and merge with background image
        background_image_bgr, background_image_depth, _ = self._augment_spatial(
            background_image_bgr, background_image_depth, d_image_bgr, d_image_depth
        )
        return background_image_bgr, background_image_depth

    def _augment_hsv(self, image: np.ndarray) -> np.ndarray:
        # Randomly calculate gains
        r = np.random.uniform(-1, 1, 3) * [self._h_gain, self._s_gain, self._v_gain] + 1

        # Split image channels
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

        # Create LUT image
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        # Transform HSV image and return as BGR
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(np.uint8)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def _augment_spatial(
        self,
        background_image_bgr: np.ndarray,
        background_image_depth: np.ndarray,
        foreground_image_bgr: np.ndarray,
        foreground_image_depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, BoundingBox]:
        # Calculate random position of the fish in the image. First step is to rotate and scale the image, because by
        # rotating and scaling the image, its dimensions will change. The new dimensions have to be used in order to
        # calculate the correct x and y ranges.
        angle = np.random.randint(0, 360)
        fx = np.random.uniform(1 - self._scale_gain, 1 + self._scale_gain)
        fy = np.random.uniform(1 - self._scale_gain, 1 + self._scale_gain)
        class_image_rs_size = self._get_rs_size(foreground_image_bgr, fx, fy, angle)

        # Calculate new x and y
        x = np.random.randint(
            self._top_offset,
            max(
                self._top_offset + 1,
                background_image_bgr.shape[0] - self._bottom_offset - class_image_rs_size[0],
            ),
        )
        y = np.random.randint(
            -0.5 * class_image_rs_size[1],
            max(1, background_image_bgr.shape[1] - 0.5 * class_image_rs_size[1]),
        )

        # Scale and rotate the image
        s_image_bgr = cv2.resize(foreground_image_bgr, (0, 0), fx=fx, fy=fy)
        s_image_depth = cv2.resize(foreground_image_depth, (0, 0), fx=fx, fy=fy)
        rs_image_bgr = self._rotate_image(s_image_bgr, angle)
        rs_image_depth = self._rotate_image(s_image_depth, angle)

        # Get the height, width and number of channels of the images
        f_h, f_w, f_c = rs_image_bgr.shape
        b_h, b_w, _ = background_image_bgr.shape

        # Convert a 4-channel image to 3-channels if nessesary
        if f_c == 4:
            rs_image_bgr = cv2.cvtColor(rs_image_bgr, cv2.COLOR_BGRA2BGR)

        # Add black pixels in order to create equal dimensions with background
        foreground_image_fs_bgr = np.zeros(background_image_bgr.shape, dtype=np.uint8)
        foreground_image_fs_depth = np.full(
            background_image_depth.shape, np.median(background_image_depth), dtype=np.uint16
        )  # Fill with depth 10000
        foreground_image_fs_bgr[max(0, x) : min(b_h, x + f_h), max(0, y) : min(b_w, y + f_w)] = rs_image_bgr[
            max(0, -x) : min(f_h, b_h - x), max(0, -y) : min(f_w, b_w - y)
        ]
        foreground_image_fs_depth[max(0, x) : min(b_h, x + f_h), max(0, y) : min(b_w, y + f_w)] = rs_image_depth[
            max(0, -x) : min(f_h, b_h - x), max(0, -y) : min(f_w, b_w - y)
        ]

        # Create a foreground mask
        foreground_mask = np.ones(foreground_image_fs_bgr.shape[:2], dtype=np.uint8)
        foreground_mask[(foreground_image_fs_bgr[:, :] == [0, 0, 0])[:, :, 0]] = 0

        # Erode the foreground mask (because of small black line around the fish)
        kernel = np.ones((5, 5), np.uint8)
        foreground_mask = cv2.erode(foreground_mask, kernel, iterations=2)

        # Merge the images using an alpha mask
        merged_image_bgr = self._alpha_bgr_merge(
            background_image_bgr, foreground_image_fs_bgr, foreground_mask, alpha_border_size=self._alpha_border_width
        )
        merged_image_depth = self._depth_merge(background_image_depth, foreground_image_fs_depth, foreground_mask)

        # Create the corresponding annotation bounding box (somehow x and y are reversed here :/)
        annotation = self._calculate_bounding_box(foreground_mask)

        return merged_image_bgr, merged_image_depth, annotation

    def _get_rs_size(self, image: np.ndarray, fx: float, fy: float, angle: int) -> np.ndarray:
        r_image = cv2.resize(image, (0, 0), fx=fx, fy=fy)
        rs_image = self._rotate_image(r_image, angle)
        return rs_image.shape

    def _depth_merge(
        self, background_image: np.ndarray, foreground_image: np.ndarray, mask: np.ndarray, border_width: int = 10
    ) -> np.ndarray:
        # Convert foreground, background and mask image as float
        foreground_image = foreground_image.astype(np.float32)
        background_image = background_image.astype(np.float32)
        mask = mask.astype(np.float32)

        # Get relative foreground depth (assume median depth = conveyer belt depth)
        rel_foreground_image = foreground_image - self._conveyer_depth

        # Multiply foreground with alpha
        rel_foreground_image = cv2.multiply(mask, rel_foreground_image)

        # Subtract the depth of the foreground from the background (closer to the camera)
        subtracted_image = cv2.add(background_image, rel_foreground_image)
        subtracted_image[subtracted_image <= 0] = 0

        return subtracted_image.astype(np.uint16)

    @staticmethod
    def _alpha_bgr_merge(
        background_image: np.ndarray, foreground_image: np.ndarray, mask: np.ndarray, alpha_border_size: float = 10
    ) -> np.ndarray:
        # Create alpha mask
        alpha_mask = cv2.distanceTransform(mask, cv2.DIST_WELSCH, 0)
        alpha_mask[alpha_mask >= alpha_border_size] = alpha_border_size
        cv2.normalize(alpha_mask, alpha_mask, 0.0, 1.0, cv2.NORM_MINMAX)  # Normalize between 0 - 1
        alpha_mask_3c = np.dstack((alpha_mask, alpha_mask, alpha_mask))  # Convert to 3 channel mask

        # Convert foreground and background image as float
        foreground_image = foreground_image.astype(np.float32)
        background_image = background_image.astype(np.float32)

        # Multiply foreground with alpha
        foreground_image = cv2.multiply(alpha_mask_3c, foreground_image)

        # Multiply background with 1 - alpha
        background_image = cv2.multiply(1.0 - alpha_mask_3c, background_image)

        # Add both images
        return cv2.add(foreground_image, background_image).astype(np.uint8)

    @staticmethod
    def _calculate_bounding_box(mask: np.ndarray) -> BoundingBox:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fish_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(fish_contour)
        return BoundingBox(x_min=x, x_max=x + w, y_min=y, y_max=y + h)

    @staticmethod
    def _bgrd_from_bgr(rgb: Path) -> RGBD:
        # Get depth image name
        depth_name = rgb.name.replace("_rgb", "_depth")

        # Check if the depth image is available
        assert Path(rgb.parent / depth_name).is_file(), f"{ depth_name } is not available!"

        # Return RGBD image
        return RGBD(
            rgb=rgb,
            depth=rgb.parent / depth_name,
        )

    @staticmethod
    def _rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        # Function from: https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py

        # Grab image dimensions and calculate the image center
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2

        # Create the rotation matrix (apply negative angle to rotate clockwise) and grab the
        # sinus an cosinus (the rotation component of the matrix).
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos_angle = np.abs(rotation_matrix[0, 0])
        sin_angle = np.abs(rotation_matrix[0, 1])

        # Compute the new dimensions
        nw = int((h * sin_angle) + (w * cos_angle))
        nh = int((h * cos_angle) + (w * sin_angle))

        # Adjust the rotation matrix tot take the new dimensions into the translation
        rotation_matrix[0, 2] += (nw / 2) - cx
        rotation_matrix[1, 2] += (nh / 2) - cy

        # Perform rotation and return
        return cv2.warpAffine(image, rotation_matrix, (nw, nh))
