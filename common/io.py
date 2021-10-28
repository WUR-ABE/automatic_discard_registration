from __future__ import annotations

from collections import OrderedDict, defaultdict
from json import dump, load
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from tqdm.notebook import tqdm

from common.detection import Detection

if TYPE_CHECKING:
    from typing import Dict, List, Union

    from tracking.tracker import Tracker


def to_image(path_or_image: Union[Path, np.ndarray]) -> np.ndarray:
    if isinstance(path_or_image, Path):
        return cv2.imread(str(path_or_image), cv2.IMREAD_COLOR)
    return path_or_image


def load_image_file_names(
    image_folder: Path, image_extensions: List[str] = [".tiff", ".png", ".jpg"]
) -> Dict[str, Path]:
    """
    Find all images in the image path, including images in sub directories.

    :param image_folder: The folder with the images.
    :param image_extensions: The image extensions that should be considered as image.
    :return: A dictionary with the image name and path.
    """
    file_dict = {}
    for image_file in tqdm(image_folder.glob("**/*"), desc="Loading images"):
        if image_file.suffix not in image_extensions:
            continue

        file_dict[image_file.stem] = image_file

    # Order the images alphabetical
    file_dict = OrderedDict(sorted(file_dict.items()))

    return file_dict


def load_detection_file(
    detection_file: Path, skip_classes: List[str] = [], create_id_if_not_specified: bool = False
) -> Dict[str, List[Detection]]:
    detection_dict = defaultdict(list)

    with detection_file.open("r", encoding="utf-8") as dt_file:
        json_data = load(dt_file)

    # Loop over the images
    for image_name, json_detections in tqdm(json_data.items(), desc="Loading detections"):
        for json_detection in json_detections:
            detection = Detection(
                image_name=image_name,
                class_probabilities=json_detection["classes"],
                box_x1y1wh=np.array(json_detection["box_x1y1wh"]),
                detection_id=json_detection.get("id", None),
                objectness=json_detection.get("objectness", None),
                create_id_if_not_specified=create_id_if_not_specified,
            )

            # Skip detection if class name should be skipped
            if detection.best_class_name in skip_classes:
                continue

            detection_dict[detection.image_name].append(detection)
    return detection_dict


def load_annotation_files(annotation_folder: Path, skip_classes: List[str] = []) -> Dict[str, List[Detection]]:
    detection_dict = defaultdict(list)

    for annotation_file in tqdm(annotation_folder.glob("**/*.json"), desc="Loading detections"):
        with annotation_file.open("r", encoding="utf-8") as json_file:
            json_data = load(json_file)

            for json_detection in json_data["annotation"]["objects"]:
                detection = Detection(
                    image_name=annotation_file.stem,
                    class_probabilities=dict(zip(json_detection["label"], [1.0] * len(json_detection["label"]))),
                    box_x1y1wh=np.array(
                        [
                            json_detection["bounding_box"]["x_min"],
                            json_detection["bounding_box"]["y_min"],
                            json_detection["bounding_box"]["x_max"] - json_detection["bounding_box"]["x_min"],
                            json_detection["bounding_box"]["y_max"] - json_detection["bounding_box"]["y_min"],
                        ]
                    ),
                    detection_id=json_detection["id"],
                    visibility=json_detection["visibility"],
                    orientation=json_detection["orientation"],
                )

                # Skip detection if class name should be skipped
                if detection.best_class_name in skip_classes:
                    continue

                detection_dict[detection.image_name].append(detection)
    return detection_dict


def write_detections_to_json(output_file: Path, output_dict: Dict[str, List[Detection]]) -> None:
    """
    Writes a dictionary to JSON. Optional arguments (kwargs).

    :param output_file: The output JSON file.
    :param output_dict: The output dict with the image names as key with a list of detections as values.
    """
    with output_file.open("w", encoding="utf-8") as output_file_handler:
        output_json = defaultdict(list)
        for image_name, detections_in_image in output_dict.items():
            for detection in detections_in_image:
                output_json[image_name].append(
                    {
                        "box_x1y1wh": detection.box_x1y1wh.flatten().tolist(),
                        "classes": detection.class_probabilities,
                        "id": detection.detection_id,
                        "objectness": detection.objectness,
                    }
                )
        dump(output_json, output_file_handler, indent=4)


def write_trackers_to_json(output_file: Path, output_trackers: List[Tracker], method: str = "basian") -> None:
    with output_file.open("w", encoding="utf-8") as output_file_handler:
        output_json = defaultdict(list)

        for _tracker in output_trackers:
            # Don't add these trackers
            if not _tracker.was_confirmed():
                continue

            if method == "basian":
                class_probabilities = _tracker.basian_class_probabilities()
            elif method == "popular_vote":
                class_probabilities = _tracker.popular_vote_class_probabilities()
            else:
                raise NotImplementedError

            # Add to each historical image
            for image_name, tracker_position in _tracker.tracker_positions.items():
                output_json[image_name].append(
                    {
                        "box_x1y1wh": tracker_position.flatten().tolist(),
                        "classes": class_probabilities,
                        "id": _tracker.track_id,
                        "objectness": None,
                        "status": _tracker._state.name,
                        "times_updated": _tracker._times_updated,
                        "times_predicted": _tracker._times_predicted,
                    }
                )

        dump(output_json, output_file_handler, indent=2)
