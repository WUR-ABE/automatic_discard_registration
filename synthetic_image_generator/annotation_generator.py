from __future__ import annotations

from json import dump
from logging import getLogger
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Dict, List, Union

    from image_generator import BoundingBox


log = getLogger(__name__)


class AnnotationGenerator(object):
    def __init__(self) -> None:
        self._objects: List[Dict[str, Union[str, List[str], Dict[str, int]]]] = []

    def clear(self) -> None:
        self._objects.clear()

    def add_annotation(self, bounding_box: BoundingBox, class_name: str) -> None:
        self._objects.append(
            {
                "id": str(uuid4()),
                "label": [
                    class_name,
                ],
                "bounding_box": {
                    "x_min": bounding_box.x_min,
                    "x_max": bounding_box.x_max,
                    "y_min": bounding_box.y_min,
                    "y_max": bounding_box.y_max,
                },
                "type": "bounding_box",
                "visibility": "80 - 100%",
                "orientation": "Ventral",
            }
        )
        log.debug(f"Add annotation with class { class_name }...")

    def write_annotation_file(self, annotation_path: Path, image_path: Path) -> None:
        annotation_content = {
            "annotation": {
                "label_fish_version": "3.4.5",  # Newest annotation format
                "folder": str(image_path.parent),
                "file": image_path.name,
                "image_name": image_path.stem,
                "path": str(image_path),
                "annotated_by": ["automatic"],
                "verified": False,
                "objects": self._objects,
            }
        }
        with annotation_path.open("w", encoding="utf-8") as annotation_file_handler:
            dump(annotation_content, annotation_file_handler, indent=2)
