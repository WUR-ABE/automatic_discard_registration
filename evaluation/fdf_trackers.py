from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, List

    from common.detection import Detection


def count_trackers_by_class(trackers: Dict[str, List[Detection]]) -> Dict[str, int]:
    output_dict = {}
    for tracker_list in trackers.values():
        for tracker in tracker_list:
            assert tracker.detection_id is not None

            output_dict[tracker.detection_id] = tracker.best_class_name

    return Counter(output_dict.values())
