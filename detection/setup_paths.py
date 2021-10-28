import sys
from pathlib import Path

submodules = ["yolov3", "apex"]

detection_base_path = Path(__file__).parent

for module in submodules:
    module_path = detection_base_path / module

    assert module_path.is_dir()

    if str(module_path) not in sys.path:
        sys.path.append(str(module_path))
