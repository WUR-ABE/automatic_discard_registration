from __future__ import annotations

from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchvision

from common.detection import Detection
from detection import setup_paths
from detection.yolov3.models import Darknet, load_darknet_weights
from detection.yolov3.utils.datasets import letterbox
from detection.yolov3.utils.torch_utils import select_device, time_synchronized
from detection.yolov3.utils.utils import box_iou, load_classes, scale_coords, xywh2xyxy

if TYPE_CHECKING:
    from typing import Dict, List, Union

with torch.no_grad():

    class FDFDetector:
        def __init__(
            self,
            weights_file: Path,
            path_dict: Dict[str, Path],
            img_size: int = 512,
            conf_tresh: float = 0.1,
            obj_tresh: float = 0.5,
            iou_tresh: float = 0.8,
        ) -> None:
            self.device = select_device()
            cfg_file = path_dict["cfg_file"]

            # Initialise the model
            self.model = Darknet(str(cfg_file), img_size=img_size)

            # Load weights
            if weights_file.suffix == ".pt":
                # Fix loading of checkpoints
                # https://github.com/ultralytics/yolov3/issues/1471
                tm = torch.load(weights_file, map_location=self.device)
                state_dict = []
                for n, p in tm["model"].items():
                    if "total_ops" not in n and "total_params" not in n:
                        state_dict.append((n, p))
                self.model.load_state_dict(dict(state_dict))
            else:
                load_darknet_weights(self.model, str(weights_file))

            # Set in evaluation mode
            self.model.to(self.device).eval()

            # Load class names and assign colors to each class
            self.names = load_classes(path_dict["names_file"])
            self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

            # Create parameters
            self.img_size = img_size
            self.obj_tresh = obj_tresh
            self.conf_tresh = conf_tresh
            self.iou_tresh = iou_tresh

        def detect(self, im0: np.ndarray, image_path: Path) -> List[Detection]:
            dt_list = []

            img = letterbox(im0, new_shape=self.img_size)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            # Load image as float in GPU
            img = torch.from_numpy(img).to(self.device)
            img = img.float()
            img /= 255.0  # type: ignore

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=False)[0]
            t2 = time_synchronized()

            pred = self._non_max_suppression_proba(
                pred,
                obj_thres=self.obj_tresh,
                conf_thres=self.conf_tresh,
                iou_thres=self.iou_tresh,
            )

            for i, det in enumerate(pred):
                s, im0 = "", im0
                s += "%gx%g " % img.shape[2:]  # type: ignore

                if det is not None and len(det):
                    # Rescale boxes
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, 5:].argmax(dim=1).unique():
                        n = (det[:, 5:].argmax(dim=1) == c).sum()
                        s += "%g %ss, " % (n, self.names[int(c)])

                    # Write results
                    for xyxy_1, xyxy_2, xyxy_3, xyxy_4, obj, *class_conf in reversed(det):
                        xyxy = torch.stack([xyxy_1, xyxy_2, xyxy_3, xyxy_4]).tolist()

                        class_conf = torch.tensor(class_conf).view(-1)
                        classes = {}
                        for i, conf in enumerate(class_conf.tolist()):
                            classes[self.names[i]] = conf

                        # This length should be larger than one, otherwise NMS should have filtered out this detection
                        assert len(classes) >= 1

                        # Create detection
                        _img_name = image_path.stem
                        dt = Detection(
                            image_name=_img_name,
                            class_probabilities=classes,
                            box_x1y1wh=[0, 0, 1, 1],  # Replace later with xyxy
                            objectness=obj.item(),
                        )
                        dt.box_x1y1x2y2 = xyxy

                        dt_list.append(dt)

                # Print time (inference + NMS)
                # tqdm.write("%sDone. (%.3fs)" % (s, t2 - t1))

            return dt_list

        @staticmethod
        def _non_max_suppression_proba(
            prediction: torch.Tensor,
            obj_thres: float = 0.1,
            conf_thres: float = 0.1,
            iou_thres: float = 0.6,
            agnostic: bool = False,
        ) -> List[Union[int, float, None]]:
            # Some settings
            merge = True
            min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
            time_limit = 10

            t = time()
            nc = prediction[0].shape[1] - 5
            output = [None] * prediction.shape[0]

            for xi, x in enumerate(prediction):
                # Apply constraints on confidence and min width-height
                x = x[x[:, 4] > obj_thres]
                x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]

                # If no predictions remain, go to the next image
                if not x.shape[0]:
                    continue

                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                box = xywh2xyxy(x[:, :4])

                # Keep box if maximum confidence is higher than the threshold
                conf, c = x[:, 5:].max(1)
                x = torch.cat((box, x[:, 4:]), 1)[conf > conf_thres]
                c = c[conf > conf_thres]

                # If no predictions remain, go the the next image
                if not x.shape[0]:
                    continue

                n = x.shape[0]

                # If agnostic, don't take class into account for NMS
                if agnostic:
                    c *= 0

                # Give boxes offset of class_number * max_wh to take class number into account for NMS
                boxes = x[:, :4].clone() + c.view(-1, 1) * max_wh
                scores = x[:, 4]

                # Apply NMS
                i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)

                # Merge NMS (boxes merged using weighted mean)
                if merge and (1 < n < 3e3):
                    try:
                        iou = box_iou(boxes[i], boxes) > iou_thres

                        # Weight boxes
                        weights = iou * scores[None]

                        # Merged boxes
                        x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
                    except:
                        pass

                # Create output
                output[xi] = x[i]

                # Stop if time limit exeeds
                if (time() - t) > time_limit:
                    break

            return output  # type: ignore
