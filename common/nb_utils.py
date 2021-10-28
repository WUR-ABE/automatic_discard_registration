from __future__ import annotations

import os
from random import choice, randint
from typing import TYPE_CHECKING
from zipfile import ZipFile

import cv2
import numpy as np
import requests
from sklearn.metrics import precision_recall_curve, confusion_matrix
from tqdm.notebook import tqdm

from common.detection import Detection
from detection import setup_paths
from detection.yolov3.utils.datasets import LoadImagesAndLabels
from detection.yolov3.utils.utils import plot_one_box, xywh2xyxy

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Dict, List, Union

    from matplotlib import axes
    from matplotlib.image import AxesImage

    from evaluation.fdf_detections import EvaluationResult

LINE_COLORS = ["#ffe119", "#61d04f", "#f58231", "#dcbeff", "#000075", "#a9a9a9"]


def download_from_url(url: str, dst: Union[str, Path]) -> None:
    if os.path.exists(dst):
        print("Dataset already downloaded...")
        return

    try:
        file_size = int(requests.head(url).headers["Content-Length"])

        if os.path.exists(dst):
            first_byte = os.path.getsize(dst)
        else:
            first_byte = 0
        if first_byte >= file_size:
            return
        header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
        pbar = tqdm(
            total=file_size, initial=first_byte, unit="B", unit_scale=True, desc=url.split("/")[-1], unit_divisor=1024
        )
        req = requests.get(url, headers=header, stream=True)
    except KeyError:
        pbar = tqdm(total=1, initial=0, unit_scale=True, desc="Downloading ", unit_divisor=1024)
        req = requests.get(url, stream=True)

    with (open(dst, "ab")) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()


def extract_zipfile(src: Union[str, Path], dst: Union[str, Path]) -> None:
    with ZipFile(src) as zf:
        for member in tqdm(zf.infolist(), desc="Extracting "):
            zf.extract(member, dst)


def show_image(image: np.ndarray, ax: axes._subplots.AxesSubplot, **kwargs) -> AxesImage:
    return ax.imshow(image[:, :, ::-1], interpolation="nearest", **kwargs)


def show_image_with_detections(
    image: np.ndarray, detections: List[Detection], ax: axes._subplots.AxesSubplot
) -> AxesImage:
    img = image.copy()
    for dt in detections:
        plot_one_box(dt.box_x1y1x2y2, img, label=dt.best_class_name)

    return show_image(img, ax)


def show_image_with_trackers(image: np.ndarray, trackers: List[Detection], ax: axes._subplots.AxesSubplot) -> AxesImage:
    img = image.copy()
    for trk in trackers:
        plot_one_box(trk.box_x1y1x2y2, img, label=f"{ trk.best_class_name } {trk.detection_id }", color=(255, 255, 0))

    img = cv2.resize(img, (544, 408))  # To save space

    return show_image(img, ax, animated=True)


def show_random_image(dataset: LoadImagesAndLabels, ax: axes._subplots.AxesSubplot, classes: List[str]) -> AxesImage:
    n = randint(0, len(dataset))

    # Repeat till we have images with labels
    while len(list(dataset.labels)[n]) == 0:
        n = randint(0, len(dataset))

    image = cv2.imread(dataset.img_files[n])

    xyxy = xywh2xyxy(dataset.labels[n][:, 1:])
    xyxy *= [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]

    for i, lbl in enumerate(dataset.labels[n]):
        plot_one_box(
            xyxy[
                i,
            ],
            image,
            label=classes[int(lbl[0])],
        )

    return show_image(image, ax)


def show_random_image_with_detection(
    images_dict: Dict[str, Path], detections_dict: Dict[str, List[Detection]], ax: axes._subplots.AxesSubplot
) -> AxesImage:
    image_name = choice([*images_dict])

    # Repeat till we have an image with a detection
    while len(detections_dict[image_name]) == 0:
        image_name = choice([*images_dict])

    image = cv2.imread(str(images_dict[image_name]))

    return show_image_with_detections(image, detections_dict[image_name], ax)


def show_mc_precision_recall_curve(
    result: EvaluationResult, classes: List[str], ax: axes._subplots.AxesSubplot, fontsize: float = 12,
) -> None:
    for i, class_name in enumerate(classes):
        y_true = []
        y_scores = []
        for gt, dt in result.match:
            if gt is None or dt is None:
                continue

            y_true.append(1 if gt.best_class_name == class_name else 0)
            y_scores.append(
                dt.objectness * dt.class_probabilities[class_name]
            )  # Class confidence in YOLO is obj * c_conf

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        label = class_name.replace("_", "\n").capitalize()
        ax.plot(recall, precision, lw=2, label=label, color=LINE_COLORS[i])

    csfont = {'fontname':'Times New Roman'}
    ax.set_xlabel("Recall", fontsize=fontsize)
    ax.set_ylabel("Precision", fontsize=fontsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid()
    ax.legend(bbox_to_anchor=(1.05, 0.6), frameon=False, fontsize=fontsize - 4)
    # ax.set_title("Precision vs. Recall")

def show_confusion_matrix(result: EvaluationResult, ax: axes._subplots.AxesSubplot) -> None:
    cm = confusion_matrix(result.y_true, result.y_pred, labels=sorted(result.classes))
    cm_norm = confusion_matrix(result.y_true, result.y_pred, labels=sorted(result.classes), normalize="true")

    # Create labels
    labels = [class_name.replace("_", "\n").capitalize() for class_name in result.classes]
    labels = [l.replace("Lesser\nspotted\ndogfish", "Lesser spotted\ndogfish") for l in labels]

    # Create the confusion matrix
    cb = ax.imshow(cm_norm, cmap="Greens")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=9, multialignment="right")
    ax.set_yticklabels(labels, fontsize=9, multialignment="right")
    ax.set_ylabel("True fish species")
    ax.set_xlabel("Predicted fish species")

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "black" if cm_norm[j, i] < 0.6 else "white"
            ax.annotate(f"{ cm[j, i] }", (i, j), color=color, va="center", ha="center")

    ax.get_figure().colorbar(cb, ax=ax, label="Percentage for each species")
