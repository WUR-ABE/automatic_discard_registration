from __future__ import annotations

import os
from itertools import repeat
from json import load
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.lib.function_base import average
from PIL import Image
from tqdm.notebook import tqdm

from detection import setup_paths
from detection.yolov3.utils.datasets import LoadImagesAndLabels, exif_size, load_image

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional


IMAGE_TYPES = [".tiff", ".png", ".jpg"]


def get_hash(files: List[str]) -> int:
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def get_images(dataset_path: Path, subset_name: str = "train") -> List[str]:
    """
    Requires dataset in format:

    dataset/
    | -- images/
    |    | -- subset_name/
    | -- annotations/
    |    | -- subset_name/
    """
    image_folder = dataset_path / "images" / subset_name
    images = []
    for f in image_folder.glob("**/*"):
        if f.suffix not in IMAGE_TYPES:
            continue
        images.append(str(f.resolve()))
    return sorted(images)


def img2label_paths(image_list: List[str]) -> List[str]:
    """
    Requires dataset in format:

    dataset/
    | -- images/
    |    | -- subset_name/
    | -- annotations/
    |    | -- subset_name/
    """
    f = []
    for img in image_list:
        assert img.count("/images/") == 1, "Cannot change image to annotations!"
        img_str = img.replace("/images/", "/annotations/")
        img_path = Path(img_str)
        f.append(str(img_path.parent / (img_path.stem + ".json")))
    return f


class FDFLoader(LoadImagesAndLabels):
    def __init__(
        self,
        dataset_path: Path,
        class_names: List[str],
        subset_name: str = "train",
        img_size: int = 640,
        batch_size: int = 16,
        augment: bool = False,
        hyp: Optional[Dict[str, Any]] = None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride: int = 32,
        pad: float = 0.0,
        rank: int = -1,
    ) -> None:
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.class_names = class_names

        # Load image and labels
        self.img_files = get_images(dataset_path, subset_name)
        self.label_files = img2label_paths(self.img_files)

        # Length should be equal
        assert len(self.img_files) > 0, "There is no image!"
        assert len(self.img_files) == len(self.label_files)

        # Check cache
        cache_path = dataset_path / "annotations" / (subset_name + ".cache")
        if cache_path.is_file():
            cache = torch.load(cache_path)
            if cache["hash"] != get_hash(self.label_files + self.img_files) or "results" not in cache:  # changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)

        [nf, nm, ne, nc, n] = cache.pop("results")  # found, missing, empty, corrupted, total
        desc = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        tqdm(None, desc=desc, total=n, initial=n)
        assert nf > 0 or not augment, f"No labels found in {cache_path}. Can not train without labels."

        # Read cache
        cache.pop("hash")  # remove hash
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = "Caching images (%.1fGB)" % (gb / 1e9)

    def cache_labels(self, path: Path) -> Dict[str, Any]:
        x = {}
        nm, nf, ne, nc = 0, 0, 0, 0
        pbar = tqdm(zip(self.img_files, self.label_files), desc="Scanning images", total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # Verify images
                im = Image.open(im_file)
                im.verify()
                shape = exif_size(im)

                assert (shape[0] > 9) & (shape[1] > 9)

                lb_file = Path(lb_file)

                # Verify labels
                if lb_file.is_file():
                    nf += 1  # Label found

                    l = []
                    with lb_file.open("r", encoding="utf-8") as fh:
                        json_data = load(fh)

                        if "annotation" in json_data:
                            dt_yolo_list = []
                            for detection in json_data["annotation"]["objects"]:
                                xmin = detection["bounding_box"]["x_min"]
                                ymin = detection["bounding_box"]["y_min"]
                                xmax = detection["bounding_box"]["x_max"]
                                ymax = detection["bounding_box"]["y_max"]
                                class_name = detection["label"][0]
                                dt_yolo_list.append(
                                    [
                                        self.class_names.index(class_name),
                                        np.average((xmin, xmax)) / shape[0],
                                        np.average((ymin, ymax)) / shape[1],
                                        (xmax - xmin) / shape[0],
                                        (ymax - ymin) / shape[1],
                                    ]
                                )
                            l = np.array(dt_yolo_list, dtype=np.float32)

                    if len(l):
                        assert l.shape[1] == 5, "labels require 5 columns each"
                        assert (l >= 0).all(), "negative labels"
                        assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels"
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], "duplicate labels"
                    else:
                        ne += 1  # Label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    ne += 1  # Label missing
                    l = np.zeros((0, 5), dtype=np.float32)

                x[im_file] = [l, shape]

            except Exception as e:
                nc += 1  # Not correct
                pbar.write("WARNING: Ignoring corrupted image and/or label {}: {}".format(im_file, e))

            pbar.desc = "Scanning {} for images and labels... {} found, {} missing, {} empty, {} corrupted".format(
                path.parent / path.stem, nf, nm, ne, nc
            )

        if nf == 0:
            print("WARNING: No labels found in {}".format(path))

        x["hash"] = get_hash(self.label_files + self.img_files)
        x["results"] = [nf, nm, ne, nc, i + 1]
        torch.save(x, path)
        return x
