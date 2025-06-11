# Automatic discard registration in cluttered environments using deep learning and object tracking: class imbalance, occlusion, and a comparison to human review

![tracking-example](tracking.gif "tracking-example")
> **Automatic discard registration in cluttered environments using deep learning and object tracking: class imbalance, occlusion, and a comparison to human review**\
> Rick van Essen, Angelo Mencarelli, Aloysius van Helmond, Linh Nguyen, Jurgen Batsleer, Jan-Jaap Poos and Gert Kootstra
> Paper: https://doi.org/10.1093/icesjms/fsab233

## About
This repository contains the code beloning to the paper "Automatic discard registration in cluttered environments using deep learning and object tracking: class imbalance, occlusion, and a comparison to human review". 

## Installation
Python 3.8 is needed with all dependencies listed in [requirements.txt](requirements.txt). Optionally, apex can be installed for faster training:

```commandline
pip install -r requirements.txt
pip install detection/apex
```

## Content
The software contains 5 notebooks:

| Notebook                               |                         | Description                                                                          |
|----------------------------------------|-------------------------|--------------------------------------------------------------------------------------|
| [create_synthetic_data](create_synthetic_data.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WUR-ABE/automatic_discard_registration/blob/master/create_synthetic_data.ipynb) | Notebook to create synthetic data. |
| [train](train.ipynb)                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WUR-ABE/automatic_discard_registration/blob/master/train.ipynb) | Notebook to train the YOLOv3 neural network.                                         |
| [detect](detect.ipynb)                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WUR-ABE/automatic_discard_registration/blob/master/detect.ipynb) | Notebook to detect fish in the images.                                         |
| [track](track.ipynb)                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WUR-ABE/automatic_discard_registration/blob/master/track.ipynb) | Notebook to track the fish over consequtive images.    |
| [evaluate](evaluate.ipynb)             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WUR-ABE/automatic_discard_registration/blob/master/evaluate.ipynb) | Notebook to evaluate the detection and count the number of tracked fish.           |


## Citation
If you find this code usefull, please consider citing our paper:

```text
@article{vanEssen2021,
    author = {van Essen, Rick and Mencarelli, Angelo and van Helmond, Aloysius and Nguyen, Linh and Batsleer, Jurgen and Poos, Jan-Jaap and Kootstra, Gert},
    title = {Automatic discard registration in cluttered environments using deep learning and object tracking: class imbalance, occlusion, and a comparison to human review},
    journal = {ICES Journal of Marine Science},
    volume = {78},
    number = {10},
    pages = {3834-3846},
    year = {2021},
    month = {11},
    issn = {1054-3139},
    doi = {10.1093/icesjms/fsab233}
}
```

The dataset belonging to this repository can be found at https://doi.org/10.4121/16622566.v1. A small [sample dataset](https://drive.google.com/file/d/1TcyeeX0UjhWldbjhLkCRJIuktDNeAMJJ/view?usp=sharing) is available for quickly testing this repository.

## Funding
The study was carried out under the Fully Documented Fisheries project initiated by the Dutch Ministry of Agriculture, Nature and Food Quality and funded by the European Maritime and Fisheries Fund.

<img src="https://github.com/user-attachments/assets/45e9ea07-def7-407a-9e86-6a30f0d315df" alt="wageningen university logo" height="50">
