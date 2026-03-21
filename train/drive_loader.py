#!/usr/bin/env python3
# Loader for the original DRIVE dataset.
import os
from collections import namedtuple
from doctest import testfile
from pathlib import Path

import torch

# import torchvision

DRIVE_DIR = Path(os.environ.get("DRIVE_DIR", "../../datasets/DRIVE/"))
"""
test
    images
        01_test.tif
    mask
        01_test_mask.gif
    1st_manual:
        01_manual1.gif
    2nd_manual:
        01_manual2.gif
training
    images
        21_training.tif
    mask
        21_training_mask.gif
    1st_manual
        21_manual1.gif
"""


DriveImage = namedtuple("DriveImage", ["image", "image_mask", "manual1", "manual2"])


def load_tif(d):
    from PIL import Image
    from torchvision.transforms import ToTensor

    image = Image.open(d)
    image = ToTensor()(image)
    return image


def load_gif(d):
    from PIL import Image
    from torchvision.transforms import ToTensor

    image = Image.open(d)
    image = ToTensor()(image)
    return image


def load_drive_dataset():
    # torchvision.io.decode_image(input: Union[Tensor, str], mode: ImageReadMode = ImageReadMode.UNCHANGED, apply_exif_orientation: bool = False) → Tensor
    def load_dir(d):
        accumulated = []
        image_dir = DRIVE_DIR / d / "images"
        for im in image_dir.iterdir():
            basename = im.stem
            image = load_tif(im)
            mask_path = DRIVE_DIR / d / "mask" / f"{basename}_mask.gif"
            mask = load_gif(mask_path)
            manual1_path = DRIVE_DIR / d / "1st_manual" / f"{basename[0:2]}_manual1.gif"
            manual1 = load_gif(manual1_path)
            manual2_path = DRIVE_DIR / d / "2nd_manual" / f"{basename[0:2]}_manual2.gif"
            manual2 = None
            if manual2_path.is_file():
                manual2 = load_gif(manual2_path)
            accumulated.append(
                DriveImage(
                    image=image, image_mask=mask, manual1=manual1, manual2=manual2
                )
            )
        return accumulated

    train = load_dir("training")
    test = load_dir("test")
    return train, test


if __name__ == "__main__":
    train, test = load_drive_dataset()
    # print(test)
