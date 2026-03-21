#!/usr/bin/env python3
# Loader for the original DRIVE dataset.
import os
from collections import namedtuple
from doctest import testfile
from pathlib import Path

import torch
import torch.nn.functional as F

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


def load_image(d):
    from PIL import Image
    from torchvision.transforms import ToTensor

    image = Image.open(d)
    image = ToTensor()(image)
    # Get center 512 by 512 pixels.
    image = image[:, 0:512, 0:512]  # Crop to center.
    return image


# Assuming mask is a tensor of shape (H, W)
def target_preprocess(mask):
    return mask
    one_hot_mask = mask.to(torch.int64)
    return one_hot_mask


def load_drive_dataset(device="cpu"):
    def load_dir(d):
        accumulated = []
        image_dir = DRIVE_DIR / d / "images"
        for im in image_dir.iterdir():
            basename = im.stem
            image = load_image(im).to(device)
            mask_path = DRIVE_DIR / d / "mask" / f"{basename}_mask.gif"
            mask = load_image(mask_path).to(device)
            manual1_path = DRIVE_DIR / d / "1st_manual" / f"{basename[0:2]}_manual1.gif"
            # manual1 = torch.round(load_image(manual1_path).to(device)).int()
            manual1 = target_preprocess(load_image(manual1_path).to(device))
            manual2_path = DRIVE_DIR / d / "2nd_manual" / f"{basename[0:2]}_manual2.gif"
            manual2 = None
            if manual2_path.is_file():
                # manual2 = torch.round(load_image(manual2_path).to(device)).int()
                manual2 = target_preprocess(load_image(manual2_path).to(device))
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
