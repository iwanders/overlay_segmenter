#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
import torchvision
from torch import Tensor

from util import load_image_file_u8

"""

This extracts an alpha-blended foreground from a solid background.
To do so, we assume that the second-most dominant color is the fully opaque overlay color.

# Alpha blend composition:
    R_c = R_f * A_f + R_b * (1 - A_f)
    G_c = G_f * A_f + G_b * (1 - A_f)
    B_c = B_f * A_f + B_b * (1 - A_f)

# Solve for alpha blend factor:
    A_f_r = (R_c - R_b) / (R_f - R_b)
    Which requires R_f, but that is something we can determine using the histogram of uniquely seen colors.
"""


def swatch(*args) -> str:
    if len(args) == 3:
        r, g, b = args
    if len(args) == 1:
        r, g, b = args[0]
    # foreground color
    return f"\033[38;2;{r};{g};{b}m██\033[0m"


def color_histogram(image_tensor: Tensor) -> Tensor:

    # Assuming image_tensor is of shape (C, H, W) and values [0, 255]
    # 1. Reshape to (H * W, C) -> e.g., (Pixels, 3)
    pixels = image_tensor.view(image_tensor.shape[0], -1).t()

    # 2. Find unique colors and their corresponding counts
    unique_colors, counts = torch.unique(pixels, dim=0, return_counts=True)

    # Sort colors by frequency to get the most dominant colors first
    counts, indices = torch.sort(counts, descending=True)

    print(unique_colors.shape, counts.shape)
    counts = counts.to(torch.int64)
    unique_colors = unique_colors.to(torch.int64)

    unique_colors = unique_colors[indices]
    # rgb | count
    histogram = torch.hstack([unique_colors, counts.unsqueeze(1)])

    return histogram


def run_extract(args):
    device = "cpu"
    for f in args.input_files:
        img = load_image_file_u8(f, device=device)
        # 3, h, w, or 4, h, w
        rgb = img[0:3, :, :]

        histogram = color_histogram(rgb)
        background = histogram[args.background_color_index, :]
        foreground = histogram[args.foreground_color_index, :]
        print(
            f"Background {background.tolist()[:3]} ({swatch(background.tolist()[:3])}) with {background.tolist()[3]}  values, foreground: {foreground.tolist()[:3]}  ({swatch(foreground.tolist()[:3])})  with {foreground.tolist()[3]} values"
        )
        print(rgb.shape)

        rgb_f = rgb.to(torch.float) / 255.0
        print(rgb_f.shape)
        # Output: (fg * alpha + bg * (255 - alpha)) / 255
        # This must fit in u16 space.

    pass


if __name__ == "__main__":
    # ./analytical.py extract ../../media/2026_07_04_analytical/Screenshot043_topright_300x150.png
    parser = argparse.ArgumentParser(prog="inference")
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    parser_extract = subparsers.add_parser(
        "extract", help="extract overlay from a solid background."
    )
    parser_extract.add_argument(
        "-b",
        "--background-color-index",
        default=0,
        type=int,
        help="Peak in the unique color histogram to use as the backgorund pixel",
    )
    parser_extract.add_argument(
        "-f",
        "--foreground-color-index",
        default=1,
        type=int,
        help="Peak in the unique color histogram to use as the foreground pixel",
    )

    _ = parser_extract.add_argument("input_files", type=Path, nargs="*")
    _ = parser_extract.add_argument(
        "-o", "--output", type=Path, required=False, default="/tmp/"
    )

    parser_extract.set_defaults(func=run_extract)

    args = parser.parse_args()

    # Execute the selected command's function
    if args.command:
        args.func(args)
    else:
        parser.print_help()
