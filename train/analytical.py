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

    counts = counts.to(torch.int64)
    unique_colors = unique_colors.to(torch.int64)

    unique_colors = unique_colors[indices]
    # rgb | count
    histogram = torch.hstack([unique_colors, counts.unsqueeze(1)])

    return histogram


# Slow, but strictly equivalent.
def extract_alpha_mask_integer(
    composite: Tensor, background_color: Tensor, foreground_color: Tensor
) -> Tensor:
    """
    Extract alpha mask using integer arithmetic and brute-force search.
    This is the most robust method for uint8 images.

    composite: (H, W, 3) uint8 image
    background_color: (3,) uint8 background color
    foreground_color: (3,) uint8 foreground color

    Returns: (H, W) uint8 alpha mask
    """
    H, W, _ = composite.shape

    # Precompute differences to speed up the loop
    C = composite.to(torch.int64)  # Use int64 to avoid overflow
    B = background_color.to(torch.int64)
    F = foreground_color.to(torch.int64)

    best_alpha = torch.zeros((H, W), dtype=torch.uint8)

    for alpha in range(256):
        # Compute the expected composite for this alpha value
        # Formula: C = round((F * alpha + B * (255 - alpha)) / 255)
        # To avoid floating point, use: (F * alpha + B * (255 - alpha) + 128) // 255

        # Vectorized computation for all pixels
        numerator = F[:] * alpha + B[:] * (255 - alpha)
        numerator = numerator.reshape((1, 1, 3))
        # Add 128 for rounding before integer division
        expected_C = (numerator + 128) // 255

        # Compute squared error for all pixels
        error = torch.sum(
            (C - expected_C) ** 2, dim=2, keepdim=False
        )  # Sum over R, G, B

        # For this alpha, find which pixels have minimum error
        # We'll track the best alpha per pixel

        # Initialize on first iteration
        if alpha == 0:
            min_error = error
        else:
            # Update where this alpha has lower error
            update_mask = error < min_error
            min_error[update_mask] = error[update_mask]
            best_alpha[update_mask] = alpha

    return best_alpha


def run_extract(args):
    device = "cpu"
    for f in args.input_files:
        basename = f.stem
        img = load_image_file_u8(f, device=device)
        # 3, h, w, or 4, h, w
        rgb = img[0:3, :, :]

        histogram = color_histogram(rgb)
        background = histogram[args.background_color_index, :]
        foreground = histogram[args.foreground_color_index, :]
        print(
            f"{f.name: <40} background {background.tolist()[:3]} ({swatch(background.tolist()[:3])}) with {background.tolist()[3]}  values, foreground: {foreground.tolist()[:3]}  ({swatch(foreground.tolist()[:3])})  with {foreground.tolist()[3]} values"
        )

        background_color = background[:3].unsqueeze(1)
        foreground_color = foreground[:3].unsqueeze(1)
        rgb_by_pixel = rgb.permute([1, 2, 0])

        alpha_mask = extract_alpha_mask_integer(
            composite=rgb_by_pixel,
            background_color=background_color,
            foreground_color=foreground_color,
        )

        # Next, store the mask.
        alpha_mask = alpha_mask.unsqueeze(0)
        torchvision.io.write_png(alpha_mask, args.output / f"{basename}_alpha.png")

        # And lets also craft the foreground with the color.
        color_tensor = foreground_color.detach().clone().view(1, 1, 3).to(torch.uint8)
        height, width, _ = rgb_by_pixel.shape
        solid_foreground = color_tensor.expand(height, width, 3).clone()

        alpha_mask = alpha_mask.permute([1, 2, 0])
        # foreground_grba = torch.dstack([solid_foreground, alpha_mask])
        # print(foreground_grba.shape)
        # foreground_grba = foreground_grba.permute([2, 0, 1])
        # print(foreground_grba.shape)
        # torchvision.io.write_png(
        #    foreground_grba, args.output / f"{basename}_foreground.png"
        # )
        # seriously, that doesn't support alpha... :(
        #
        foreground_by_pixel = solid_foreground.permute([2, 0, 1])
        alpha_mask = alpha_mask.permute([2, 0, 1])

        pil_fg = torchvision.transforms.functional.to_pil_image(
            foreground_by_pixel, mode=None
        )
        pil_fg.putalpha(torchvision.transforms.functional.to_pil_image(alpha_mask))
        pil_fg.save(args.output / f"{basename}_foreground.png")

        # and lets generate the composite
        color_tensor = background_color.detach().clone().view(1, 1, 3).to(torch.uint8)
        solid_background = color_tensor.expand(height, width, 3).clone()
        background_by_pixel = solid_background.permute([2, 0, 1])
        background_img = torchvision.transforms.functional.to_pil_image(
            background_by_pixel, mode=None
        )
        background_img.paste(pil_fg, (0, 0), pil_fg)
        background_img.save(args.output / f"{basename}_recreated.png")

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
