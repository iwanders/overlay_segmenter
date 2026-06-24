#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from dataset_generator import load_image_file
from torch import Tensor
from torchvision.utils import draw_bounding_boxes

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    best_device = torch.device("cuda:0")  # or "cuda" for the current device
else:
    print("No GPU available. Training will run on CPU.")
    best_device = torch.device("cpu")


def grow_mask(mask, iterations=1):
    """
    Dilates a binary mask by 1 pixel (or more) using PyTorch convolution.
    Assumes mask shape: (N, C, H, W) or (C, H, W)
    """
    # Ensure mask is float and has batch/channel dimensions if necessary
    is_1d = mask.ndim == 2
    if is_1d:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)

    orig_dtype = mask.dtype
    mask = mask.float()

    # 3x3 structuring element for dilation
    kernel = torch.ones((1, 1, 3, 3), device=mask.device)

    # Perform iterations
    for _ in range(iterations):
        # Pad by 1 to maintain original dimensions after convolution
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode="replicate")
        # Convolve and binarize (values > 0 become 1)
        mask = (F.conv2d(padded_mask, kernel, padding=0) > 0).float()

    # Revert to original dimensions/type
    if is_1d:
        return mask.squeeze(0).squeeze(0).to(orig_dtype)
    elif mask.shape[0] == 1:
        return mask.squeeze(0).to(orig_dtype)
    return mask.to(orig_dtype)


def create_distincting_kernel(all_segments, desired_segment) -> Tensor:
    # Ideally this is created on demand, based on which segments are possible in this position.
    # Take the OR of all the bitmasks.
    if all_segments:
        print("all_segments shape: ", torch.stack(all_segments).shape)
        max_values = torch.any(torch.stack(all_segments), dim=0)
    else:
        shape = desired_segment.shape
        max_values = torch.zeros(shape)

    max_values = max_values.to(torch.float)
    max_values = -1.0 * max_values
    print("max_values shape: ", max_values.shape)
    print("desired_segment shape: ", desired_segment.shape)

    # Grow the desired one by one pixel, because otherwise we may get some slight black borders next to the white
    # of the desired.
    # desired_segment = grow_mask(desired_segment)
    # huh, without gives a better distinction... with; 20.96 vs 12.5, without; 19.9 vs 10.15

    # what if there is a section we in this tile, that is not in any of the others?
    distinct = desired_segment.clone()
    for s in all_segments:
        if s.equal(desired_segment):
            continue
        distinct &= s
    distinct_max = distinct.max()
    if distinct_max.item() and all_segments:
        print("Would have distinct!", distinct_max)

    max_values[desired_segment] = desired_segment[desired_segment].to(torch.float)
    # max_values += distinct.to(torch.float) * 0.5
    return max_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="inference")

    parser.add_argument(
        "--segment-dir",
        type=Path,
        help="Directory where all the segments are, we glob in this.",
    )
    parser.add_argument("--segment", type=Path, help="Path to the desired segment")
    parser.add_argument(
        "mask", type=Path, help="Path to the mask on which to detect the segment."
    )

    args = parser.parse_args()
    segment_dirname = args.segment_dir.name

    all_segments = []
    for p in args.segment_dir.glob("*.png"):
        segment_file = Path(p)
        segment = load_image_file(p, device="cpu")
        kernel = segment[3, :, :] > 0.5
        all_segments.append(kernel)

    desired_segment = load_image_file(args.segment, device="cpu")
    desired_segment = desired_segment[3, :, :] > 0.5

    distincting_kernel = create_distincting_kernel(all_segments, desired_segment)
    torchvision.utils.save_image(
        distincting_kernel,
        f"/tmp/{args.segment.name}_distincting_kernel_d_{segment_dirname}.png",
        normalize=True,
    )
    distincting_kernel = distincting_kernel

    masked = load_image_file(args.mask, device="cpu") / 255.0
    shutil.copy(args.segment, f"/tmp/{args.segment.name}")

    # print("masked shape", masked.shape)
    # print("segment shape", segment.shape, segment.max())

    image = masked.unsqueeze(dim=0)
    kernel = distincting_kernel.unsqueeze(dim=0).unsqueeze(dim=0)
    # print(image.shape)
    # print(kernel.shape)

    # 2. Normalize the kernel for zero-mean and unit variance (for pattern matching)
    # kernel = kernel - kernel.mean()
    # kernel = kernel / kernel.norm()

    # 3. Perform the convolution
    # (stride controls step size, padding handles boundary drops)
    output = F.conv2d(image, kernel, stride=1, padding=0)
    # print("output.shape", output.shape)

    # 4. Find the position of the exact highest match
    max_index = torch.argmax(output)
    torchvision.utils.save_image(
        output,
        f"/tmp/{args.segment.name}_output_d_{segment_dirname}.png",
        normalize=True,
    )

    print("max_index", max_index, "max value", output.ravel()[max_index])
    # 5. Convert flattened index to (Batch, Channel, Y, X) coordinates
    output_size = output.shape[2:]
    y_match = max_index // output_size[1]
    x_match = max_index % output_size[1]

    peaks = []
    top_numbers = 300
    values, indices = torch.topk(output.flatten(), k=top_numbers)
    for vi, i in enumerate(indices):
        y_match = (i // output_size[1]).item()
        x_match = (i % output_size[1]).item()
        curpos = np.array([x_match, y_match])
        skip = False
        for ppos, _, _ in peaks:
            prevpos = ppos
            dist = np.linalg.norm((curpos - prevpos))
            if dist < 20:
                skip = True
                break
        if not skip:
            peaks.append((curpos, i.item(), values[vi].item()))
    print("\n".join(f"{p}: {h}" for p, _, h in peaks))

    # Lets also output a version with some labels, just for funsies.

    single_channel = output.squeeze()
    print("single_channel shape", single_channel.shape)
    rgb_output = torch.stack([single_channel, single_channel, single_channel], dim=0)
    print("rgb_output shape", rgb_output.shape)

    # Get min and max values
    tensor_min = rgb_output.min()
    tensor_max = rgb_output.max()

    # Min-Max Scale to 0-1, then multiply by 255
    tensor_0_255 = ((rgb_output - tensor_min) / (tensor_max - tensor_min) * 255).to(
        torch.uint8
    )

    box_size = distincting_kernel.shape
    left = -box_size[1] / 2
    right = box_size[1] / 2
    bottom = -box_size[0] / 2
    top = box_size[0] / 2

    colors = {0: "red", 1: "yellow", 2: "green"}

    # In reverse order, so best is on top.
    for i, (pos, _ind, value) in enumerate(peaks[::-1]):
        i = len(peaks) - i - 1
        x = int(pos[0])
        y = int(pos[1])
        # # 2. Define boxes (xmin, ymin, xmax, ymax) and corresponding labels
        boxes = torch.tensor([[x + left, y + bottom, x + right, y + top]])
        labels = [f"{pos}: {value: >4.4g}"]

        # 3. Draw boxes and overlay text
        tensor_0_255 = draw_bounding_boxes(
            image=tensor_0_255,
            boxes=boxes,
            labels=labels,
            colors=colors.get(i, "gray"),
            #  font_size=20,
        )
    torchvision.utils.save_image(
        tensor_0_255 / 255.0,
        f"/tmp/{args.segment.name}_output_d_{segment_dirname}_labelled.png",
        normalize=False,
    )

    # raveled = output.ravel()
    # print("around", raveled[max_index - 10 : max_index + 10])
