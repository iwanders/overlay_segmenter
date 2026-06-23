#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from dataset_generator import load_image_file
from torch import Tensor

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    best_device = torch.device("cuda:0")  # or "cuda" for the current device
else:
    print("No GPU available. Training will run on CPU.")
    best_device = torch.device("cpu")


if __name__ == "__main__":
    masked = load_image_file(sys.argv[1], device="cpu") / 255.0

    for inputname in sys.argv[2:]:
        print()
        print(inputname)
        segment_file = Path(inputname)
        segment = load_image_file(inputname, device="cpu")

        # print("masked shape", masked.shape)
        # print("segment shape", segment.shape, segment.max())

        image = masked.unsqueeze(dim=0)
        kernel = (
            (segment[3, :, :] > 0.5).to(torch.float).unsqueeze(dim=0).unsqueeze(dim=0)
        )
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
            output, f"/tmp/output_{segment_file.name}.png", normalize=True
        )

        print("max_index", max_index, "max value", output.ravel()[max_index])
        # 5. Convert flattened index to (Batch, Channel, Y, X) coordinates
        output_size = output.shape[2:]
        y_match = max_index // output_size[1]
        x_match = max_index % output_size[1]

        top_numbers = 50
        values, indices = torch.topk(output.flatten(), k=top_numbers)
        for vi, i in enumerate(indices):
            y_match = i // output_size[1]
            x_match = i % output_size[1]
            print([x_match.item(), y_match.item()], i.item(), values[vi].item())

        # raveled = output.ravel()
        # print("around", raveled[max_index - 10 : max_index + 10])

        print(
            f"Best match found at spatial coordinates (Y, X): ({y_match.item()}, {x_match.item()})"
        )
