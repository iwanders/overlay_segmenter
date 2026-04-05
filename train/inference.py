#!/usr/bin/env python3
#
import argparse
import time
from pathlib import Path

import torch
import torchvision
from torch import Tensor

from dataset_generator import load_image_file
from model import Unet

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    best_device = torch.device("cuda:0")  # or "cuda" for the current device
else:
    print("No GPU available. Training will run on CPU.")
    best_device = torch.device("cpu")


def load_model(path: Path, device=best_device) -> Unet:
    model = Unet(channels_in=3, channels_out=2)

    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def tiled_inference(
    model: Unet, image: Tensor, tile_size=256, device=best_device
) -> Tensor:
    if image.shape[0] == 4:
        image = image[0:3, :, :]

    image_dimensions = (image.shape[1], image.shape[2])

    # Okay, now we have an image in full resolution.
    size = tile_size
    step = size
    tiles = image.unfold(1, size, step).unfold(2, size, size)

    print(tiles.shape)
    tiles = tiles.permute([1, 2, 0, 3, 4])
    print(tiles.shape)
    # Nowe we have channel x vertical_i x horizontal_i x tile_size x tile_size

    masked = torch.zeros(
        (tiles.shape[0], tiles.shape[1], 2, tiles.shape[3], tiles.shape[4]),
        dtype=torch.float,
        device=device,
    )
    print("masked.shape", masked.shape)
    print("tiles.shape", tiles.shape)

    for w in range(tiles.shape[1]):
        with torch.no_grad():
            masked[:, w, :, :, :] = model(tiles[:, w, :, :, :])

    just_tiled_section_size = (
        2,
        masked.shape[0] * tile_size,
        masked.shape[1] * tile_size,
    )
    original_image_size = (2, image_dimensions[0], image_dimensions[1])
    mask_image = torch.zeros(
        original_image_size,
        dtype=torch.float,
        device=device,
    )

    for y in range(masked.shape[0]):
        for x in range(masked.shape[1]):
            mask_image[
                :,
                y * tile_size : (y + 1) * tile_size,
                x * tile_size : (x + 1) * tile_size,
            ] = masked[y, x, :, :, :]

    return mask_image


def write_network_output(mask_image: Tensor, directory: Path, name_prefix: str):
    mask_img = directory / f"{name_prefix}_mask.png"
    index_mask = mask_image.argmax(0)
    torchvision.utils.save_image(index_mask.to(torch.float), mask_img)
    # print(f"index_mask: {index_mask.shape}", index_mask)
    values_img = directory / f"{name_prefix}_values.png"
    t = mask_image[1, :, :]
    span = t.max() - t.min()
    t = (t - t.min()) / span
    torchvision.utils.save_image(t.to(torch.float), values_img)


def run_inference(args):
    model = load_model(args.checkpoint)
    out_dir = args.output
    print(args)
    for f in args.input:
        if "_mask.png" in str(f) or "_values.png" in str(f):
            print(f"Ignoring {f} because it looks like our output")
            continue
        s = time.time()
        image = load_image_file(f, device=best_device)
        masked = tiled_inference(model, image, device=best_device)
        name_prefix = Path(f).stem
        print(f"Done {name_prefix} in {time.time() - s} seconds")
        write_network_output(masked, directory=out_dir, name_prefix=name_prefix)


if __name__ == "__main__":
    print(f"Using device: {best_device}")

    # 1. Main parser
    parser = argparse.ArgumentParser(prog="inference")
    parser.add_argument("checkpoint", type=Path)
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # 2. 'login' subcommand
    parser_inference = subparsers.add_parser("inference", help="Login to the service")
    parser_inference.add_argument("--output", type=Path, default=Path("/tmp/"))
    parser_inference.add_argument("input", type=Path, nargs="+")
    parser_inference.set_defaults(func=run_inference)

    args = parser.parse_args()

    # Execute the selected command's function
    if args.command:
        args.func(args)
    else:
        parser.print_help()
