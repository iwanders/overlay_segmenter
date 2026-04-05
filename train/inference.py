#!/usr/bin/env python3
#
import argparse
import math
import time
from itertools import batched
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


class TileCutter:
    def __init__(
        self,
        input_shape: tuple[int, int],
        tile_size=256,
        overlap: int = 32,
    ):
        self._tile_size = tile_size  # in H, W!
        self._overlap = overlap
        self._input_shape = input_shape
        self._actual_size = self._tile_size - 2 * self._overlap

    # Input is [C, H, W], output is [B, C, tile_size,tile_size]
    def split(self, input_tensor: Tensor) -> Tensor:
        pad_value = self._tile_size
        p = torchvision.transforms.Pad(pad_value, fill=0, padding_mode="reflect")

        padded = p(input_tensor)

        height = input_tensor.shape[1]
        width = input_tensor.shape[2]
        channels = input_tensor.shape[0]
        assert (height, width) == self._input_shape

        x_max = math.ceil(width / self._actual_size)
        y_max = math.ceil(height / self._actual_size)

        tiles = torch.zeros(
            (x_max * y_max, channels, self._tile_size, self._tile_size),
            dtype=torch.float,
            device=input_tensor.device,
        )
        o = self._overlap
        for y in range(y_max):
            for x in range(x_max):
                tile_index = y * x_max + x
                as_x = x * self._actual_size + pad_value
                as_y = y * self._actual_size + pad_value
                as_x_end = (x + 1) * self._actual_size + pad_value
                as_y_end = (y + 1) * self._actual_size + pad_value
                tiles[tile_index, :, :, :] = padded[
                    :, (as_y - o) : (as_y_end + o), (as_x - o) : (as_x_end + o)
                ]
        return tiles

    # Input is  [B, C, tile_size,tile_size], output is [C, H, W]
    def merge(self, tiles: Tensor) -> Tensor:
        height, width = self._input_shape
        x_max = math.ceil(width / self._actual_size)
        y_max = math.ceil(height / self._actual_size)
        channels = tiles[0].shape[0]
        pad_value = self._tile_size
        merged = torch.zeros(
            (channels, height + pad_value, width + pad_value),
            dtype=torch.float,
            device=tiles[0].device,
        )
        o = self._overlap
        for y in range(y_max):
            for x in range(x_max):
                tile_index = y * x_max + x
                as_x = x * self._actual_size
                as_y = y * self._actual_size
                as_x_end = (x + 1) * self._actual_size
                as_y_end = (y + 1) * self._actual_size
                merged[:, as_y:as_y_end, as_x:as_x_end] = tiles[
                    tile_index, :, o:-o, o:-o
                ]
        merged = merged[:, 0:height, 0:width].detach().clone()
        return merged

    def debug_dump_batch(self, tiles: Tensor, output="/tmp/my_batch.png"):
        height, width = self._input_shape
        x_max = math.ceil(width / self._actual_size)
        # y_max = math.ceil(height / self._actual_size)
        torchvision.utils.save_image(tiles, output, nrow=x_max, normalize=True)


def run_test(args):

    dummy = torch.zeros(
        (3, 300, 300),
        dtype=torch.float,
        device="cpu",
    )
    dummy = load_image_file("/tmp/Screenshot738.png", device="cpu")
    cutter = TileCutter(input_shape=dummy.shape[1:])
    tiles = cutter.split(dummy)
    print(tiles.shape)
    torchvision.utils.save_image(tiles, "/tmp/my_batch.png", nrow=10, normalize=True)

    merged = cutter.merge(tiles)
    torchvision.utils.save_image(merged, "/tmp/merged.png", normalize=True)


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

    # print(tiles.shape)
    tiles = tiles.permute([1, 2, 0, 3, 4])
    # print(tiles.shape)
    # Nowe we have channel x vertical_i x horizontal_i x tile_size x tile_size

    masked = torch.zeros(
        (tiles.shape[0], tiles.shape[1], 2, tiles.shape[3], tiles.shape[4]),
        dtype=torch.float,
        device=device,
    )
    # print("masked.shape", masked.shape)
    # print("tiles.shape", tiles.shape)

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


def batched_inference(model: Unet, tiles: Tensor, batch_size=10) -> Tensor:
    output_size = (
        tiles.shape[0],
        model.channels_out(),
        tiles.shape[2],
        tiles.shape[3],
    )
    # Output buffer
    output_data = torch.zeros(
        output_size,
        dtype=torch.float,
        device=tiles.device,
    )
    for w in batched(range(tiles.shape[0]), batch_size):
        with torch.no_grad():
            output_data[w, :, :, :] = model(tiles[w, :, :, :])
    return output_data


def run_inference(args):
    model = load_model(args.checkpoint)
    out_dir = args.output
    print(args)
    for f in args.input:
        if "_mask.png" in str(f) or "_values.png" in str(f) or "_batch.png" in str(f):
            print(f"Ignoring {f} because it looks like our output")
            continue
        s = time.time()
        image = load_image_file(f, device=best_device)
        if image.shape[0] == 4:
            image = image[0:3, :, :]

        time_start = time.time()
        if True:
            # This one takes 0.003677845001220703 for subsequent calls.
            masked = tiled_inference(model, image, device=best_device)
        else:
            # This one takes 0.57 for subsequent calls O_o
            cutter = TileCutter(image.shape[1:], overlap=16)
            tiles = cutter.split(image)
            batch_masks = batched_inference(model, tiles)
            masked = cutter.merge(batch_masks)

        time_end = time.time()

        name_prefix = Path(f).stem
        batch_path = out_dir / f"{name_prefix}_batch.png"
        # cutter.debug_dump_batch(tiles, batch_path)
        print(f"Done {name_prefix} in {time_end - time_start} seconds")
        write_network_output(masked, directory=out_dir, name_prefix=name_prefix)


if __name__ == "__main__":
    # ./inference.py /tmp/train/099/model.pth  inference  /tmp/Screenshot*
    print(f"Using device: {best_device}")

    parser = argparse.ArgumentParser(prog="inference")
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    parser_inference = subparsers.add_parser("inference", help="Run inference")
    parser_inference.add_argument("-c", "--checkpoint", type=Path, required=True)
    parser_inference.add_argument("--output", type=Path, default=Path("/tmp/"))
    parser_inference.add_argument("input", type=Path, nargs="+")
    parser_inference.set_defaults(func=run_inference)

    parser_test = subparsers.add_parser("test", help="Run inference")
    parser_test.set_defaults(func=run_test)

    args = parser.parse_args()

    # Execute the selected command's function
    if args.command:
        args.func(args)
    else:
        parser.print_help()
