#!/usr/bin/env python3
#
import sys
from pathlib import Path

import torch
import torchvision

from dataset_generator import load_image_file
from model import Unet

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    device = torch.device("cuda:0")  # or "cuda" for the current device
else:
    print("No GPU available. Training will run on CPU.")
    device = torch.device("cpu")

print(f"Using device: {device}")

tile_size = 256

model = Unet(channels_in=3, channels_out=2)

state_dict = torch.load(sys.argv[1])
model.load_state_dict(state_dict)
model.to(device)


image = load_image_file(sys.argv[2], device=device)
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


mask_image = torch.zeros(
    (2, masked.shape[0] * tile_size, masked.shape[1] * tile_size),
    dtype=torch.float,
    device=device,
)
for y in range(masked.shape[0]):
    for x in range(masked.shape[1]):
        mask_image[
            :, y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size
        ] = masked[y, x, :, :, :]

mask_img = "/tmp/eval_mask.png"
index_mask = mask_image.argmax(0)
torchvision.utils.save_image(index_mask.to(torch.float), mask_img)
# print(f"index_mask: {index_mask.shape}", index_mask)
values_img = "/tmp/eval_values.png"
t = mask_image[1, :, :]
span = t.max() - t.min()
t = (t - t.min()) / span
torchvision.utils.save_image(t.to(torch.float), values_img)
