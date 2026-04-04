#!/usr/bin/env python3
#

import concurrent.futures
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import Tensor
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms import ToTensor


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


def alpha_blend(fg, bg, alpha):
    """
    Blends foreground and background using an alpha mask.
    All tensors should be in [0, 1] range.
    fg: (C, H, W) or (N, C, H, W)
    bg: (C, H, W) or (N, C, H, W)
    alpha: (1, H, W) or (N, 1, H, W)
    """
    # Formula: BG * (1 - alpha) + FG * alpha
    # Or simplified: bg + alpha * (fg - bg)
    return bg + alpha * (fg - bg)


def augment_jpg_roundtrip(img, quality=20):
    desired_device = img.device
    # print(desired_device)
    # Go from floats to u8
    img = (img * 255.0).to(dtype=torch.uint8).to("cpu")
    # print(img)
    encoded = encode_jpeg(img, quality=quality)
    # print(encoded)
    # print(desired_device)
    return (decode_jpeg(encoded)).to(dtype=torch.float, device=desired_device) / 255.0


def rng_choice(rng, container):
    i = rng.integers(0, high=len(container))
    return container[i]


class ImageLoader:
    def __init__(
        self,
        image_dir: Path,
        crop_top_left: None | tuple[int, int] = None,
        crop_size: None | tuple[int, int] = None,
        device="cpu",
        remove_alpha=False,
    ):
        self._crop_top_left = crop_top_left
        print(self._crop_top_left)
        self._crop_size = crop_size
        self._images = []
        self._device = device
        self._image_dir: Path = image_dir
        self._remove_alpha = remove_alpha
        self.load_images()

    @staticmethod
    def background_loader(image_dir, **kwargs):
        if "crop_top_left" not in kwargs:
            kwargs["crop_top_left"] = (105, 27)
        if "crop_size" not in kwargs:
            kwargs["crop_size"] = (1700, 825)
        if "remove_alpha" not in kwargs:
            kwargs["remove_alpha"] = True
        v = ImageLoader(image_dir=image_dir, **kwargs)
        return v

    @staticmethod
    def foreground_loader(image_dir, **kwargs):
        v = ImageLoader(image_dir=image_dir, **kwargs)
        return v

    def load_image_file(self, d):
        image = Image.open(d)
        image = ToTensor()(image)
        image = image.to(self._device)
        # print("load image", type(image))
        return image

    def load_image(self, d):
        image = self.load_image_file(d)
        left, top = (0, 0) if self._crop_top_left is None else self._crop_top_left
        width, height = (
            (image.shape[1] - left, image.shape[2] - top)
            if self._crop_size is None
            else self._crop_size
        )
        bottom = top + height
        right = left + width
        image = image[
            :,
            top:bottom,
            left:right,
        ]
        # print("load load_background_image", type(image))
        # Background images may have an alpha channel, but we don't want that.
        if self._remove_alpha:
            if image.shape[0] == 4:
                image = image[0:3, :, :]
        return image

    def load_images(self):
        for f in self._image_dir.rglob("*.png"):
            background_image = self.load_image(f)
            self._images.append(background_image)

    def images(self) -> list[Tensor]:
        return self._images


class DatasetGenerator:
    def __init__(
        self,
        background_dir=None,
        foreground_dir=None,
        limit=float("inf"),
        device="cpu",
        batch_size=None,
        batch_count=None,
    ):
        self._limit = limit
        self._device = device
        self._batch_size = batch_size
        self._batch_count = batch_count
        self._background_images = []
        self._foreground_images = []
        if background_dir and foreground_dir:
            self.load_images(
                foreground_dir=Path(foreground_dir), background_dir=Path(background_dir)
            )

    @staticmethod
    def combine(
        generators,
        **kwargs,
    ):
        n = DatasetGenerator(**kwargs)
        for g in generators:
            n._background_images.extend([t.to(n._device) for t in g._background_images])
            n._foreground_images.extend([t.to(n._device) for t in g._foreground_images])

        return n

    def debug_dump(self):
        output = Path("/tmp/debug_dump")
        output.mkdir(exist_ok=True)

        print("Generating")
        start = time.time()
        generated = [
            (
                augment_jpg_roundtrip(a),
                b,
            )
            for a, b in self.generate(count=10)
        ]
        print("done generating")
        print(f"took {time.time() - start} s")

        for i, img in enumerate(self._background_images):
            torchvision.utils.save_image(img, output / f"background_{i}.png")
        for i, img in enumerate(self._foreground_images):
            torchvision.utils.save_image(img, output / f"foreground_{i}.png")

        for i, (sample_img, sample_mask) in enumerate(generated):
            torchvision.utils.save_image(sample_img, output / f"sample_{i}_img.png")
            torchvision.utils.save_image(
                sample_mask.to(torch.float), output / f"sample_{i}_mask.png"
            )

    @staticmethod
    def sample_tile(img, tile_size, rng):
        # channels, width, height
        width = img.shape[1]
        height = img.shape[2]
        # Sample mostly from the center, but corners are possible.
        x = rng.normal(loc=(width / 2.0) - (tile_size[0] / 2), scale=width / 4.0)
        y = rng.normal(loc=(height / 2.0) - (tile_size[1] / 2), scale=height / 4.0)
        # x = (width / 2.0) - (tile_size[0] / 2)
        # y = (height / 2.0) - (tile_size[1] / 2)
        # Int cast and clamp x and y such that the range falls within the image.
        x = clamp(int(x), 0, width - tile_size[0])
        y = clamp(int(y), 0, height - tile_size[1])
        # print(x, y, width, height)

        return img[:, x : x + tile_size[0], y : y + tile_size[1]]
        # return img[:, y : y + tile_size[1], x : x + tile_size[0]]

    def generate(self, count=1, tile_size=(256, 256), seed=1, alpha_factor=1.0):
        results = []
        rng = np.random.default_rng(seed=seed)

        def create_tile(rng):
            bg = rng_choice(rng, self._background_images)
            fg = rng_choice(rng, self._foreground_images)
            # Next, sample a tile from this.
            bg_tile = DatasetGenerator.sample_tile(bg, tile_size=tile_size, rng=rng)
            fg_tile = DatasetGenerator.sample_tile(fg, tile_size=tile_size, rng=rng)

            fg_rgb = fg_tile[:3]
            fg_alpha = fg_tile[3:]  # (1, H, W)

            # Now, we perform the blit to create the combined texture....
            combined = alpha_blend(fg_rgb, bg_tile, alpha=fg_alpha * alpha_factor)
            mask = fg_alpha >= 0.5
            # combined = torch.from_numpy(combined)
            mask = mask.to(torch.int64).squeeze()
            # combined = augment_jpg_roundtrip(combined, quality=20)
            return (combined, mask)

        threaded = False
        if threaded:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                rngs = [np.random.default_rng(seed=seed + t) for t in range(count)]
                results = list(executor.map(create_tile, rngs))
        else:
            results = list(
                create_tile(r)
                for r in (np.random.default_rng(seed=seed + t) for t in range(count))
            )

        # results = [
        #    create_tile(z)
        #    for z in list(np.random.default_rng(seed=seed + t) for t in range(count))
        # ]
        return results

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def set_batch_count(self, batch_count):
        self._batch_count = batch_count

    def __iter__(self):
        def gen():
            for i in range(self._batch_count):
                g = self.generate(self._batch_size)
                d = torch.cat([z[0].unsqueeze(0) for z in g], dim=0)
                m = torch.cat([z[1].unsqueeze(0) for z in g], dim=0)
                yield (d, m)

        return gen()


if __name__ == "__main__":
    background_dir = "../../datasets/background/cave/"
    foreground_dir = "../../datasets/foreground/cave/"

    l = ImageLoader.background_loader(Path(background_dir))
    print(l.images())
    print(l.images()[0].shape)
    print(l)

    d = DatasetGenerator(
        background_dir, foreground_dir=foreground_dir, limit=2, device="cuda:0"
    )
    d.debug_dump()

    d.set_batch_size(4)

    # data <class 'list'>
    # inputs <class 'torch.Tensor'> torch.Size([4, 3, 256, 256])
    # labels <class 'torch.Tensor'> torch.Size([4, 256, 256])
    for i, data in enumerate(d):
        print("data", type(data))
        # Every data instance is an input + label pair
        inputs, labels = data
        print("inputs", type(inputs), inputs.shape)
        print("labels", type(labels), labels.shape)
