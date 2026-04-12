#!/usr/bin/env python3
#

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms import ToTensor

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    preferred_device = torch.device("cuda:0")  # or "cuda" for the current device
else:
    print("No GPU available. Training will run on CPU.")
    preferred_device = torch.device("cpu")


def lookup_device(input_str) -> torch.device:
    if input_str == "auto":
        return preferred_device

    return input_str


class TensorNameTracker:
    def __init__(self):
        self._names = {}

    def set_name(self, t: Tensor, name: str):
        self._names[id(t)] = name

    def get_name(self, t: Tensor):
        return self._names.get(id(t))


tensor_tracker = TensorNameTracker()


def load_paths(path_file):
    with open(path_file) as f:
        return [a.strip() for a in f.readlines()]


class DataPair(BaseModel):
    foreground_subdir: list[str]
    background_subdir: list[str]


class DataGenerationSpec(BaseModel):
    background_dir: str
    foreground_dir: str
    data_pair: list[DataPair]


class CollectionPair(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    foreground: list[Tensor]
    background: list[Tensor]


class DataLoader:
    def __init__(self, config_file):
        with open(config_file) as f:
            d = yaml.safe_load(f)
        self._spec = DataGenerationSpec.model_validate(d)
        self._bg_images: dict[str, list[Tensor]] = {}
        self._fg_images: dict[str, list[Tensor]] = {}
        self.load_images()

    def load_images(self):
        bg_dir = Path(self._spec.background_dir)
        fg_dir = Path(self._spec.foreground_dir)

        def load_datapair(data_pairs):
            for fg_subdir in data_pairs.foreground_subdir:
                if fg_subdir not in self._fg_images:
                    self._fg_images[fg_subdir] = ImageLoader.foreground_loader(
                        fg_dir / fg_subdir
                    ).images()

            for bg_subdir in data_pairs.background_subdir:
                if bg_subdir not in self._bg_images:
                    self._bg_images[bg_subdir] = ImageLoader.background_loader(
                        bg_dir / bg_subdir
                    ).images()

        for data_pair in self._spec.data_pair:
            load_datapair(data_pair)

    def generate_data_pairs(self) -> list[CollectionPair]:
        # This is where we actually make the collection that can be trained on.
        r = []
        for data_pairs in self._spec.data_pair:
            foreground = []
            background = []
            for fg_subdir in data_pairs.foreground_subdir:
                foreground.extend(self._fg_images[fg_subdir])
            for bg_subdir in data_pairs.background_subdir:
                background.extend(self._bg_images[bg_subdir])
            p = CollectionPair(foreground=foreground, background=background)
            r.append(p)

        return r


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


def augment_jpg_roundtrip(img, quality=50):
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


def rng_shuffle(rng, container):
    shuffled_i = list(range(len(container)))
    rng.shuffle(shuffled_i)
    return [container[i] for i in shuffled_i]


def load_image_file(d, device):
    image = Image.open(d)
    image = ToTensor()(image)
    image = image.to(device)
    # print("load image", type(image))
    return image


class ImageLoader:
    def __init__(
        self,
        crop_top_left: None | tuple[int, int] = None,
        crop_size: None | tuple[int, int] = None,
        device="cpu",
        remove_alpha=False,
    ):
        self._crop_top_left = crop_top_left
        self._crop_size = crop_size
        self._images = []
        self._device = device
        self._remove_alpha = remove_alpha

    @staticmethod
    def background_loader(image_dir, **kwargs):
        if "crop_top_left" not in kwargs:
            kwargs["crop_top_left"] = (105, 27)
        if "crop_size" not in kwargs:
            kwargs["crop_size"] = (1700, 825)
        if "remove_alpha" not in kwargs:
            kwargs["remove_alpha"] = True
        v = ImageLoader(**kwargs)
        v.load_images(image_dir)
        return v

    @staticmethod
    def foreground_loader(image_dir, **kwargs):
        v = ImageLoader(**kwargs)
        v.load_images(image_dir)
        return v

    def load_image(self, d):
        image = load_image_file(d, device=self._device)
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

    def load_images(self, image_dir: Path):
        to_load = list(image_dir.rglob("*.png"))

        def load_img(f):
            img = self.load_image(f)
            filename = f.stem
            tensor_tracker.set_name(img, filename)
            return f, img

        with ThreadPoolExecutor() as executor:
            res = list(executor.map(load_img, to_load))
            self._images.extend([img for _, img in sorted(res)])

    def images(self) -> list[Tensor]:
        return self._images


class DatasetGenerator:
    """
        Main data set generator.

        Instead of background and foreground images and allowing freely mixing each, it now takes:
            data: list[CollectionPair]


    class CollectionPair(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        foreground: list[Tensor]
        background: list[Tensor]
    """

    def __init__(
        self,
        data_pairs: list[CollectionPair],
        device="cpu",
        batch_size: int | None = None,
        batch_count: int | None = None,
        rng: np.random.Generator | None = None,
        tile_size=(256, 256),
        alpha_factor=1.0,
    ):
        self._device = device
        self._batch_size: int | None = batch_size
        self._batch_count: int | None = batch_count

        # prepare data pairs for sampling with background first, second pair mapping to the foreground options.
        self._sample_entries: list[tuple[Tensor, list[Tensor]]] = []
        for data_pair in data_pairs:
            for bg in data_pair.background:
                if len(data_pair.foreground) == 0:
                    raise ValueError(
                        "Foreground count is zero for bg: ",
                        tensor_tracker.get_name(bg),
                    )
                self._sample_entries.append((bg, data_pair.foreground))
        self._rng = rng
        self._tile_size = tile_size
        self._alpha_factor = alpha_factor
        if self._sample_entries:
            self._sample_entries = rng_shuffle(self._rng, self._sample_entries)

    def split_out_validation(
        self,
        rng,
        ratio: float = 0.1,
    ) -> "DatasetGenerator":
        validation = DatasetGenerator([], rng=rng)
        validation._device = self._device
        validation._batch_size = self._batch_size
        validation._batch_count = self._batch_count
        validation._tile_size = self._tile_size
        validation._alpha_factor = self._alpha_factor
        total_bg = len(self._sample_entries)
        validation_bg_split = int(total_bg * ratio)
        validation_entries = self._sample_entries[0:validation_bg_split]
        train_entries = self._sample_entries[validation_bg_split + 1 :]
        validation._sample_entries = validation_entries
        self._sample_entries = train_entries
        return validation

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

        # for i, img in enumerate(self._background_images):
        #    torchvision.utils.save_image(img, output / f"background_{i}.png")
        # for i, img in enumerate(self._foreground_images):
        #    torchvision.utils.save_image(img, output / f"foreground_{i}.png")

        for i, (sample_img, sample_mask) in enumerate(generated):
            torchvision.utils.save_image(sample_img, output / f"sample_{i}_img.png")
            torchvision.utils.save_image(
                sample_mask.to(torch.float), output / f"sample_{i}_mask.png"
            )

    @staticmethod
    def sample_tile(img, tile_size, rng):
        # channels, height, width,
        height = img.shape[1]
        width = img.shape[2]

        if width < tile_size[1] or height < tile_size[0]:
            padder = torchvision.transforms.Pad(
                (max((tile_size[1] - width), 0), max((tile_size[0] - height), 0)),
                fill=0,
                padding_mode="constant",
            )
            img = padder(img)
            width = img.shape[1]
            height = img.shape[2]
        # Sample mostly from the center, but corners are possible.
        x = rng.normal(loc=(width / 2.0) - (tile_size[0] / 2), scale=width / 4.0)
        y = rng.normal(loc=(height / 2.0) - (tile_size[1] / 2), scale=height / 4.0)
        # kwargs["crop_size"] = (1700, 825)
        # x = (width / 2.0) - (tile_size[0] / 2)
        # y = (height / 2.0) - (tile_size[1] / 2)
        # Int cast and clamp x and y such that the range falls within the image.
        x = clamp(int(x), 0, width - tile_size[0])
        y = clamp(int(y), 0, height - tile_size[1])
        # print(x, y, width, height)

        return img[:, y : y + tile_size[1], x : x + tile_size[0]]

    @staticmethod
    def image_overlay(background, foreground, b_x, b_y, f_x, f_y) -> Tensor:
        # We've selected the position in the canvas, and the position in the overlay.
        # next, we have to determine the rectangle in which the bounds overlap.
        # We will place the overlay coordinate onto the canvas coordinate.

        # Calculate the overlapping region
        bg_h, bg_w = background.shape[1], background.shape[2]
        fg_h, fg_w = foreground.shape[1], foreground.shape[2]

        b_x = int(b_x - bg_w / 2)
        b_y = int(b_y - bg_h / 2)
        f_x = int(f_x - bg_w / 2)
        f_y = int(f_y - bg_h / 2)

        # x_offset and y_offset is the top left corner of the overlay in bg coordinates.
        x_offset = int(b_x - f_x)
        y_offset = int(b_y - f_y)

        # x_offset = 5
        # # y_offset = 15
        # print("x_offset: ", x_offset)
        # print("y_offset: ", y_offset)

        # Determine intersection coordinates (handles boundary crossing)
        y1 = max(0, y_offset)
        y2 = min(bg_h, y_offset + fg_h)
        x1 = max(0, x_offset)
        x2 = min(bg_w, x_offset + fg_w)

        # Corresponding coordinates in the foreground image
        fg_y1 = max(0, -y_offset)
        fg_y2 = fg_y1 + (y2 - y1)
        fg_x1 = max(0, -x_offset)
        fg_x2 = fg_x1 + (x2 - x1)

        # Handle two situations where the intersection is disjoint; ie; the overlay is outside of the bg.
        if y2 < y1 or x2 < x1:
            return background

        if fg_y2 < fg_y1 or fg_x2 < fg_x1:
            return background

        # Apply the overlay
        background[:, y1:y2, x1:x2] = foreground[:, fg_y1:fg_y2, fg_x1:fg_x2]
        return background
        pass

    @staticmethod
    def stamp_tile(
        rng,
        tile_size,
        overlay,
    ):
        # channels, height, width,
        # Sample mostly from the center, but corners are possible.
        o_height, o_width = overlay.shape[1:]
        c_height, c_width = tile_size
        scale_divisor = 6.0
        o_x = int(rng.normal(loc=o_width / 2, scale=o_width / scale_divisor))
        o_y = int(rng.normal(loc=o_height / 2, scale=o_height / scale_divisor))
        c_x = int(rng.normal(loc=c_width / 2, scale=c_width / scale_divisor))
        c_y = int(rng.normal(loc=c_height / 2, scale=c_height / scale_divisor))
        canvas = torch.zeros((4, tile_size[0], tile_size[1]), dtype=torch.float)

        return DatasetGenerator.image_overlay(canvas, overlay, c_x, c_y, o_x, o_y)

    @staticmethod
    def create_tile(
        rng, bg: Tensor, fg: Tensor, alpha_factor: float = 1.0, tile_size=256
    ):
        # Next, sample a tile from this.
        bg_tile = DatasetGenerator.sample_tile(bg, tile_size=tile_size, rng=rng).clone()
        # fg_tile = DatasetGenerator.sample_tile(fg, tile_size=tile_size, rng=rng)
        fg_tile = DatasetGenerator.stamp_tile(rng=rng, tile_size=tile_size, overlay=fg)

        fg_rgb = fg_tile[:3]
        fg_alpha = fg_tile[3:]  # (1, H, W)

        # Now, we perform the blit to create the combined texture....
        combined = alpha_blend(fg_rgb, bg_tile, alpha=fg_alpha * alpha_factor)
        mask = fg_alpha >= 0.5
        # combined = torch.from_numpy(combined)
        mask = mask.to(torch.int64).squeeze()
        combined = augment_jpg_roundtrip(combined, quality=10)
        return (combined, mask)

    def generate(self, count=1):
        results = []

        def create_tile(rng):
            (bg, fg_options) = rng_choice(rng, self._sample_entries)
            fg = rng_choice(rng, fg_options)

            (img, mask) = DatasetGenerator.create_tile(
                rng=self._rng,
                bg=bg,
                fg=fg,
                alpha_factor=self._alpha_factor,
                tile_size=self._tile_size,
            )
            # img = augment_jpg_roundtrip(img)
            return (img, mask)

        # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        #     rngs = [np.random.default_rng(seed=seed + t) for t in range(count)]
        #     results = list(executor.map(create_tile, rngs))
        results = [create_tile(self._rng) for t in range(count)]

        return results

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def set_batch_count(self, batch_count):
        self._batch_count = batch_count


class DynamicGenerator:
    def __init__(self, generator, batch_count=20, batch_size=4, device="cpu"):
        self._generator = generator
        self._batch_count = batch_count
        self._batch_size = batch_size
        self._device = device

    def __iter__(self):
        def gen():
            for i in range(self._batch_count):
                g = self._generator.generate(self._batch_size)
                d = torch.cat([z[0].unsqueeze(0) for z in g], dim=0)
                m = torch.cat([z[1].unsqueeze(0) for z in g], dim=0)
                d = d.to(self._device)
                m = m.to(self._device)
                yield (d, m)

        return gen()


def test_image_overlay():
    import sys

    def d(b_x, b_y, f_x, f_y):
        background = torch.ones((3, 128 * 1, 128 * 1), dtype=torch.float) * 0.2
        foreground = torch.ones((3, 32, 32), dtype=torch.float)

        r = DatasetGenerator.image_overlay(
            background, foreground, b_x=b_x, b_y=b_y, f_x=f_x, f_y=f_y
        )
        torchvision.utils.save_image(
            r, f"/tmp/canvas_overlay_{b_x}_{b_y}__{f_x}_{f_y}.png"
        )

    d(b_x=64, b_y=64, f_x=32, f_y=32)
    d(b_x=64, b_y=64, f_x=16, f_y=32)
    d(b_x=64, b_y=64, f_x=32, f_y=16)
    d(b_x=90, b_y=64, f_x=32, f_y=32)
    # d(b_x=0, b_y=0, f_x=32, f_y=64)
    # d(b_x=64, b_y=64, f_x=96, f_y=96)
    # d(b_x=0, b_y=0, f_x=0, f_y=0)

    sys.exit(1)


# Newfangled data pipeline
class DistributionUniformInt(BaseModel):
    min: int = 1
    max: int = 1


class DistributionNormalInt(BaseModel):
    # Mean of the distribution, 0 is center.
    mean: float = 0.0
    # Sigma of the distribution.
    sigma: float = 4.0
    # Whether to use the canvas dimensions
    by_canvas: bool = True
    # Whether to use our own dimensions.
    by_self: bool = False


# A named group of data input.
class DataInput(BaseModel):
    base_dir: Path | None = None
    dirs: list[Path]
    augmentations: list[str] = []
    remove_alpha: bool = False
    pattern: str = "*.png"
    top_left: tuple[int, int] | None = None
    size: tuple[int, int] | None = None
    device: str = "auto"


class DataApplicator(BaseModel):
    # Ratio of data samples to apply this to.
    ratio: float = 1.0
    # Count to apply when this is applied.
    count: Union[int, DistributionUniformInt] = 1
    # Whether applications can overlap.
    overlap: bool = False
    # Position to place this.
    position_x: Union[DistributionNormalInt, int] = 0
    position_y: Union[DistributionNormalInt, int] = 0

    # Crop the applied image to this size, if the first image, this determines the canvas size.
    crop: Union[None, tuple[int, int]] = None


class DataPostprocess(BaseModel):
    # Name of the postprocessing function.
    name: str
    # Configuration for the postprocessing function
    config: Any
    # Ratio to which this postprocessing function is applied.
    ratio: float = 1.0


class DataStack(BaseModel):
    # List of inputs, (key of DataApplicator, key of DataInput)
    inputs: list[tuple[str, str]]
    # The layer that will make the mask.
    mask_layer: int = 1
    mask_alpha: float = 0.5
    # List of postprocessing actions, mapping to DataPostprocess
    postprocess: list[str] = []


class DataConfig(BaseModel):
    base_dir: Path = Path()
    applicators: dict[str, DataApplicator]
    inputs: dict[str, DataInput]
    data: list[DataStack]


class DataPipeline:
    def __init__(self, config_file):
        with open(config_file) as f:
            d = yaml.safe_load(f)
        self._data_config = DataConfig.model_validate(d["config"])
        print(self._data_config)
        self.load_inputs()

    def load_inputs(self):
        self._inputs = {}
        for name, input_group in self._data_config.inputs.items():
            print(name, input_group)
            this_set = []

            for subdir in input_group.dirs:
                base_dir = (
                    input_group.base_dir
                    if input_group.base_dir is not None
                    else self._data_config.base_dir
                )
                full_dir = base_dir / subdir
                print(full_dir)
                loader = ImageLoader(
                    full_dir,
                    crop_top_left=input_group.top_left,
                    crop_size=input_group.size,
                    remove_alpha=input_group.remove_alpha,
                    device=lookup_device(input_group.device),
                )
                loader.load_images()
                this_set.extend(loader.images())
            self._inputs[name] = this_set
            print(self._inputs)


def test_new_spec():
    import sys

    config_file = "dataset.priv.yaml"
    z = DataPipeline(config_file)
    print(z)
    sys.exit(0)


if __name__ == "__main__":
    # test_image_overlay()
    test_new_spec()

    l = DataLoader("dataset.priv.yaml")
    print()
    d = DatasetGenerator(
        data_pairs=l.generate_data_pairs(),
    )
    d.debug_dump()
