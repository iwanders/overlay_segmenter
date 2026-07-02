from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import yaml
from torch import Tensor

from util import (
    load_image_file,
    load_image_file_u8,
    lookup_device,
)

from .model import CollectionPair, DataGenerationSpec


class TensorNameTracker:
    def __init__(self):
        self._names = {}

    def set_name(self, t: Tensor, name: str):
        self._names[id(t)] = name

    def get_name(self, t: Tensor):
        return self._names.get(id(t))


tensor_tracker = TensorNameTracker()


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
                    )

            for bg_subdir in data_pairs.background_subdir:
                if bg_subdir not in self._bg_images:
                    self._bg_images[bg_subdir] = ImageLoader.background_loader(
                        bg_dir / bg_subdir
                    )

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


class ImageLoader:
    def __init__(
        self,
        crop_top_left: None | tuple[int, int] = None,
        crop_size: None | tuple[int, int] = None,
        device: torch.device | str = "cpu",
        remove_alpha=False,
        as_u8=False,
    ):
        self._crop_top_left = crop_top_left
        self._crop_size = crop_size
        self._device = device
        self._remove_alpha = remove_alpha
        self._as_u8 = as_u8

    @staticmethod
    def background_loader(image_dir, **kwargs):
        if "crop_top_left" not in kwargs:
            kwargs["crop_top_left"] = (105, 27)
        if "crop_size" not in kwargs:
            kwargs["crop_size"] = (1700, 825)
        if "remove_alpha" not in kwargs:
            kwargs["remove_alpha"] = True
        v = ImageLoader(**kwargs)
        return v.load_images(image_dir)

    @staticmethod
    def foreground_loader(image_dir, **kwargs):
        v = ImageLoader(**kwargs)
        return v.load_images(image_dir)

    def load_image(self, d):
        if self._as_u8:
            image = load_image_file_u8(d, device=self._device)
        else:
            image = load_image_file(d, device=self._device)
        left, top = (0, 0) if self._crop_top_left is None else self._crop_top_left
        width, height = (
            (image.shape[2] - left, image.shape[1] - top)
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
                image = image[0:3, :, :].clone()
        return image

    def load_images(self, image_dir: Path) -> list[tuple[Path, Tensor]]:
        to_load = sorted(list(image_dir.rglob("*.png")))

        def load_img(f):
            img = self.load_image(f)
            filename = f.stem
            tensor_tracker.set_name(img, filename)
            return f, img

        with ThreadPoolExecutor() as executor:
            res = list(executor.map(load_img, to_load))
            return [(path, img) for path, img in sorted(res)]
