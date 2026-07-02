from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision
from torch import Tensor
from torchvision.io import decode_jpeg, encode_jpeg

from .model import DataPostprocess
from .pytorch_contrib import _hsv_to_rgb, _rgb_to_hsv
from .rng_util import rng_choice, rng_shuffle


class PostProcess:
    def __init__(self, ratio):
        self._ratio = ratio

    def should_apply(self, rng: np.random.Generator) -> bool:
        return rng.random() <= self._ratio

    @abstractmethod
    def apply(self, rng: np.random.Generator, tensor: Tensor) -> Tensor:
        pass

    def set_ratio(self, ratio: float):
        self._ratio = ratio

    @staticmethod
    def instantiate(postprocess_config: DataPostprocess) -> "PostProcess":
        match postprocess_config.function:
            case "blur":
                return PostprocessBlur(
                    postprocess_config.config,
                    ratio=postprocess_config.ratio,
                )
            case "jpg":
                return PostprocessJpg(
                    postprocess_config.config,
                    ratio=postprocess_config.ratio,
                )
            case "combined":
                return PostprocessCombined(
                    postprocess_config.config,
                    ratio=postprocess_config.ratio,
                )
            case "flip_horizontal":
                return PostprocessFlipHorizontal(
                    postprocess_config.config,
                    ratio=postprocess_config.ratio,
                )
            case "hsv_transform":
                return PostprocessHsvTransform(
                    postprocess_config.config,
                    ratio=postprocess_config.ratio,
                )
            case "channel_clamp":
                return PostprocessChannelClamp(
                    postprocess_config.config,
                    ratio=postprocess_config.ratio,
                )
            case "resize_roundtrip":
                return PostprocessResizeRoundtrip(
                    postprocess_config.config,
                    ratio=postprocess_config.ratio,
                )
            case _ as missing:
                raise NotImplementedError(f"Not implemented postprocess: {missing}")


class PostprocessBlur(PostProcess):
    def __init__(self, config, ratio):
        super().__init__(ratio)
        self._min = config["min"]
        self._max = config["max"]

    def apply(self, rng: np.random.Generator, tensor: Tensor) -> Tensor:
        if not self.should_apply(rng):
            return tensor

        blur_size = rng.uniform(self._min, self._max)

        kernel_size = int(blur_size * 2)
        if kernel_size % 2 == 0:
            kernel_size += 1

        res = torchvision.transforms.functional.gaussian_blur(
            tensor, kernel_size, sigma=blur_size
        )
        return res.to(tensor.device)


def augment_jpg_roundtrip(img, quality=50):
    desired_device = img.device
    # print(desired_device)
    # Go from floats to u8
    inputtype = img.dtype
    if img.dtype == torch.float:
        img = (img * 255.0).to(dtype=torch.uint8).to("cpu")
    elif img.dtype == torch.uint8:
        img = img.to("cpu")
    else:
        raise NotImplementedError(
            "missing augment_jpg_roundtrip for dtype {}".format(fg.dtype)
        )
    encoded = encode_jpeg(img, quality=quality)
    # print(encoded)
    # print(desired_device)
    if inputtype == torch.float:
        return (decode_jpeg(encoded)).to(
            dtype=torch.float, device=desired_device
        ) / 255.0
    else:
        return (decode_jpeg(encoded)).to(device=desired_device)


class PostprocessJpg(PostProcess):
    def __init__(self, config, ratio):
        super().__init__(ratio)
        self._min = config["min"]
        self._max = config["max"]

    def apply(self, rng: np.random.Generator, tensor: Tensor) -> Tensor:
        if not self.should_apply(rng):
            return tensor
        quality = rng.integers(self._min, self._max)
        res = augment_jpg_roundtrip(tensor, quality)
        return res.to(tensor.device)


class PostprocessFlipHorizontal(PostProcess):
    def __init__(self, config, ratio):
        super().__init__(ratio)

    def apply(self, rng: np.random.Generator, tensor: Tensor) -> Tensor:
        if not self.should_apply(rng):
            return tensor
        return torch.flip(tensor, [2])


class PostprocessCombined(PostProcess):
    def __init__(self, config, ratio):
        super().__init__(ratio)
        self._operators = []
        self._child_ratio = config.get("child_ratio", 1.0)
        for conf in config["functions"]:
            parsed = DataPostprocess.model_validate(conf)
            operator = PostProcess.instantiate(parsed)
            operator.set_ratio(self._child_ratio)
            self._operators.append(operator)

    def apply(self, rng: np.random.Generator, tensor: Tensor) -> Tensor:
        if not self.should_apply(rng):
            return tensor
        res = tensor
        for f in self._operators:
            res = f.apply(rng, res)

        return res.to(tensor.device)


class PostprocessResizeRoundtrip(PostProcess):
    def __init__(self, config, ratio):
        super().__init__(ratio)
        self._factors = config["factors"]

    def apply(self, rng: np.random.Generator, tensor: Tensor) -> Tensor:
        if not self.should_apply(rng):
            return tensor
        factor = rng_choice(rng, self._factors)
        current_resolution = tensor.shape[1:]
        small_resolution = (
            int(tensor.shape[1] / factor),
            int(tensor.shape[2] / factor),
        )
        downscaled = torchvision.transforms.functional.resize(tensor, small_resolution)
        upscaled = torchvision.transforms.functional.resize(
            downscaled, current_resolution
        )
        return upscaled


class PostprocessHsvTransform(PostProcess):
    def __init__(self, config, ratio):
        super().__init__(ratio)

        self._h_min = config.get("hue", {}).get("min", 0.0)
        self._h_max = config.get("hue", {}).get("max", 0.0)
        self._s_min = config.get("saturation", {}).get("min", 0.0)
        self._s_max = config.get("saturation", {}).get("max", 0.0)
        self._v_min = config.get("value", {}).get("min", 0.0)
        self._v_max = config.get("value", {}).get("max", 0.0)

    def apply(self, rng: np.random.Generator, tensor: Tensor) -> Tensor:
        if not self.should_apply(rng):
            return tensor

        h = rng.uniform(self._h_min, self._h_max)
        s = rng.uniform(self._s_min, self._s_max)
        v = rng.uniform(self._v_min, self._v_max)

        tensor = tensor.clone().to(torch.float) / 255.0
        rgb = tensor[0:3, :, :]
        # alpha = tensor[3:, :, :]

        # _hsv_to_rgb, _rgb_to_hsv
        as_hsv = _rgb_to_hsv(rgb)

        a_h: Tensor = as_hsv[0, :, :]
        a_h += h
        as_hsv[0, :, :] = a_h.remainder(1.0)

        a_s: Tensor = as_hsv[1, :, :]
        a_s += s
        as_hsv[1, :, :] = a_s.clamp(0, 1.0)

        a_v: Tensor = as_hsv[2, :, :]
        a_v += v
        as_hsv[2, :, :] = a_v.clamp(0, 1.0)

        as_rgb = _hsv_to_rgb(as_hsv)
        tensor[0:3, :, :] = as_rgb

        return (tensor * 255.0).to(torch.uint8).to(tensor.device)


class PostprocessChannelClamp(PostProcess):
    def __init__(self, config, ratio):
        super().__init__(ratio)

        self._rgb_min_min = config.get("rgb", {}).get("min", {}).get("min", 0.0)
        self._rgb_min_max = config.get("rgb", {}).get("min", {}).get("max", 0.0)
        self._rgb_max_min = config.get("rgb", {}).get("max", {}).get("min", 1.0)
        self._rgb_max_max = config.get("rgb", {}).get("max", {}).get("max", 1.0)

    def apply(self, rng: np.random.Generator, tensor: Tensor) -> Tensor:
        if not self.should_apply(rng):
            return tensor
        rgb_clamp_min = rng.uniform(self._rgb_min_min, self._rgb_min_max)
        rgb_clamp_max = rng.uniform(self._rgb_max_min, self._rgb_max_max)
        tensor = tensor.clone().to(torch.float) / 255.0
        tensor[0:3, :, :] = tensor[0:3, :, :].clamp(rgb_clamp_min, rgb_clamp_max)
        return (tensor * 255.0).to(torch.uint8).to(tensor.device)
