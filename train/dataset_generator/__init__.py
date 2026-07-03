#!/usr/bin/env python3
#

import colorsys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import torchvision
import yaml
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from letter_support import Glyphset
from util import (
    load_image_file,
    load_image_file_u8,
    lookup_device,
)

from .loader import DataLoader, ImageLoader
from .model import (
    CollectionPair,
    DataGenerationSpec,
    DataInput,
    DataPair,
    DataPostprocess,
    DistributionNormalInt,
    DistributionUniformFloat,
    DistributionUniformInt,
)
from .post_process import PostProcess
from .rng_util import rng_choice, rng_shuffle


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


def alpha_blend(fg, bg, alpha, blend_alpha=None):
    """
    Blends foreground and background using an alpha mask.
    All tensors should be in [0, 1] range.
    fg: (C, H, W) or (N, C, H, W)
    bg: (C, H, W) or (N, C, H, W)
    alpha: (1, H, W) or (N, 1, H, W)
    """
    # Formula: BG * (1 - alpha) + FG * alpha
    # Or simplified: bg + alpha * (fg - bg)
    if fg.dtype == torch.float:
        blend_alpha = 1.0 if blend_alpha is None else blend_alpha
        return bg + alpha * blend_alpha * (fg - bg)
    if fg.dtype == torch.uint8:
        # Output: (fg * alpha + bg * (255 - alpha)) / 255
        # This must fit in u16 space.

        blend_alpha = 255 if blend_alpha is None else blend_alpha

        alpha_u16 = alpha.to(torch.int32)
        alpha_u16 = (alpha_u16 * blend_alpha) // 255

        fg = fg.to(torch.int32)
        bg = bg.to(torch.int32)
        blended = (fg * alpha_u16 + bg * (255 - alpha_u16)) // 255
        res = blended.to(dtype=torch.uint8)
        return res
    else:
        raise NotImplementedError("missing blend for dtype {}".format(fg.dtype))


def load_paths(path_file):
    with open(path_file) as f:
        return [a.strip() for a in f.readlines()]


@dataclass
class Rect:
    x: tuple[int, int]
    y: tuple[int, int]

    def overlaps(self, other: "Rect") -> bool:
        return (
            self.x[0] < other.x[1]
            and self.x[1] > other.x[0]
            and self.y[0] < other.y[1]
            and self.y[1] > other.y[0]
        )


@dataclass
class OverlayResult:
    composite: Tensor
    overlaid: Tensor | None
    composite_x: tuple[int, int]
    composite_y: tuple[int, int]

    def composite_rect(self) -> Rect:
        return Rect(x=self.composite_x, y=self.composite_y)


def image_overlay(
    background,
    foreground,
    b_x,
    b_y,
    f_x,
    f_y,
    return_overlay=False,
    dtype=torch.float,
) -> OverlayResult:
    # We've selected the position in the canvas, and the position in the overlay.
    # next, we have to determine the rectangle in which the bounds overlap.
    # We will place the overlay coordinate onto the canvas coordinate.

    # Calculate the overlapping region
    bg_h, bg_w = background.shape[-2], background.shape[-1]
    fg_h, fg_w = foreground.shape[-2], foreground.shape[-1]

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

    mask = None
    if return_overlay:
        mask = torch.zeros(
            background.shape,
            dtype=dtype,
        )

    # Handle two situations where the intersection is disjoint; ie; the overlay is outside of the bg.
    if y2 < y1 or x2 < x1:
        return OverlayResult(
            composite=background,
            overlaid=mask,
            composite_x=(0, 0),
            composite_y=(0, 0),
        )

    if fg_y2 < fg_y1 or fg_x2 < fg_x1:
        return OverlayResult(
            composite=background,
            overlaid=mask,
            composite_x=(0, 0),
            composite_y=(0, 0),
        )

    # Apply the overlay
    if foreground.shape[0] == 4:
        background[:, y1:y2, x1:x2] = foreground[:, fg_y1:fg_y2, fg_x1:fg_x2]
    else:
        if dtype == torch.float:
            background[0:3, y1:y2, x1:x2] = foreground[0:3, fg_y1:fg_y2, fg_x1:fg_x2]
            background[3, :, :] = 1.0
        elif dtype == torch.uint8:
            background[0:3, y1:y2, x1:x2] = foreground[0:3, fg_y1:fg_y2, fg_x1:fg_x2]
            background[3, :, :] = 255
        elif dtype == torch.int64:
            single_channel = foreground[fg_y1:fg_y2, fg_x1:fg_x2]
            indices = single_channel != 0
            background[y1:y2, x1:x2][indices] = single_channel[indices]

        else:
            raise NotImplementedError("dtype not supported")

    if return_overlay:
        mask[:, y1:y2, x1:x2] = foreground[:, fg_y1:fg_y2, fg_x1:fg_x2]

    return OverlayResult(
        composite=background,
        overlaid=mask,
        composite_x=(x1, x2),
        composite_y=(y1, y2),
    )


class DynamicGenerator:
    def __init__(self, batch_generator, batch_count=20, batch_size=4, device="cpu"):
        self._batch_generator = batch_generator
        self._batch_count = batch_count
        self._batch_size = batch_size
        self._device = device

    def __iter__(self):
        def gen():
            for i in range(self._batch_count):
                g = self._batch_generator(self._batch_size)
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


class ImageApplicatorConfig(BaseModel):
    # Ratio of data samples to apply this to.
    ratio: float = 1.0
    # Count to apply when this is applied.
    count: Union[int, DistributionUniformInt] = 1
    # Whether applications can overlap.
    overlap: bool = False
    # Position to place this.
    position_x: Union[DistributionNormalInt, int] = 0
    position_y: Union[DistributionNormalInt, int] = 0

    blend_alpha: Union[DistributionUniformFloat, float] = 1.0

    # Crop the applied image to this size, if the first image, this determines the canvas size.
    crop: Union[None, tuple[int, int]] = None

    pre_process_image: list[str] = []
    post_process_image: list[str] = []

    # Whether this applicator's mask is used as a mask.
    use_mask: bool = False


class TextGenerationConfig(BaseModel):
    # Key that maps to the glyphs sets.
    glyph_set: str

    # The height of the text canvas
    canvas_height: int

    # WHere on the canvas to place the baseline
    canvas_baseline: int

    # THe margin added to the determined text width on the left
    margin_left: int = 0
    # THe margin added to the determined text width on the right
    margin_right: int = 0
    # Text file with each line a possible string option to pick from.
    text_lines: Path | list[str]
    # Background color to use before drawing the text.
    background_color_rgba_u8: tuple[int, int, int, int] | None = None

    skip_missing_characters: bool = True


class OverlaySource(ABC):
    @abstractmethod
    def get_count(self) -> int:
        pass

    @abstractmethod
    def create(self, rng: np.random.Generator) -> Tensor:
        pass


class ImagePicker(OverlaySource):
    def __init__(self, collection: list[Tensor]):
        self._collection = collection

    def get_count(self) -> int:
        return len(self._collection)

    def create(self, rng: np.random.Generator) -> Tensor:
        return rng_choice(rng, self._collection)


class TextSource(OverlaySource):
    def __init__(
        self, config: TextGenerationConfig, config_file: Path, glyphset: Glyphset
    ):
        self._config = config
        if isinstance(self._config.text_lines, Path):
            text_line_path = Path(config_file).parent / self._config.text_lines
            if not text_line_path.is_file():
                raise FileNotFoundError(f"Failed to open {text_line_path}")
            with open(text_line_path, "r") as f:
                self._config.text_lines = f.read().splitlines()
        self._glyphset = glyphset

        necessary_characters = set("".join(self._config.text_lines))
        have_characters = set([a.tokens() for a in self._glyphset.glyphs()])
        missing = necessary_characters - have_characters
        if config.skip_missing_characters:

            def remover(a: str) -> str:
                for m in missing:
                    a = a.replace(m, "")
                return a

            for i in range(len(self._config.text_lines)):
                self._config.text_lines[i] = remover(self._config.text_lines[i])

        else:
            if missing:
                raise ValueError(
                    f"Missing characters, can't assemble all strings, necessary; {necessary_characters}, got {have_characters}, missing {missing}"
                )

    def get_count(self) -> int:
        return len(self._config.text_lines)

    def create(self, rng: np.random.Generator) -> Tensor:
        chosen_text = rng_choice(rng, self._config.text_lines)
        tokens = list(chosen_text[:])
        width = self._glyphset.typeset_width(tokens)
        canvas_width = width + self._config.margin_left + self._config.margin_right
        canvas_height = self._config.canvas_height

        background_color_rgba_u8 = self._config.background_color_rgba_u8
        if background_color_rgba_u8 is None:
            background_color_rgba_u8 = [0, 0, 0, 0]
        # canvas = torch.zeros((4, canvas_height, canvas_width), dtype=torch.uint8)
        color = torch.tensor(
            background_color_rgba_u8,
            dtype=torch.uint8,
        ).view(4, 1, 1)
        canvas = color.expand(4, canvas_height, canvas_width).clone()

        self._glyphset.typeset(
            canvas, tokens, x=self._config.margin_left, y=self._config.canvas_baseline
        )
        return canvas  # (canvas * 255.0).to(torch.uint8)


class DataStack(BaseModel):
    # List of inputs, (key of DataApplicator, key of DataInput)
    inputs: list[tuple[str, str]]
    for_input_keys: dict[str, list[str]] = {"__dumy__": ["__DUMMY__"]}

    # List of postprocessing actions, mapping to DataPostprocess
    post_process: list[str] = []


class GlyphsetConfig(BaseModel):
    config: Path


class DataConfig(BaseModel):
    base_dir: Path = Path()
    process_device: str = "auto"
    image_applicators: dict[str, ImageApplicatorConfig]
    text_groups: dict[str, TextGenerationConfig] = {}
    image_groups: dict[str, DataInput]
    generator: list[DataStack]
    post_process: dict[str, DataPostprocess] = {}
    glyphsets: dict[str, GlyphsetConfig] = {}


class LabelledOverlay(BaseModel):
    label: Tensor
    overlay: Tensor
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ImageApplicator:
    def __init__(
        self,
        config: ImageApplicatorConfig,
        device: torch.device,
        post_processors: dict[str, PostProcess],
    ):
        self._config = config
        self._device = device
        self._pre_process_image = [post_processors[k] for k in config.pre_process_image]
        self._post_process_image = [
            post_processors[k] for k in config.post_process_image
        ]

    def __str__(self):
        return f"<Applicator {self._config} at 0x{id(self):x}>"

    def crop(self) -> tuple[int, int] | None:
        return self._config.crop

    def ratio(self) -> float:
        return self._config.ratio

    def use_mask(self) -> bool:
        return self._config.use_mask

    def _get_count(self, rng: np.random.Generator) -> int:
        if type(self._config.count) is int:
            return self._config.count
        else:
            return int(rng.uniform(self._config.count.min, self._config.count.max))

    @staticmethod
    def _determine_pos(
        rng: np.random.Generator,
        config: Union[DistributionNormalInt, int],
        value_self: int,
        value_canvas: int,
    ) -> int:
        if type(config) is int:
            return config

        normal_config: DistributionNormalInt = config
        # It must be a normal sampling.
        offset = 0
        scale = value_canvas
        if normal_config.by_self:
            scale = value_self
        else:
            # This centers it >_<
            offset = -value_canvas / 2 + value_self / 2

        return int(
            rng.normal(loc=normal_config.mean, scale=normal_config.sigma) * scale
            + offset
        )

    @staticmethod
    def _determine_blend_alpha(
        rng: np.random.Generator, config: Union[DistributionUniformFloat, float]
    ) -> float:
        if type(config) is float:
            return config
        return float(rng.uniform(config.min, config.max))

    def apply(
        self,
        rng: np.random.Generator,
        canvas: Tensor,
        overlay_source: OverlaySource,
        return_mask: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        canvas = canvas
        mask = None
        placed = []
        for i in range(self._get_count(rng)):
            raw_overlay = overlay_source.create(rng)
            raw_mask = None
            if return_mask:
                overlay = raw_overlay.overlay
                raw_mask = raw_overlay.label
            else:
                overlay = raw_overlay

            for preprocessor in self._pre_process_image:
                overlay = preprocessor.apply(rng, overlay)

            o_height, o_width = overlay.shape[1:]
            c_height, c_width = canvas.shape[1:]

            sub_canvas = torch.zeros(
                (4, canvas.shape[1], canvas.shape[2]),
                dtype=torch.uint8,
                device=self._device,
            )

            o_x = self._determine_pos(rng, self._config.position_x, o_width, c_width)
            o_y = self._determine_pos(rng, self._config.position_y, o_height, c_height)

            c_x = int(c_width / 2)
            c_y = int(c_height / 2)

            if not self._config.overlap:
                # Super ugly duplicated from image_overlay
                def pos_overlapping(x, y):
                    # Calculate the overlapping region
                    bg_h, bg_w = c_height, c_width
                    fg_h, fg_w = o_height, o_width

                    b_x = int(c_x - bg_w / 2)
                    b_y = int(c_y - bg_h / 2)
                    f_x = int(x - bg_w / 2)
                    f_y = int(y - bg_h / 2)

                    # x_offset and y_offset is the top left corner of the overlay in bg coordinates.
                    x_offset = int(b_x - f_x)
                    y_offset = int(b_y - f_y)
                    # Determine intersection coordinates (handles boundary crossing)
                    y1 = max(0, y_offset)
                    y2 = min(bg_h, y_offset + fg_h)
                    x1 = max(0, x_offset)
                    x2 = min(bg_w, x_offset + fg_w)
                    new_rect = Rect(x=(x1, x2), y=(y1, y2))
                    for o in placed:
                        if o.overlaps(new_rect):
                            return True

                    return False

                positioned = False
                for _attempt in range(100):
                    if pos_overlapping(o_x, o_y):
                        o_x = self._determine_pos(
                            rng, self._config.position_x, o_width, c_width
                        )
                        o_y = self._determine_pos(
                            rng, self._config.position_y, o_height, c_height
                        )
                    else:
                        positioned = True
                        break
                if not positioned:
                    print("Failed to position because of overlap")

            overlay_result = image_overlay(
                sub_canvas,
                overlay,
                c_x,
                c_y,
                o_x,
                o_y,
                return_overlay=return_mask,
                dtype=torch.uint8,
            )

            sub_canvas = overlay_result.composite
            if return_mask:
                if raw_mask is None:
                    raise ValueError(
                        "raw mask is None while return mask, something is wrong :/"
                    )
                else:
                    mask_canvas = torch.zeros(
                        (canvas.shape[1], canvas.shape[2]),
                        dtype=torch.int64,
                        device=self._device,
                    )
                    mask_result = image_overlay(
                        mask_canvas,
                        raw_mask,
                        c_x,
                        c_y,
                        o_x,
                        o_y,
                        return_overlay=False,
                        dtype=torch.int64,
                    )
                    mask = mask_result.composite

            blend_alpha = self._determine_blend_alpha(rng, self._config.blend_alpha)
            blend_alpha = int(blend_alpha * 255)

            fg_rgb = sub_canvas[:3]
            fg_alpha = sub_canvas[3:]  # (1, H, W)
            canvas = alpha_blend(
                fg_rgb, canvas[:3, :, :], fg_alpha, blend_alpha=blend_alpha
            )
            placed.append(overlay_result.composite_rect())

        for processor in self._post_process_image:
            canvas = processor.apply(rng, canvas)

        return canvas, mask


class DataGenerator:
    def __init__(
        self,
        stack: list[tuple[ImageApplicator, list[Tensor]]],
        config: DataStack,
        device: torch.device,
        post_processors: list[PostProcess],
    ):
        self._stack = stack
        self._config = config
        self._device = device
        self._post_processors = post_processors

    def generate(self, rng: np.random.Generator) -> (Tensor, Tensor):
        canvas = None
        mask = None
        for layer, (applicator, images) in enumerate(self._stack):
            if canvas is None and applicator.crop() is None:
                raise ValueError("Dont have a cropping first applicator")

            if canvas is None:
                canvas = torch.zeros(
                    (4, applicator.crop()[0], applicator.crop()[1]),
                    dtype=torch.uint8,
                    device=self._device,
                )
                canvas[3, :, :] = 255

            # Check if this layer is supposed to apply.
            if rng.random() >= applicator.ratio():
                continue

            if applicator.use_mask():
                canvas, new_mask = applicator.apply(
                    rng, canvas, images, return_mask=True
                )

                if mask is None:
                    mask = new_mask
                elif new_mask is not None:
                    # Actually apply the mask, by copying over values that are non zero.

                    non_zero = new_mask != 0
                    mask[non_zero] = new_mask[non_zero]

            else:
                canvas, _ = applicator.apply(rng, canvas, images, return_mask=False)

        # Drop the alpha from the canvas.
        canvas = canvas[0:3, :, :].clone()

        # Perform postprocessing.
        for post_processor in self._post_processors:
            canvas = post_processor.apply(rng, canvas)

        return canvas, mask

    def first_input_count(self) -> int:
        return self._stack[0][1].get_count()


class DataPipeline:
    def __init__(self, config_file: Path | None = None, full_init=True):
        self._config_file = config_file
        if config_file is not None:
            self._config_file = Path(config_file)
            with open(config_file) as f:
                d = yaml.safe_load(f)
            self._data_config = DataConfig.model_validate(d["data_config"])
            # print(self._data_config)
            self._device = lookup_device(self._data_config.process_device)
            self.load_input_groups()

            data_generator_intersection = set(
                self._data_config.image_groups.keys()
            ).intersection(self._data_config.text_groups.keys())
            if data_generator_intersection:
                raise KeyError(
                    f"The key {data_generator_intersection} exists in both image_groups and text_groups"
                )

        if full_init:
            self.post_image_init()

    def post_image_init(self):
        self.input_augment()
        self.load_postprocess()
        self.load_applicators()
        self.load_glyphsets()
        self.load_text_groups()
        self.create_generators()
        self.calculate_generator_weights()

    def split_validation(self, rng: np.random.Generator, ratio=0.1) -> "DataPipeline":
        # Split the images for all images in validation_split
        validation_pipeline = DataPipeline(full_init=False)
        validation_pipeline._data_config = self._data_config
        validation_pipeline._device = self._device
        validation_pipeline._input_groups = {}
        validation_pipeline._config_file = self._config_file
        for name, input_group in self._data_config.image_groups.items():
            if input_group.validation_split:
                images = self._input_groups[name][:]
                images = rng_shuffle(rng, images)

                total_bg = len(images)
                validation_bg_split = int(total_bg * ratio)
                validation_entries = images[0:validation_bg_split]
                self._input_groups[name] = images[validation_bg_split:]
                validation_pipeline._input_groups[name] = validation_entries
            else:
                validation_pipeline._input_groups[name] = self._input_groups[name]

        validation_pipeline.post_image_init()
        return validation_pipeline

    def _substitute_path(self, path: Path, extra=None) -> Path:
        if extra is None:
            extra = {}
        path_as_str = str(path)
        return Path(path_as_str.format(base_dir=self._data_config.base_dir, **extra))

    def print_inputs(self):
        for name, images in self._input_groups.items():
            if isinstance(images[0], LabelledOverlay):
                print(
                    f"Inputs: {name: >20} has {len(images): >4} labelled images with {images[0].overlay.shape} in {images[0].overlay.dtype} size on {images[0].overlay.device}"
                )
            else:
                print(
                    f"Inputs: {name: >20} has {len(images): >4}          images with {images[0].shape} in {images[0].dtype} size on {images[0].device}"
                )

    def load_input_groups(self):
        self._input_groups = {}
        for name, input_group in self._data_config.image_groups.items():
            loader = ImageLoader(
                crop_top_left=input_group.top_left,
                crop_size=input_group.size,
                remove_alpha=input_group.remove_alpha,
                device=lookup_device(input_group.device),
                as_u8=True,
            )
            if input_group.is_overlay:
                if (
                    input_group.label_directory_name is None
                    and input_group.rgba_directory_name is None
                ):
                    # traditional handling, we alpha mask here.
                    this_set = []
                    for subdir in input_group.dirs:
                        base_dir = (
                            self._substitute_path(input_group.base_dir)
                            if input_group.base_dir is not None
                            else self._data_config.base_dir
                        )
                        full_dir = base_dir / subdir

                        this_set.extend(
                            loader.load_images(
                                full_dir,
                            )
                        )
                    # Drop the filenames.
                    this_set = [img for _, img in this_set]
                    masks = []
                    for t in this_set:
                        labels = t[3, :, :] >= (input_group.mask_alpha * 255)
                        labels = labels.to(torch.int64) * input_group.mask_alpha_label
                        masks.append(labels)
                    this_set = [
                        LabelledOverlay(label=label, overlay=overlay)
                        for label, overlay in zip(masks, this_set)
                    ]
                    self._input_groups[name] = this_set
                else:
                    # If it is an overlay, we have two images for each.
                    # LabelledOverlay
                    if (
                        input_group.label_directory_name is None
                        or input_group.rgba_directory_name is None
                    ):
                        print("missing label_directory_name or rgba_directory_name")

                    all_images = [[], []]
                    for subdir in input_group.dirs:
                        for category_i, subdir_name in enumerate(
                            [
                                input_group.label_directory_name,
                                input_group.rgba_directory_name,
                            ]
                        ):
                            base_dir = (
                                self._substitute_path(
                                    input_group.base_dir,
                                    {"rgba_or_label_directory": subdir_name},
                                )
                                if input_group.base_dir is not None
                                else self._data_config.base_dir
                            )
                            full_dir = base_dir / subdir
                            images_with_name = loader.load_images(
                                full_dir,
                            )
                            # strip the path up to the category.
                            images_with_name = [
                                (str(name.relative_to(base_dir)), img)
                                for name, img in images_with_name
                            ]
                            all_images[category_i].extend(images_with_name)

                    # next, zip up the images.
                    labels, rgbas = all_images
                    labels_dict = dict(labels)
                    rgbas_dict = dict(rgbas)
                    notpaired = labels_dict.keys() ^ rgbas_dict.keys()
                    if notpaired:
                        print(f"Missing counterparts for: {str(notpaired)}")
                        print("continuing but skipping those")

                    # Find exact counterparts and convert labels to integers.
                    paired_label_image = []
                    for k, label_3channel in labels_dict.items():
                        rgb = rgbas_dict.get(k)
                        if rgb is None:
                            print(f"Missing {k}, skipping this tile")
                            continue
                        labelled = label_3channel[1, :, :]
                        labelled = labelled.to(torch.int64)
                        paired = LabelledOverlay(label=labelled, overlay=rgb)
                        paired_label_image.append(paired)

                    self._input_groups[name] = paired_label_image

            else:
                this_set = []
                for subdir in input_group.dirs:
                    base_dir = (
                        self._substitute_path(input_group.base_dir)
                        if input_group.base_dir is not None
                        else self._data_config.base_dir
                    )
                    full_dir = base_dir / subdir
                    these_images = loader.load_images(
                        full_dir,
                    )
                    these_images = [img for _p, img in these_images]
                    this_set.extend(these_images)
                self._input_groups[name] = this_set

    def load_glyphsets(self):
        self._glyphsets = {}
        for name, input_group in self._data_config.glyphsets.items():
            self._glyphsets[name] = Glyphset(self._substitute_path(input_group.config))

    def load_text_groups(self):
        self._text_groups = {}
        for name, input_group in self._data_config.text_groups.items():
            if not isinstance(input_group.text_lines, list):
                input_group.text_lines = self._substitute_path(input_group.text_lines)
            glyphset = self._glyphsets.get(input_group.glyph_set)
            if glyphset is None:
                raise ValueError(f"Glyphset {input_group.glyph_set} not found")

            self._text_groups[name] = TextSource(
                input_group, config_file=self._config_file, glyphset=glyphset
            )

    def input_augment(self):
        for name, input_group in self._data_config.image_groups.items():
            these_images = self._input_groups[name]
            new_images = []
            for augmentation in input_group.augmentations:
                if augmentation == "flip_horizontal":
                    for img in these_images:
                        new_images.append(torch.flip(img, [2]))
            self._input_groups[name].extend(new_images)

    def load_applicators(self):
        self._image_applicators = {}
        for name, applicator_config in self._data_config.image_applicators.items():
            self._image_applicators[name] = ImageApplicator(
                applicator_config,
                device=self._device,
                post_processors=self._postprocess,
            )

    def load_postprocess(self):
        self._postprocess = {}
        for name, postprocess_config in self._data_config.post_process.items():
            self._postprocess[name] = PostProcess.instantiate(postprocess_config)

    def create_generators(self):
        self._generators: list[DataGenerator] = []
        for config in self._data_config.generator:
            # Verify that all for_input_keys entries are the same length.
            entry_length = len(list(config.for_input_keys.values())[0])
            for label, values in config.for_input_keys.items():
                if len(values) != entry_length:
                    raise ValueError(
                        f"for_input_keys lengths not equal to each other in {label}"
                    )

            for index in range(entry_length):
                substitutions = {k: v[index] for k, v in config.for_input_keys.items()}
                typed_stack = []
                for applicator_name, collection_name in config.inputs:
                    applicator_name = applicator_name.format(**substitutions)
                    collection_name = collection_name.format(**substitutions)
                    applicator = self._image_applicators[applicator_name]
                    if collection_name in self._input_groups:
                        overlay_source = ImagePicker(
                            self._input_groups[collection_name]
                        )
                    elif collection_name in self._text_groups:
                        overlay_source = self._text_groups[collection_name]
                    else:
                        raise KeyError(
                            f"Could not find collection name anywhere: {collection_name}"
                        )

                    typed_stack.append((applicator, overlay_source))
                # Collect postprocessors
                post_processors = []
                for post_process_name in config.post_process:
                    post_processors.append(self._postprocess[post_process_name])
                generator = DataGenerator(
                    typed_stack,
                    config=config,
                    device=self._device,
                    post_processors=post_processors,
                )
                self._generators.append(generator)

    def calculate_generator_weights(self):
        # Calculate this based on the first layer input count.
        w = []
        for generator in self._generators:
            count = generator.first_input_count()
            w.append(count)
        w = np.array(w)
        self._generator_weights = w / np.sum(w, dtype=np.float64)

    def generate_with_generator(
        self, index: int, rng: np.random.Generator
    ) -> tuple[Tensor, Tensor]:
        img, mask = self._generators[index].generate(rng)
        img = img.to(torch.float) / 255.0
        return img, mask

    def generate(self, rng: np.random.Generator) -> tuple[Tensor, Tensor]:
        choice = rng.choice(range(len(self._generators)), p=self._generator_weights)
        return self.generate_with_generator(choice, rng)

    def batch_generator_fun(self, rng: np.random.Generator) -> Any:
        def batch_generator(batch_size):
            return [self.generate(rng) for _ in range(batch_size)]

        return batch_generator


def mask_label_map(m: Tensor, label_map: dict[int, int]) -> Tensor:
    if m.dim() == 2:
        # single image.
        new_mask = torch.zeros(m.shape, dtype=torch.int64, device=m.device)
        for k, v in label_map.items():
            where = m == k
            ones = torch.ones(m.shape, dtype=torch.int64, device=m.device)
            new_mask[where] = ones[where] * v
        return new_mask
    elif m.dim() == 3:
        # with batch
        new_mask = torch.zeros(m.shape, dtype=torch.int64, device=m.device)
        for i in range(m.shape[0]):
            for k, v in label_map.items():
                where = m[i, :, :] == k
                ones = torch.ones(m.shape[1:], dtype=torch.int64, device=m.device)
                new_mask[i, :, :][where] = ones[where] * v
        return new_mask
    else:
        raise NotImplementedError(f"missing mask_label_map handling for {m.shape}")


def label_map_to_rgbmask(m: Tensor, label_to_rgb) -> Tensor:
    if m.dim() == 2:
        return label_to_rgb.to(device=m.device)[m].permute([2, 0, 1])

    elif m.dim() == 3:
        B = m.shape[0]
        h = m.shape[1]
        w = m.shape[2]

        rgbmask = torch.zeros((B, 3, h, w), dtype=torch.float, device=m.device)
        for i in range(B):
            rgbmask[i, :, :, :] = label_map_to_rgbmask(m[i, :, :], label_to_rgb)
        return rgbmask


def generate_color_palette(classess: int, force_black=True) -> Tensor:
    colors = []
    if force_black:
        colors.append((0, 0, 0))
        classess -= 1
    for i in range(classess):
        # 1. Divide hue space evenly between 0.0 and 1.0
        hue = i / classess

        # 2. Keep saturation and value high for vibrant, distinct colors
        saturation = 1.0
        value = 1.0

        # 3. Convert HSV to an RGB tuple (returns floats from 0.0 to 1.0)
        rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)

        # 4. Scale to standard 0-255 integer format
        # rgb_int = tuple(int(channel * 255) for channel in rgb_float)

        colors.append(rgb_float)
    return torch.tensor(colors)


def logits_to_rgb_values(logits, label_to_rgb) -> list[Tensor]:
    output_channels = logits.shape[0]
    height, width = logits.shape[1:]
    if output_channels != len(label_to_rgb):
        raise ValueError("Output channels doesn't match label to rgb length")

    per_channel = []

    for c in range(output_channels):
        t = logits[c, :, :]
        span = t.max() - t.min()
        t = (t - t.min()) / span
        color_tensor = label_to_rgb[c].to(device=logits.device)
        solid_color_image = color_tensor.view(3, 1, 1).expand(3, height, width)
        scaled_color = t * solid_color_image
        per_channel.append(scaled_color)

    return per_channel


def test_new_spec():
    import sys

    config_file = "dataset_example.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    z = DataPipeline(Path(config_file))
    print(z)
    rng = np.random.default_rng(3)
    generated = []

    label_map = {
        # Background
        0: 0,
        # Foreground
        255: 1,
        # Foreground special1
        128: 2,
        # Foreground special2
        64: 2,
    }
    label_to_rgb = generate_color_palette(3)

    z.print_inputs()
    for i in range(10):
        a = z.generate(rng)
        generated.append(a)

    batch_generator = DynamicGenerator(
        batch_generator=z.batch_generator_fun(rng), batch_size=4
    )
    rollout_gen = iter(batch_generator)
    generated = [(img, mask) for img, mask in rollout_gen]

    output = Path("/tmp/")
    for i, (sample_img, sample_mask) in enumerate(generated):
        torchvision.utils.save_image(sample_img, output / f"sample_{i}_img.png")
        print("sample_mask min max", sample_mask.min(), sample_mask.max())
        mask_labels = mask_label_map(sample_mask, label_map)
        rgbmask = label_map_to_rgbmask(mask_labels, label_to_rgb)
        # torchvision.utils.save_image([img, rgbmask], out_path, normalize=False)
        torchvision.utils.save_image(
            rgbmask.to(torch.float),
            output / f"sample_{i}_mask.png",
            normalize=False,
        )
    sys.exit(0)
