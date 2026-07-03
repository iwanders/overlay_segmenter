from pathlib import Path
from typing import Union

from pydantic import BaseModel

from .model import (
    DataInput,
    DataPostprocess,
    DistributionNormalInt,
    DistributionUniformFloat,
    DistributionUniformInt,
)

"""
See the dataset_example.yaml file for full descriptions, it's better documentation than the comments in this file.
"""


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
