from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from torch import Tensor


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


# Newfangled data pipeline
class DistributionUniformInt(BaseModel):
    min: int = 1
    max: int = 1


# Newfangled data pipeline
class DistributionUniformFloat(BaseModel):
    min: float = 0.0
    max: float = 1.0


class DistributionNormalInt(BaseModel):
    # Mean of the distribution, 0 is center.
    mean: float = 0.0
    # Sigma of the distribution.
    sigma: float = 4.0
    # Whether to use our own dimensions for scaling the distrubition, if false use canvas dimensions.
    by_self: bool = True


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
    validation_split: bool = False
    is_overlay: bool = False
    label_directory_name: str | None = None
    rgba_directory_name: str | None = None
    mask_alpha: float = 0.5
    mask_alpha_label: int = 255


class DataPostprocess(BaseModel):
    # Name of the postprocessing function.
    function: str
    # Configuration for the postprocessing function
    config: dict[str, Any] = {}
    # Ratio to which this postprocessing function is applied.
    ratio: float = 1.0
