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
