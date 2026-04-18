#!/usr/bin/env python3


"""
This is a helper file to go from a png file to glyphs, then we can concatenate these glyphs like a letterpress to make
text in the original font without relying on having the font, or having text rendering.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import yaml
from pydantic import BaseModel
from torch import Tensor

from dataset_generator import load_image_file


class GlyphSpec(BaseModel):
    start: int | None = None
    # End, inclusive!
    end: int | None = None
    tokens: str


class FontSpec(BaseModel):
    image_path: Path
    baseline: int
    ascender: int
    descender: int
    # Skip this many pixels from the left (such that we can put some annotation pixels)
    skip_left: int = 1
    inter_character_minimum: int = 1
    glyphs: list[GlyphSpec]


# This is this:
# https://en.wikipedia.org/wiki/Sort_(typesetting)
#
# But then... with Glyph prefixed to it because 'Sort' is too vague.
class GlyphSort:
    def __init__(self, tokens: str, img: Tensor, baseline: int):
        self._tokens = tokens
        self._img = img
        self._baseline = baseline

    def image(self) -> Tensor:
        return self._img

    def tokens(self) -> str:
        return self._tokens

    def filename_tokens(self) -> str:
        return self._tokens.replace("/", "_slash_").replace('"', "_quote_")


class Glyphset:
    def __init__(self, glyphset_yaml: Path):
        with open(args.input) as f:
            d = yaml.safe_load(f)
        spec = FontSpec.model_validate(d)
        self._spec = spec
        self.create_glyphs()

    @staticmethod
    def find_first_nonzero(v: Tensor, start_index: int | None) -> int | None:
        nonzero_indices = torch.nonzero(v[start_index:])
        if nonzero_indices.numel() > 0:
            return nonzero_indices[0] + start_index
        else:
            return None

    @staticmethod
    def find_start_consecutive_zero(
        v: Tensor, start_index: int | None, min_consecutive: int
    ) -> int | None:
        target_value = 0
        N = min_consecutive
        v = v[start_index:]
        # Create a boolean mask where the condition is met
        mask = (v == target_value).float()

        # Define a kernel of ones of length N
        kernel = torch.ones(N).view(1, 1, -1)

        # Apply 1D convolution
        # Padding is 0, stride is 1
        res = F.conv1d(mask.view(1, 1, -1), kernel)

        # Find where the sum equals N
        indices = (res == N).nonzero()

        if indices.numel() > 0:
            return indices[0, 2].item() + start_index
        return None

    def create_glyphs(self):
        d = load_image_file(args.input.parent / self._spec.image_path, device="cpu")
        # First, get the region between the ascender and the descender.
        self._line = d[:, self._spec.ascender : self._spec.descender + 1, :]
        # Now that we have the line, next up is just striding through the line in horizontal direction and segmenting
        # the letters.
        # Make an horizontally flattened version of that.
        flat_line_no_channel = self._line.sum(dim=0)
        flat_line = flat_line_no_channel.sum(dim=0)
        self._glyphs = []
        line_pos = self._spec.skip_left
        for glyph_spec in self._spec.glyphs:
            glyph_start = (
                glyph_spec.start
                if glyph_spec.start is not None
                else Glyphset.find_first_nonzero(flat_line, line_pos)
            )
            glyph_end = (
                glyph_spec.end + 1
                if glyph_spec.end is not None
                else Glyphset.find_start_consecutive_zero(
                    flat_line,
                    glyph_start,
                    min_consecutive=self._spec.inter_character_minimum,
                )
            )
            if glyph_start is None:
                raise ValueError("glyph start could not be determined")
            if glyph_end is None:
                raise ValueError("glyph glyph_end could not be determined")
            glyph_image = self._line[:, :, glyph_start:glyph_end]
            self._glyphs.append(
                GlyphSort(glyph_spec.tokens, glyph_image, baseline=self._spec.baseline)
            )
            line_pos = glyph_end

    def glyphs(self) -> list[GlyphSort]:
        return self._glyphs


def run_glyphset(args):
    glyphset = Glyphset(args.input)

    torchvision.utils.save_image(glyphset._line, "/tmp/line.png")
    for i, glyph in enumerate(glyphset.glyphs()):
        torchvision.utils.save_image(
            glyph.image(), f"/tmp/glyph_{i:0>3}_{glyph.filename_tokens()}.png"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="letter_support")
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    parser_glyphset = subparsers.add_parser(
        "glyphset",
        help="Obtain a glyph set, coordinates are pixel position when the pixel is selected in GIMP",
    )
    parser_glyphset.add_argument("input", type=Path)
    parser_glyphset.set_defaults(func=run_glyphset)

    # parser_test = subparsers.add_parser("test", help="Run inference")
    # parser_test.set_defaults(func=run_test)

    args = parser.parse_args()

    # Execute the selected command's function
    if args.command:
        args.func(args)
    else:
        parser.print_help()
