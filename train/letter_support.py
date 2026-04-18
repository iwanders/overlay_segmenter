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

from util import load_image_file, load_image_file_u8


class GlyphSpec(BaseModel):
    # Override the master baseline for this glyph (are you really sure!?)
    # This is relative to the image obtained from cliping the font between ascender and descender!
    # IE; it is the baseline value used for positiong the glyph
    baseline: int | None = None
    # Start, inclusive, value in gimp.
    start: int | None = None
    # End, inclusive, value in gimp.
    end: int | None = None
    # String representation of this section.
    tokens: str

    # Extra spacing on the left of the character, may be negative.
    left_spacing: int = 0
    # Extra spacing on the right of the character, may be negative.
    right_spacing: int = 0

    # Marks the character as a space (tab, " ", etc...)
    is_space: bool = False

    # THis is not really a character it merely denotes a skipped character during the glyphset creation.
    is_skipped: bool = False


class FontSpec(BaseModel):
    # Image that holds the individual glyphs.
    image_path: Path
    # Baseline behaves as the anchor of a line, it is the 'base' of all letters. Selection of baseline in gimp.
    # https://en.wikipedia.org/wiki/Baseline_(typography)
    baseline: int
    # The ascender is the highest pixel value of any character. Value is inclusive, as selected in gimp.
    ascender: int
    # Descender is the lowest pixel value of any character. Value is inclusive, as selected in gimp.
    descender: int
    # Skip this many pixels from the left (such that we can put some annotation pixels)
    skip_left: int = 1
    # This many transparent pixels between characters, to ensure we merge " and characters like that.
    inter_character_minimum: int = 1
    # List of all individual glyphs, where individual values can be specified.
    glyphs: list[GlyphSpec]

    # Width of a space character that delimites two words.
    space_width: int

    # Distance of whitespace between two letters
    letter_spacing: int


# This is this:
# https://en.wikipedia.org/wiki/Sort_(typesetting)
#
# But then... with Glyph prefixed to it because 'Sort' is too vague.
class GlyphSort:
    def __init__(self, spec: GlyphSpec, img: Tensor):
        self._spec = spec
        if self._spec.baseline is None:
            raise ValueError("At this point the baseline must be set!")
        self._img = img

    def image(self) -> Tensor:
        return self._img

    def tokens(self) -> str:
        return self._spec.tokens

    def filename_tokens(self) -> str:
        return self._spec.tokens.replace("/", "_slash_").replace('"', "_quote_")

    def typeset(self, canvas: Tensor, x: int, y: int):
        # print(f"Typesetting {self._spec.tokens} at x={x}, y={y}")
        c, h, w = self._img.shape
        # print("c:", c, "h", h, "w", w)
        b = self._spec.baseline
        canvas_t = y - b
        canvas_b = canvas_t + h
        canvas_l = x + self._spec.left_spacing
        canvas_r = canvas_l + w
        # Grab the section from the canvas onto which we are going to blend our own glyph.
        bg = canvas[0:c, canvas_t:canvas_b, canvas_l:canvas_r].clone()

        back_t = bg
        front_t = self._img
        # 3. Separate RGB and Alpha
        back_rgb = back_t[:3, :, :].to(torch.int32)
        back_a = back_t[3:, :, :].to(torch.int32)
        front_rgb = front_t[:3, :, :].to(torch.int32)
        front_a = front_t[3:, :, :].to(torch.int32)
        # print("back_a:", back_a.shape)
        # print("front_a:", front_a.shape)

        # 4. Alpha Blending Formula
        # Porter-Duff Over
        # https://en.wikipedia.org/wiki/Alpha_compositing
        out_a = front_a + ((back_a * (255 - front_a)) + 127) // 255
        subterm = (back_rgb * back_a * (255 - front_a)) // 255
        out_rgb = (front_rgb * front_a + subterm) // (out_a + 1)

        # Combine RGB and Alpha
        result_t = torch.cat([out_rgb, out_a], dim=0)

        canvas[:, canvas_t:canvas_b, canvas_l:canvas_r] = result_t

    def width(self) -> int:
        return self._img.shape[2] + self._spec.right_spacing

    def is_space(self) -> bool:
        return self._spec.is_space


class Glyphset:
    def __init__(self, glyphset_yaml: Path):
        self._glyphset_yaml_path = glyphset_yaml
        with open(glyphset_yaml) as f:
            d = yaml.safe_load(f)
        spec = FontSpec.model_validate(d)
        self._spec = spec
        self.create_glyphs()
        space_glyph = GlyphSort(
            GlyphSpec(
                tokens=" ",
                baseline=1,
                is_space=True,
            ),
            torch.zeros((4, 1, self._spec.space_width), dtype=torch.uint8),
        )
        self._glyphs.append(space_glyph)
        self._glyph_sorts = {v.tokens(): v for v in self._glyphs}

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
        d = load_image_file_u8(
            self._glyphset_yaml_path.parent / self._spec.image_path, device="cpu"
        )
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
            if glyph_spec.baseline is None:
                glyph_spec.baseline = self._spec.baseline - self._spec.ascender
            if not glyph_spec.is_skipped:
                self._glyphs.append(GlyphSort(glyph_spec, glyph_image))
            line_pos = glyph_end

    def glyphs(self) -> list[GlyphSort]:
        return self._glyphs

    def typeset_width(self, tokens: list[str]) -> int:
        return self._typeset_worker(canvas=None, tokens=tokens, x=0, y=0)

    def typeset(self, canvas: Tensor, tokens: list[str], x: int, y: int):
        self._typeset_worker(canvas=canvas, tokens=tokens, x=x, y=y)

    def _typeset_worker(self, canvas: Tensor | None, tokens: list[str], x: int, y: int):
        """
        Typesets the list of tokens onto canvas, with baseline starting at pos.
        """

        xpos = x
        previous_token: GlyphSort | None = None
        for i, t in enumerate(tokens):
            # print(f"Rendering {t} at {xpos}")
            if (
                t != " "
                and previous_token is not None
                and not previous_token.is_space()
            ):
                xpos += self._spec.letter_spacing

            glyph_set = self._glyph_sorts.get(t)
            if glyph_set is None:
                raise KeyError(f"missing glyph {t}")
            if canvas is not None:
                glyph_set.typeset(canvas, x=xpos, y=y)
            xpos += glyph_set.width()

            previous_token = glyph_set
        return xpos


def run_glyphset_dump(args):
    glyphset = Glyphset(args.input)

    torchvision.utils.save_image(
        glyphset._line.to(torch.float) / 255.0, "/tmp/line.png"
    )
    for i, glyph in enumerate(glyphset.glyphs()):
        glyphimg = glyph.image().to(torch.float) / 255.0
        torchvision.utils.save_image(
            glyphimg, f"/tmp/glyph_{i:0>3}_{glyph.filename_tokens()}.png"
        )


def run_typeset(args):
    glyphset = Glyphset(args.input)
    tokens = args.text[:]
    width = args.width
    if width is None:
        width = glyphset.typeset_width(tokens)

    canvas = torch.zeros((4, args.height, width), dtype=torch.uint8)
    glyphset.typeset(canvas, tokens, x=0, y=int(args.height / 2))
    print(canvas.dtype)
    canvas = canvas.to(torch.float) / 255.0
    torchvision.utils.save_image(canvas, args.output)


if __name__ == "__main__":
    # ./letter_support.py glyphset ./example_glyphset/test_font.yaml
    parser = argparse.ArgumentParser(prog="letter_support")
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    parser_glyphset = subparsers.add_parser(
        "glyphset_dump",
        help="Obtain a glyph set, coordinates are pixel position when the pixel is selected in GIMP",
    )
    parser_glyphset.add_argument("input", type=Path)
    parser_glyphset.set_defaults(func=run_glyphset_dump)

    parser_typeset = subparsers.add_parser(
        "typeset",
        help="Typeset to a canvas",
    )
    parser_typeset.add_argument("input", type=Path)
    parser_typeset.add_argument("text", type=str)
    parser_typeset.add_argument(
        "--width",
        type=int,
        help="Canvas width, by default autodetermined",
        default=None,
    )
    parser_typeset.add_argument("--height", type=int, help="Canvas height", default=50)
    parser_typeset.add_argument(
        "-o",
        dest="output",
        type=Path,
        help="Output path",
        default=Path("/tmp/typeset.png"),
    )
    parser_typeset.set_defaults(func=run_typeset)

    # parser_test = subparsers.add_parser("test", help="Run inference")
    # parser_test.set_defaults(func=run_test)

    args = parser.parse_args()

    # Execute the selected command's function
    if args.command:
        args.func(args)
    else:
        parser.print_help()
