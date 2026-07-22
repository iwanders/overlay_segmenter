use super::{GridOverlay, Position};
use flash_powder::nn;
use flash_powder::prelude::*;
use flash_powder::{StableTorchResult, Ten, Tensor};
use flash_powder_image::prelude::*;

#[derive(Debug, Clone)]
struct Layer {
    scale: isize,
    data: Tensor,
}

/// A multiresolution image stack.
#[derive(Debug, Clone)]
pub struct Pyramid {
    layers: Vec<Layer>,
}
impl Pyramid {
    pub fn new(input_data: Ten<'_>, layer_count: usize) -> StableTorchResult<Pyramid> {
        let mut scale = 1;
        let mut data = input_data.unsqueeze(0)?.unsqueeze(0)?.to_owned()?;
        let mut layers = vec![Layer {
            scale: 1,
            data: data.clone(),
        }];
        for _i in 0..layer_count {
            scale = scale * 2;
            data = data.image_resize(
                [data.isize(-2) / 2, data.isize(-1) / 2],
                nn::functional::InterpolateAlgorithm::Bilinear,
            )?;
            layers.push(Layer {
                scale: scale,
                data: data.clone(),
            });
        }

        Ok(Self { layers })
    }

    pub fn get_tensor(&self, index: usize) -> Option<Ten<'_>> {
        self.layers.get(index).map(|l| l.data.ten().ok()).flatten()
    }

    pub fn dump_pyramid(
        &self,
        debug_dir: Option<(&std::path::Path, &str)>,
    ) -> StableTorchResult<()> {
        if let Some((debug_path, prefix)) = debug_dir {
            for (i, layer) in self.layers.iter().enumerate() {
                layer
                    .data
                    .image_scale_to_domain()?
                    .save_image(debug_path.join(&format!("pyramid_{prefix}_layer_{i}.png")))?;
            }
        }
        Ok(())
    }

    pub fn pyramid_aligner(
        &self,
        other: &Pyramid,

        debug_dir: Option<(&std::path::Path, &str)>,
    ) -> StableTorchResult<(f32, Position)> {
        // Position of 'other' expressed in 'base', always in the full resolution scale.
        let mut pos: (isize, isize) = (0, 0);
        let mut score = 0.0;
        // Go through the pyramids in reverse order, from small images to large.
        for (layer, (b, o)) in self
            .layers
            .iter()
            .rev()
            .zip(other.layers.iter().rev())
            .enumerate()
        {
            let base_img = b.data.ten()?;
            let other_img = o.data.ten()?;

            let (base_img, other_img, padding) = if layer == 0 {
                // At the first layer we do a full search, with half the dimensions of padding, this ensures that we try
                // all the overlap, but only at this resolution.
                let padding = (base_img.isize(-2) as i64 / 2, base_img.isize(-1) as i64 / 2);
                (base_img, other_img, padding)
            } else {
                // At the other layers, we do a more minimal search with just the padding that we need to compensate
                // of the resolution loss.
                let base_pos = (pos.0 / b.scale, pos.1 / b.scale);
                let mut grid = GridOverlay::new();
                let base_id = grid.add_tensor(&base_img, base_pos);
                let other_id = grid.add_tensor(&other_img, (0, 0));
                let (bx, by) = grid.overlap_irange(base_id);
                let (ox, oy) = grid.overlap_irange(other_id);
                let base_img = base_img.i((.., .., by, bx))?;
                let other_img = other_img.i((.., .., oy, ox))?;
                (base_img, other_img, (SEARCH_PADDING, SEARCH_PADDING))
            };

            let options = nn::functional::Conv2dOptions {
                padding,
                ..Default::default()
            };
            let conv2 = nn::functional::conv2d(&base_img, &other_img, None, &options)?;

            if let Some((dir, prefix)) = debug_dir {
                Self::dump_correlation(dir, prefix, layer, &base_img, &other_img, &conv2.ten()?)?;
            }

            let (this_score, dx, dy) = conv_peak(&conv2)?;
            score = this_score;
            pos.0 += dx * b.scale;
            pos.1 += dy * b.scale;

            if let Some((dir, prefix)) = debug_dir {
                let base_pos = (pos.0 / b.scale, pos.1 / b.scale);
                Self::dump_alignment(dir, prefix, layer, &b.data.ten()?, &o.data.ten()?, base_pos)?;
            }
        }

        Ok((score, Position::new(pos.0, pos.1)))
    }

    /// Saves the correlation output, both correlated crops, and their red/green overlay.
    fn dump_correlation(
        output_dir: &std::path::Path,
        prefix: &str,
        layer: usize,
        base: &Ten<'_>,
        other: &Ten<'_>,
        conv: &Ten<'_>,
    ) -> StableTorchResult<()> {
        conv.image_scale_to_domain()?
            .save_image(output_dir.join(format!("{prefix}_level_{layer}_conv.png")))?;
        base.image_scale_to_domain()?
            .save_image(output_dir.join(format!("{prefix}_level_{layer}_base.png")))?;
        other
            .image_scale_to_domain()?
            .save_image(output_dir.join(format!("{prefix}_level_{layer}_kernel.png")))?;

        let canvas = overlay_tensors(base, (0, 0), other, (0, 0))?;
        canvas
            .image_scale_to_domain()?
            .save_image(output_dir.join(format!("{prefix}_aligned_pre_{layer}.png")))
    }

    /// Saves a red/green overlay of `base` (offset by `base_pos`) and `other` (at the origin).
    fn dump_alignment(
        output_dir: &std::path::Path,
        prefix: &str,
        layer: usize,
        base: &Ten<'_>,
        other: &Ten<'_>,
        base_pos: (isize, isize),
    ) -> StableTorchResult<()> {
        let canvas = overlay_tensors(base, base_pos, other, (0, 0))?;
        canvas
            .image_scale_to_domain()?
            .save_image(output_dir.join(format!("{prefix}_aligned_{layer}.png")))
    }
}

/// Bounded search radius (in pixels) used when refining an estimate at the finer layers.
const SEARCH_PADDING: i64 = 2;

/// Returns the `(value, dx, dy)` of a correlation's peak, where `dx`/`dy` are offsets from
/// the centre of the correlation output.
fn conv_peak(conv: &Tensor) -> StableTorchResult<(f32, isize, isize)> {
    let (values, indices) = conv.flatten(0, None)?.topk(1, &Default::default())?;
    let value = *values.cpu()?.as_f32()?;
    let index = *indices.cpu()?.as_i64()? as isize;
    let width = conv.isize(-1) as isize;
    let (x, y) = (index % width, index / width);
    Ok((value, width / 2 - x, conv.isize(-2) as isize / 2 - y))
}

/// Composes `a` and `b` onto a single canvas (a in the red channel, b in the green channel),
/// placed at the given positions, sized to their bounding box.
fn overlay_tensors(
    a: &Ten<'_>,
    a_pos: (isize, isize),
    b: &Ten<'_>,
    b_pos: (isize, isize),
) -> StableTorchResult<Tensor> {
    let mut grid = GridOverlay::new();
    let a_id = grid.add_tensor(a, a_pos);
    let b_id = grid.add_tensor(b, b_pos);
    let (w, h) = grid.full_size();

    let options = flash_powder::factory::TensorOptions {
        dtype: Some(a.dtype()),
        device: Some(a.device()),
        ..Default::default()
    };
    let mut canvas: Tensor = Tensor::zeros(&[3, h, w], &options)?;
    let (ax, ay) = grid.full_grid_irange(a_id);
    canvas.i_mut((0, ay, ax))?.copy_from_tensor(&a.squeeze()?)?;
    let (bx, by) = grid.full_grid_irange(b_id);
    canvas.i_mut((1, by, bx))?.copy_from_tensor(&b.squeeze()?)?;
    Ok(canvas)
}

#[cfg(test)]
mod test {
    use super::*;
    use flash_powder::DType;
    use flash_powder::tensor::BlobOptionsBytes;

    /// Builds an `h x w` f32 image with a filled disk of radius `r` centered at `(cx, cy)`.
    fn circle_image(
        h: usize,
        w: usize,
        cx: isize,
        cy: isize,
        r: isize,
    ) -> StableTorchResult<Tensor> {
        let mut data = vec![0f32; h * w];
        for y in 0..h as isize {
            for x in 0..w as isize {
                let (ddx, ddy) = (x - cx, y - cy);
                if ddx * ddx + ddy * ddy <= r * r {
                    data[(y as usize) * w + (x as usize)] = 1.0;
                }
            }
        }
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let ten = Ten::from_bytes(
            &bytes,
            &BlobOptionsBytes {
                sizes: &[h, w],
                strides: &[w, 1],
                dtype: DType::F32,
            },
        )?;
        ten.to_owned()
    }

    #[test]
    fn test_pyramid_aligner_recovers_offset() -> StableTorchResult<()> {
        let (h, w) = (128usize, 128usize);
        let r = 16;
        // `other`'s disk is shifted by (+20, +12) in (x, y) relative to `base`'s.
        let (shift_x, shift_y) = (20isize, 12isize);
        let base = circle_image(h, w, 64, 64, r)?;
        let other = circle_image(h, w, 64 + shift_x, 64 + shift_y, r)?;
        other.save_image("/tmp/circle_other.png")?;

        let base_p = Pyramid::new(base.ten()?, 3)?;
        let other_p = Pyramid::new(other.ten()?, 3)?;

        let (value, pos) = base_p.pyramid_aligner(&other_p, None)?;
        println!("shift=({shift_x},{shift_y}) recovered=({pos:?}) value={value}");

        // `pyramid_aligner` returns the position of `other` relative to `self`, so the
        // recovered offset should equal the shift we applied (within a pixel).
        assert!(
            (pos.x - shift_x).abs() <= 1,
            "dx={}, expected {shift_x}",
            pos.x
        );
        assert!(
            (pos.y - shift_y).abs() <= 1,
            "dy={}, expected {shift_y}",
            pos.y
        );
        assert!(value > 0.0);
        Ok(())
    }
}
