use super::{GridOverlay, grid::GridId};
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
    ) -> StableTorchResult<(isize, isize)> {
        // Position of 'other' expressed in 'base', always in the full resolution scale.
        let mut pos: (isize, isize) = (0, 0);
        // Go through the pyramids in reverse order, from small images to large.
        for (layer, (b, o)) in self
            .layers
            .iter()
            .rev()
            .zip(other.layers.iter().rev())
            .enumerate()
        {
            println!();
            println!("Global pos {pos:?}");

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

            if let Some((output_dir, output_prefix)) = debug_dir {
                let img_norm = conv2.image_scale_to_domain()?;
                let path = output_dir.join(&format!("{output_prefix}_level_{layer}_conv.png"));
                img_norm.save_image(&path)?;
                base_img.image_scale_to_domain()?.save_image(
                    &output_dir.join(&format!("{output_prefix}_level_{layer}_base.png")),
                )?;

                other_img.image_scale_to_domain()?.save_image(
                    &output_dir.join(&format!("{output_prefix}_level_{layer}_kernel.png")),
                )?;

                {
                    let h = base_img.isize(-2);
                    let w = base_img.isize(-1);
                    let options = flash_powder::factory::TensorOptions {
                        dtype: Some(base_img.dtype()),
                        device: Some(base_img.device()),
                        ..Default::default()
                    };
                    let mut canvas: Tensor = Tensor::zeros(&[3, h, w], &options)?; // base_channels
                    // Blit the base;
                    canvas
                        .i_mut((0, .., ..))?
                        .copy_from_tensor(&base_img.squeeze()?)?;
                    // And blit the other part
                    canvas
                        .i_mut((1, .., ..))?
                        .copy_from_tensor(&other_img.squeeze()?)?;

                    canvas.image_scale_to_domain()?.save_image(
                        &output_dir.join(&format!("{output_prefix}_aligned_pre_{layer}.png")),
                    )?;
                }
            }

            let (score, dx, dy) = conv_peak(&conv2)?;
            // best_value = value;
            pos.0 += dx * b.scale;
            pos.1 += dy * b.scale;

            if let Some((output_dir, output_prefix)) = debug_dir {
                let mut grid = GridOverlay::new();

                let base_img = self
                    .layers
                    .last()
                    .as_ref()
                    .map(|z| z.data.ten().unwrap())
                    .unwrap();
                let other_img = self
                    .layers
                    .last()
                    .as_ref()
                    .map(|z| z.data.ten().unwrap())
                    .unwrap();
                let base_id = grid.add_tensor(&base_img, (0, 0));
                let other_id = grid.add_tensor(&other_img, ((pos.0 / b.scale), (pos.1 / b.scale)));
                let full = grid.full_size();
                println!("full: {full:?}");
                let options = flash_powder::factory::TensorOptions {
                    dtype: Some(base_img.dtype()),
                    device: Some(base_img.device()),
                    ..Default::default()
                };
                let mut canvas: Tensor = Tensor::zeros(&[3, full.1, full.0], &options)?; // base_channels
                // Blit the base;
                let (bx, by) = grid.full_grid_irange(base_id);
                canvas
                    .i_mut((0, by, bx))?
                    .copy_from_tensor(&base_img.squeeze()?)?;
                // And blit the other part
                let (bx, by) = grid.full_grid_irange(other_id);
                canvas
                    .i_mut((1, by, bx))?
                    .copy_from_tensor(&other_img.squeeze()?)?;

                canvas.image_scale_to_domain()?.save_image(
                    &output_dir.join(&format!("{output_prefix}_aligned_{layer}.png")),
                )?;
            }
        }

        Ok((0, 0))
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
