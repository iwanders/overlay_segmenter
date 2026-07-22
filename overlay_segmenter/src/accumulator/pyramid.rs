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
        let mut current_full_res_pos: (isize, isize) = (0, 0);
        // Go through the pyramids in reverse order, from small images to large.
        for (layer, (b, o)) in self
            .layers
            .iter()
            .rev()
            .zip(other.layers.iter().rev())
            .enumerate()
        {
            println!();
            println!("Global pos {current_full_res_pos:?}");

            // Layer 0 is a bit special, since there we just pad on all sides with the entire image size >_<
            // All other layers we only really need to do 4 left and right? which we can just do by chopping the borders
            // off the kernel.

            println!("b.date shape: {:?}", b.data.shape());
            println!("o.date shape: {:?}", o.data.shape());
            let base_img = b.data.ten()?;
            let other_img = o.data.ten()?;

            let options = if layer == 0 {
                // Padding here is
                nn::functional::Conv2dOptions {
                    padding: (b.data.isize(-2) as i64 / 2, b.data.isize(-1) as i64 / 2),
                    ..Default::default()
                }
            } else {
                nn::functional::Conv2dOptions {
                    padding: (20, 20), // 4 or 2?
                    ..Default::default()
                }
            };
            let offset;
            let (base_img, other_img) = if layer == 0 {
                offset = (0isize, 0isize);
                (base_img, other_img)
            } else {
                let mut grid = GridOverlay::new();
                let base_id = grid.add_tensor(
                    &base_img,
                    (
                        (current_full_res_pos.0 / (b.scale as isize)),
                        (current_full_res_pos.1 / (b.scale as isize)),
                    ),
                );
                let other_id = grid.add_tensor(&other_img, (0, 0));
                let (base_overlap_x, base_overlap_y) = grid.overlap_irange(base_id);
                let (other_overlap_x, other_overlap_y) = grid.overlap_irange(other_id);
                offset = (grid.overlap().0.x, grid.overlap().0.y);

                // That's in the previous dimension though, so next we need to scale that by two, and then we can slice.

                let base_overlap_x = base_overlap_x.start..base_overlap_x.end;
                let base_overlap_y = base_overlap_y.start..base_overlap_y.end;
                let other_overlap_x = other_overlap_x.start..other_overlap_x.end;
                let other_overlap_y = other_overlap_y.start..other_overlap_y.end;

                let base_img = base_img.i((.., .., base_overlap_y, base_overlap_x))?;
                let other_img = other_img.i((.., .., other_overlap_y, other_overlap_x))?;

                (base_img, other_img)
            };

            let conv2 = nn::functional::conv2d(&base_img, &other_img, None, &options)?;
            // Get the peak, check if the peak is higher than current best match.

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

            let (values, indices) = conv2.flatten(0, None)?.topk(1, &Default::default())?;

            let value = *values.cpu()?.as_f32()?;
            let indices = indices.cpu()?;
            let y = (*indices.as_i64()? as usize) / conv2.isize(-1);
            let x = (*indices.as_i64()? as usize) % conv2.isize(-1);

            // y_match = (i // output_size[1]).item()
            // x_match = (i % output_size[1]).item()
            // let rows = flat_indices // matrix.shape[1]
            // cols = flat_indices % matrix.shape[1]

            println!("shape conv2: {:?}", conv2.shape());
            //println!("conv2: {:?}", conv2);
            println!("values: {:?} {:?} {value}", values.shape(), values);

            if layer == 0 {
                current_full_res_pos = (
                    (offset.0 + conv2.isize(-1) as isize / 2 - x as isize) * b.scale as isize,
                    (offset.1 + conv2.isize(-2) as isize / 2 - y as isize) * b.scale as isize,
                );
            } else {
                current_full_res_pos.0 +=
                    (conv2.isize(-1) as isize / 2 - x as isize) * b.scale as isize;
                current_full_res_pos.1 +=
                    (conv2.isize(-2) as isize / 2 - y as isize) * b.scale as isize;
            }
            println!(
                "indices: {:?} peak at {x},{y}  at scale {}  offset {offset:?}  current_full_res_pos: {current_full_res_pos:?}",
                indices, b.scale
            );

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
                let scale = 1;
                let base_id = grid.add_tensor(&base_img, (0, 0));
                let other_id = grid.add_tensor(
                    &other_img,
                    (
                        (current_full_res_pos.0 / b.scale),
                        (current_full_res_pos.1 / b.scale),
                    ),
                );
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
