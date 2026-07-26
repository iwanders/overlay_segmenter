use flash_powder as fp;
use flash_powder::prelude::*;
use flash_powder::{Ten, Tensor, nn};
use flash_powder_image::prelude::*;
use std::path::Path;
pub mod grid;
use grid::{GridOverlay, Position};
pub mod pyramid;
use pyramid::Pyramid;
use serde::{Deserialize, Serialize};

use fp::StableTorchResult;

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
struct FrameMatch {
    pub frame_index: usize,
    pub value: f32,
    pub pos: Position,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct FrameRelation {
    matches: Vec<FrameMatch>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Accumulator {
    #[serde(with = "crate::serde_tensor::vec_tensor")]
    frames: Vec<Tensor>,
    pyramids: Vec<Pyramid>,
    frame_relations: Vec<FrameRelation>,
}
impl Accumulator {
    pub fn new() -> Self {
        Self {
            frames: vec![],
            pyramids: vec![],
            frame_relations: vec![],
        }
    }
    pub fn feed_logits_frame(
        &mut self,
        frame: &Ten<'_>,

        debug_dir: Option<&Path>,
    ) -> StableTorchResult<()> {
        let new_frame_index = self.frames.len();
        let frame = frame.to(&fp::DType::F32.into())?;

        // Should we do a softmax here? or do we just run with the raw logits?
        let frame = nn::functional::softmax_int(&frame, 1, None)?;

        let multi_res_stack = pyramid::Pyramid::new(frame.i((0, 1, .., ..))?, 3)?;

        let new_frame_prefix = format!("{new_frame_index}");

        multi_res_stack
            .dump_pyramid(debug_dir.as_ref().map(|a| (*a, new_frame_prefix.as_str())))?;

        // Compare it against all the existing frames, because why not.
        // Hmm, we should probably just compare against a running composite...
        let mut matches = vec![];
        for (i, _base_frame) in self.frames.iter().enumerate() {
            let other_pyramid = &self.pyramids[i];
            let this_frame_prefix = format!("{i}");
            let (value, pos) = multi_res_stack.pyramid_aligner(
                &other_pyramid,
                debug_dir.as_ref().map(|a| (*a, this_frame_prefix.as_str())),
            )?;
            // Record the alignment of this new frame against the existing frame `i`.
            matches.push(FrameMatch {
                frame_index: i,
                value,
                pos,
            });
        }

        self.pyramids.push(multi_res_stack);
        self.frames.push(frame);
        self.frame_relations.push(FrameRelation { matches });

        Ok(())
    }

    fn create_grid(&self) -> StableTorchResult<GridOverlay> {
        // Iterate over all the frames, collect the best scoring one, then create the composite.
        let mut grid = GridOverlay::new();
        let mut grid_ids = vec![];
        for (i, base) in self.frame_relations.iter().enumerate() {
            println!("frame {i}");

            let mut best_entry: Option<(f32, Position, FrameMatch)> = None;
            for other in base.matches.iter() {
                if let Some((best_score, _, _)) = best_entry.as_ref() {
                    if best_score < &other.value {
                        best_entry = Some((other.value, other.pos, other.clone()));
                    }
                } else {
                    best_entry = Some((other.value, other.pos, other.clone()));
                }
            }

            if let Some((score, p, frame_match)) = best_entry {
                println!(" score: {score}");
                println!(" pos  : {p:?}");
                println!(" to   : {:?}", frame_match.frame_index);

                let parent_pos = grid.grid_position(grid_ids[frame_match.frame_index]);

                grid_ids.push(grid.add_tensor(
                    &self.frames[i].ten()?,
                    (parent_pos + frame_match.pos).into(),
                ));
            } else {
                grid_ids.push(grid.add_tensor(&self.frames[i].ten()?, (0, 0)));
            }
        }

        Ok(grid)
    }

    pub fn debug_use_accumulation(&self, debug_dir: &Path) -> StableTorchResult<()> {
        let grid = self.create_grid()?;
        // Stack the grid, round robin channel.
        let (w, h) = grid.full_size();

        let options = flash_powder::factory::TensorOptions {
            dtype: Some(self.frames[0].dtype()),
            device: Some(self.frames[0].device()),
            ..Default::default()
        };
        let mut canvas: Tensor = Tensor::zeros(&[3, h, w], &options)?;

        let mut channel = 0;
        for (grid_id, frame) in grid.ids().zip(self.frames.iter()) {
            let (ax, ay) = grid.full_grid_irange(grid_id);
            canvas
                .i_mut((channel, ay, ax))?
                .copy_from_tensor(&frame.i((0, 1, .., ..))?.squeeze()?)?;
            channel = (channel + 1) % 3;
        }
        canvas
            .image_scale_to_domain()?
            .save_image(debug_dir.join(format!("accumulated.png")))?;

        Ok(())
    }

    fn inflated_non_zero_present(&self, frame: Ten<'_>) -> StableTorchResult<Tensor> {
        let indices_at_pixel = frame.squeeze()?.argmax(Some(0), Some(true))?;
        let zero_i64: Tensor = 0i64.try_into()?;
        let zero_f32: Tensor = 0.0f32.try_into()?;
        let non_bg = indices_at_pixel.ne(&zero_i64)?;

        // Next, we convert that back to float sand convolute it with the inflation.
        let non_bg_f32 = non_bg.to(&fp::DType::F32.into())?;

        let convolution_circle = pyramid::circle_image(21, 21, 10, 10, 10)?;
        convolution_circle
            .image_scale_to_domain()?
            .save_image("/tmp/convolution_kernel.png")?;

        // Next, do the actual convolution.
        let padding = (10, 10);
        let options = nn::functional::Conv2dOptions {
            padding,
            ..Default::default()
        };
        let non_bg_f32 = non_bg_f32.unsqueeze(0)?;
        let conv_mask = convolution_circle.unsqueeze(0)?.unsqueeze(0)?;
        println!("non_bg_f32 shape: {:?}", non_bg_f32.shape());
        println!("conv_mask shape: {:?}", conv_mask.shape());
        let conv2 = nn::functional::conv2d(&non_bg_f32, &conv_mask, None, &options)?;
        conv2
            .image_scale_to_domain()?
            .save_image("/tmp/conv_result.png")?;

        let boolean_mask = conv2.gt(&zero_f32)?;
        let non_bg = boolean_mask.squeeze()?;
        println!("non_bg shape: {:?}", non_bg.shape());
        Ok(non_bg.to_owned()?)
    }

    // This currently breaks down as we don't account for the 'revealed' area, and only sections that are in view
    // longer than they are obscured will work, results for those is super crisp though.
    // Count should use an inflated boolean mask around non-background.
    pub fn combined_avg(&self, debug_dir: &Path) -> StableTorchResult<Tensor> {
        let grid = self.create_grid()?;
        // Stack the grid, round robin channel.
        let (w, h) = grid.full_size();
        let channels = self.frames.first().map(|a| a.isize(-3)).unwrap_or(0);
        // let batch = self.frames.first().map(|a| a.isize(-4)).unwrap_or(0);
        let fdtype = self.frames[0].dtype();
        println!("fdtype: {fdtype:?}");

        let options = flash_powder::factory::TensorOptions {
            dtype: Some(fdtype),
            device: Some(self.frames[0].device()),
            ..Default::default()
        };
        let mut canvas: Tensor = Tensor::zeros(&[channels, h, w], &options)?;
        let mut counts: Tensor = Tensor::zeros(
            &[1, h, w],
            &flash_powder::factory::TensorOptions {
                dtype: Some(fp::DType::I64),
                device: Some(self.frames[0].device()),
                ..Default::default()
            },
        )?;
        let counts_one = Tensor::ones(
            &[1, h, w],
            &flash_powder::factory::TensorOptions {
                dtype: Some(fp::DType::I64),
                device: Some(self.frames[0].device()),
                ..Default::default()
            },
        )?;
        println!("counts_one shape: {:?}", counts_one.shape());

        for (grid_id, frame) in grid.ids().zip(self.frames.iter()) {
            let (ax, ay) = grid.full_grid_irange(grid_id);
            // First add the actual values.
            let current_values = canvas.i((.., ay.clone(), ax.clone()))?;
            let with_addition = current_values.add(&frame.squeeze()?)?;
            canvas
                .i_mut((.., ay.clone(), ax.clone()))?
                .copy_from_tensor(&with_addition)?;

            // Then add the counts.
            let current_values = counts.i((.., ay.clone(), ax.clone()))?;
            println!("current_values shape: {:?}", current_values.shape());
            // let with_addition =
            //     current_values.add(&counts_one.i((.., ay.clone(), ax.clone()))?)?;
            let presence_mask_with_ones = self.inflated_non_zero_present(frame.ten()?)?;
            presence_mask_with_ones
                .image_scale_to_domain()?
                .save_image(debug_dir.join(format!("presence_mask_with_ones_{:?}.png", grid_id)))?;

            let with_addition = current_values.add(&presence_mask_with_ones)?;
            println!("with_addition shape: {:?}", with_addition.shape());
            counts
                .i_mut((.., ay, ax))?
                .copy_from_tensor(&with_addition)?;
        }

        // Now that we are here, we can divide the actual values by the counts... and we should get a pristine image.
        // Acounts contains zeros, which is a problem as it blows up the values to nan.
        let zero: Tensor = 0i64.try_into()?;
        //let all_positive_ones = counts.ne(&zero)?.squeeze()?.to_owned()?;
        let positions_with_zero = counts.eq(&zero)?.squeeze()?.to_owned()?;
        let zeros_made_into_ones_but_counts_else = counts.add(&positions_with_zero)?;

        let r = canvas.div(&zeros_made_into_ones_but_counts_else.to(&fdtype.into())?)?;

        Ok(r)
    }

    pub fn write_postcard<Q: AsRef<Path>>(&mut self, output_path: Q) -> StableTorchResult<()> {
        let data = postcard::to_allocvec(self)?;
        println!("dta is {:?}", data.len());
        std::fs::write(output_path, &data).map_err(|e| anyhow::format_err!(e))
    }
    pub fn read_postcard<Q: AsRef<Path>>(output_path: Q) -> StableTorchResult<Self> {
        let data = std::fs::read(output_path).map_err(|e| anyhow::format_err!(e))?;
        let v: Self = postcard::from_bytes(&data)?;
        Ok(v)
    }
}
