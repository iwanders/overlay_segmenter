use flash_powder as fp;
use flash_powder::prelude::*;
use flash_powder::{Ten, TenMut, Tensor, nn};
use flash_powder_image::prelude::*;
use std::path::Path;
pub mod grid;
use grid::{GridOverlay, Position};
pub mod pyramid;
use pyramid::Pyramid;
use serde::{Deserialize, Serialize};

use fp::StableTorchResult;

/*


Currently, this works by:
    Take logit frame
        Softmax
        Convolute against previous frames, capture score, pos,
        store frame and positions against previous frames.
    Accumulate:
        Iterate over all the frames
        Position each frame relative to its parent (best scoring relation)
        Do the whole masked merge.


We want do to this all in a live way, so not keep around all the frames.
    We could keep around the logit sum + counts, and do the masking on that at the end.
    In which case we'd use the Accumulator to keep a sliding window.

    tick    :              0            1             2             3            4
    frame 0                <-------------------------->
    frame 1                             <-------------------------->
    frame 2                                           <-------------------------->
                                        ^
                                        merge frame 1 away
    We can keep frame relations for frames that are merged away I guess... we only need to keep the history on the left
    side... Then we can pick the best out of the history, and then update the anchor of the accumulator and remove
    a frame?

    In the overall accumulator we can keep \sigma softmax(logits), counts



Keep the total accumulated result away from the logit frames, otherwise the total result becomes slower and slower
as time goes on and map size grows? But at the cost of drift... maybe we should anchor against the accumulated result?

*/

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
struct FrameMatch {
    pub frame_index: usize,
    pub value: f32,
    pub pos: Position,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FrameRelation {
    matches: Vec<FrameMatch>,
}
impl FrameRelation {
    fn best_match(&self) -> Option<(f32, Position, FrameMatch)> {
        let mut best_entry: Option<(f32, Position, FrameMatch)> = None;
        for other in self.matches.iter() {
            if let Some((best_score, _, _)) = best_entry.as_ref() {
                if best_score < &other.value {
                    best_entry = Some((other.value, other.pos, other.clone()));
                }
            } else {
                best_entry = Some((other.value, other.pos, other.clone()));
            }
        }
        best_entry
    }
}

fn inflated_non_zero_present(frame: Ten<'_>, area_radius: usize) -> StableTorchResult<Tensor> {
    let indices_at_pixel = frame.squeeze()?.argmax(Some(0), Some(true))?;
    let zero_i64: Tensor = 0i64.try_into()?;
    let zero_i64 = zero_i64.to(&indices_at_pixel.device().into())?;
    let zero_f32: Tensor = 0.0f32.try_into()?;
    let zero_f32 = zero_f32.to(&indices_at_pixel.device().into())?;
    let non_bg = indices_at_pixel.ne(&zero_i64)?;

    // Next, we convert that back to float sand convolute it with the inflation.
    let non_bg_f32 = non_bg.to(&fp::DType::F32.into())?;

    let convolution_circle = pyramid::circle_image(
        area_radius * 2 + 1,
        area_radius * 2 + 1,
        area_radius as isize,
        area_radius as isize,
        area_radius as isize,
    )?;
    let convolution_circle = convolution_circle.to(&frame.device().into())?;
    // convolution_circle
    //     .image_scale_to_domain()?
    //     .save_image("/tmp/convolution_kernel.png")?;

    // Next, do the actual convolution.
    let padding = (area_radius as _, area_radius as _);
    let options = nn::functional::Conv2dOptions {
        padding,
        ..Default::default()
    };
    let non_bg_f32 = non_bg_f32.unsqueeze(0)?;
    let conv_mask = convolution_circle.unsqueeze(0)?.unsqueeze(0)?;

    let conv2 = nn::functional::conv2d(&non_bg_f32, &conv_mask, None, &options)?;
    // conv2
    //     .image_scale_to_domain()?
    //     .save_image("/tmp/conv_result.png")?;

    let boolean_mask = conv2.gt(&zero_f32)?;
    let non_bg = boolean_mask.squeeze()?;
    Ok(non_bg.to_owned()?)
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub struct AccumulationConfig {
    pub fit_against_previous_frames: usize,
    pub min_observations: usize,
    pub area_radius: usize,
    pub layer_count: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Accumulation {
    #[serde(with = "crate::serde_tensor::tensor")]
    accumulation_values: Tensor,
    #[serde(with = "crate::serde_tensor::tensor")]
    accumulation_counts: Tensor,
    config: AccumulationConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LocalizedFrame {
    #[serde(with = "crate::serde_tensor::tensor")]
    frame: Tensor,
    pyramid: Pyramid,
    frame_relation: FrameRelation,
}
impl LocalizedFrame {
    pub fn into_device(self, device: &fp::Device) -> StableTorchResult<Self> {
        Ok(Self {
            frame: self.frame.to(&flash_powder::factory::ToOptions {
                device: Some(*device),
                ..Default::default()
            })?,
            pyramid: self.pyramid.into_device(*device)?,
            frame_relation: self.frame_relation,
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Accumulator {
    localized_frames: Vec<LocalizedFrame>,

    accumulation: Option<Accumulation>,
}
impl Accumulator {
    pub fn new() -> Self {
        Self {
            localized_frames: vec![],
            accumulation: None,
        }
    }
    pub fn enable_accumulator(&mut self, config: AccumulationConfig) -> StableTorchResult<()> {
        self.accumulation = Some(Accumulation {
            accumulation_values: Tensor::zeros(&[], &Default::default())?,
            accumulation_counts: Tensor::zeros(&[], &Default::default())?,
            config,
        });
        Ok(())
    }

    pub fn accumulate_logits_frame(&mut self, frame: &Ten<'_>) -> StableTorchResult<()> {
        if self.accumulation.is_none() {
            anyhow::bail!("trying to accumulate without config");
        }

        let config = self.accumulation.as_ref().map(|a| a.config).unwrap();

        self.feed_logits_frame(frame, config.layer_count, None)?;

        if self.localized_frames.len() > config.fit_against_previous_frames {
            // Stack 'n' frames together, filter them, and merge them into the main accumulation.
            let mut frames = vec![];
            let mut current_pos = Position::origin();
            for localized_frame in self
                .localized_frames
                .iter()
                .take(config.fit_against_previous_frames)
            {
                let offset = localized_frame
                    .frame_relation
                    .best_match()
                    .map(|(_, p, _)| p)
                    .unwrap_or(Position::origin());
                current_pos = current_pos + offset;
                frames.push((current_pos, frame.lazy_clone()?));
            }

            let (values, counts) = Self::combine_frames(&frames, config.area_radius)?;
            // Now we need to merge values into the accrued values.
            // Skip that for now.
            // Pop the frame from the front.
            self.localized_frames.remove(0);
            // Update the frame relations, also make sure that we incorporate the position...
            // Ugh, these indices don't match anymore.
            //
            todo!("make frame indices stable")
        }

        Ok(())
    }

    pub fn feed_logits_frame(
        &mut self,
        frame: &Ten<'_>,
        layer_count: usize,
        debug_dir: Option<&Path>,
    ) -> StableTorchResult<()> {
        let new_frame_index = self.localized_frames.len();
        let frame = frame.to(&fp::DType::F32.into())?;

        // Should we do a softmax here? or do we just run with the raw logits?
        let frame = nn::functional::softmax_int(&frame, 1, None)?;

        // Layer count 3 = 30
        // layer count 4 = 20 # seems like a reasonable resolution still?
        let multi_res_stack = pyramid::Pyramid::new(frame.i((0, 1, .., ..))?, layer_count)?;

        let new_frame_prefix = format!("{new_frame_index}");

        multi_res_stack
            .dump_pyramid(debug_dir.as_ref().map(|a| (*a, new_frame_prefix.as_str())))?;

        // Compare it against all the existing frames, because why not.
        // Hmm, we should probably just compare against a running composite...
        let mut matches = vec![];
        for (i, _base_frame) in self.localized_frames.iter().enumerate() {
            let other_pyramid = &self.localized_frames[i].pyramid;

            let this_frame_prefix = format!("{i}");
            let (value, pos) = multi_res_stack.pyramid_aligner(
                &other_pyramid,
                debug_dir.as_ref().map(|a| (*a, this_frame_prefix.as_str())),
            )?;
            println!("Frame {i} with new: {value:?}, at {pos:?}");
            // Record the alignment of this new frame against the existing frame `i`.
            matches.push(FrameMatch {
                frame_index: i,
                value,
                pos,
            });
        }

        self.localized_frames.push(LocalizedFrame {
            frame,
            pyramid: multi_res_stack,
            frame_relation: FrameRelation { matches },
        });

        Ok(())
    }

    fn create_grid(&self) -> StableTorchResult<GridOverlay> {
        // Iterate over all the frames, collect the best scoring one, then create the composite.
        let mut grid = GridOverlay::new();
        let mut grid_ids = vec![];
        for (i, base) in self.localized_frames.iter().enumerate() {
            println!("frame {i}");

            let best_entry: Option<(f32, Position, FrameMatch)> = base.frame_relation.best_match();

            if let Some((score, p, frame_match)) = best_entry {
                println!(" score: {score}");
                println!(" pos  : {p:?}");
                println!(" to   : {:?}", frame_match.frame_index);

                let parent_pos = grid.grid_position(grid_ids[frame_match.frame_index]);
                println!(
                    " parent_pos   : {:?}    (parent_pos + frame_match.pos): {:?}",
                    parent_pos,
                    (parent_pos + frame_match.pos)
                );

                grid_ids.push(
                    grid.add_tensor(&base.frame.ten()?, (parent_pos + frame_match.pos).into()),
                );
            } else {
                grid_ids.push(grid.add_tensor(&base.frame.ten()?, (0, 0)));
            }
        }

        Ok(grid)
    }

    pub fn debug_use_accumulation(&self, debug_dir: &Path) -> StableTorchResult<()> {
        let grid = self.create_grid()?;
        // Stack the grid, round robin channel.
        let (w, h) = grid.full_size();

        let options = flash_powder::factory::TensorOptions {
            dtype: Some(self.localized_frames[0].frame.dtype()),
            device: Some(self.localized_frames[0].frame.device()),
            ..Default::default()
        };
        let mut canvas: Tensor = Tensor::zeros(&[3, h, w], &options)?;

        let mut channel = 0;
        for (grid_id, localized_frame) in grid.ids().zip(self.localized_frames.iter()) {
            let (ax, ay) = grid.full_grid_irange(grid_id);
            canvas
                .i_mut((channel, ay, ax))?
                .copy_from_tensor(&localized_frame.frame.i((0, 1, .., ..))?.squeeze()?)?;
            channel = (channel + 1) % 3;
        }
        canvas
            .image_scale_to_domain()?
            .save_image(debug_dir.join(format!("accumulated.png")))?;

        Ok(())
    }

    pub fn combine_frames(
        frames: &[(Position, Tensor)],
        area_radius: usize,
    ) -> StableTorchResult<(Tensor, Tensor)> {
        let mut grid = GridOverlay::new();
        for (position, t) in frames.iter() {
            grid.add_tensor(t, (position.x, position.y));
        }
        if frames.is_empty() {
            anyhow::bail!("no frames provided")
        }
        let first_frame = &frames.first().unwrap().1;
        let channels = first_frame.isize(-3);
        // let batch = self.frames.first().map(|a| a.isize(-4)).unwrap_or(0);
        let fdtype = first_frame.dtype();

        let options = flash_powder::factory::TensorOptions {
            dtype: Some(fdtype),
            device: Some(first_frame.device()),
            ..Default::default()
        };
        let (w, h) = grid.full_size();
        let mut values: Tensor = Tensor::zeros(&[channels, h, w], &options)?;
        let mut counts: Tensor = Tensor::zeros(
            &[1, h, w],
            &flash_powder::factory::TensorOptions {
                dtype: Some(fp::DType::I64),
                device: Some(first_frame.device()),
                ..Default::default()
            },
        )?;

        for (grid_id, (_pos, frame)) in grid.ids().zip(frames.iter()) {
            let (ax, ay) = grid.full_grid_irange(grid_id);

            // Then add the counts.
            let current_counts = counts.i((.., ay.clone(), ax.clone()))?;
            let presence_mask_with_ones = inflated_non_zero_present(frame.ten()?, area_radius)?;

            let with_addition = current_counts.add(&presence_mask_with_ones)?;
            counts
                .i_mut((.., ay.clone(), ax.clone()))?
                .copy_from_tensor(&with_addition)?;

            // Next, we can add the values, but we also multiply those by the mask we just created.
            // Such that we only consider points in the vicinity.
            let current_values = values.i((.., ay.clone(), ax.clone()))?;
            let with_addition =
                current_values.add(&frame.squeeze()?.mul(&presence_mask_with_ones)?)?;
            values
                .i_mut((.., ay.clone(), ax.clone()))?
                .copy_from_tensor(&with_addition)?;
        }

        Ok((values, counts))
    }

    // This currently breaks down as we don't account for the 'revealed' area, and only sections that are in view
    // longer than they are obscured will work, results for those is super crisp though.
    // Count should use an inflated boolean mask around non-background.
    pub fn combined_avg(
        &self,
        area_radius: usize,
        min_observations: usize,
        debug_dir: Option<&Path>,
    ) -> StableTorchResult<Tensor> {
        let mut frames = vec![];
        let mut current_pos = Position::origin();
        for localized_frame in self.localized_frames.iter() {
            let offset = localized_frame
                .frame_relation
                .best_match()
                .map(|(_, p, _)| p)
                .unwrap_or(Position::origin());
            current_pos = current_pos + offset;
            frames.push((current_pos, localized_frame.frame.lazy_clone()?));
        }

        let (values, counts) = Self::combine_frames(&frames, area_radius)?;

        // Now that we are here, we can divide the actual values by the counts... and we should get a pristine image.
        // Acounts contains zeros, which is a problem as it blows up the values to nan, so on the locations where the
        // count is actually zero, we just add one, this shouldn't matter as the values there should be zero.
        let zero: Tensor = 0i64.try_into()?;
        let positions_with_zero = counts.eq(&zero)?.squeeze()?.to_owned()?;
        let zeros_made_into_ones_but_counts_else = counts.add(&positions_with_zero)?;

        if let Some(debug_dir) = debug_dir {
            zeros_made_into_ones_but_counts_else
                .image_scale_to_domain()?
                .save_image(debug_dir.join(format!("zeros_made_into_ones_but_counts_else.png")))?;
        }

        let r = values.div(&zeros_made_into_ones_but_counts_else.to(&values.dtype().into())?)?;

        // We can do this to further clean it up, accepting only data where 'n' observations were present.
        if min_observations >= 2 {
            let two: Tensor = (min_observations as i64).try_into()?;
            let positions_with_twos = counts.ge(&two)?.squeeze()?.to_owned()?;
            let r = r.mul(&positions_with_twos)?;
            return Ok(r);
        }

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

    pub fn into_device(mut self, device: fp::Device) -> StableTorchResult<Self> {
        let mut localized_frames = vec![];
        for p in self.localized_frames.drain(..) {
            localized_frames.push(p.into_device(&device)?);
        }

        Ok(Self {
            localized_frames,
            accumulation: self.accumulation,
        })
    }
    pub fn frame_count(&self) -> usize {
        self.localized_frames.len()
    }

    pub fn pop_left(&mut self) -> Option<(Tensor, FrameRelation, Pyramid)> {
        if self.frame_count() > 1 {
            let r = self.localized_frames.remove(0);
            let LocalizedFrame {
                frame,
                pyramid,
                frame_relation,
            } = r;
            Some((frame, frame_relation, pyramid))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use pyramid::circle_image;

    fn make_circle_logits(cx: isize, cy: isize) -> StableTorchResult<Tensor> {
        let (h, w) = (128usize, 128usize);
        let r = 16;
        // `other`'s disk is shifted by (+20, +12) in (x, y) relative to `base`'s.
        let other = circle_image(h, w, cx, cy, r)?;

        let zero: Tensor = 0.0f32.try_into()?;
        let one: Tensor = 3.0f32.try_into()?;
        let base = other.eq(&zero)?.mul(&one)?;

        // let black = Tensor::zeros(&[h, w], &Default::default())?;
        // let black = black;
        let zero: Tensor = 0.0f32.try_into()?;
        let half: Tensor = 0.01f32.try_into()?;
        let black = other.mul(&half)?;

        // Next we need to stack that.

        fp::torch::stack(&[base.ten()?, other.ten()?, black.ten()?], 0)?
            .unsqueeze(0)?
            .to_owned()
    }

    #[test]
    fn test_accumulator() -> StableTorchResult<()> {
        let frame_0 = make_circle_logits(30, 30)?;
        let frame_1 = make_circle_logits(30 + 50, 30 + 32)?;
        // frame_0.save_image("/tmp/frame_0.png")?;
        // frame_1.save_image("/tmp/frame_1.png")?;
        // println!("base; {:?}", frame_0.shape());
        // println!("other; {:?}", frame_1.shape());

        let mut accum = Accumulator::new();

        let layer_count = 4;
        accum.feed_logits_frame(&frame_0.ten()?, layer_count, None)?;
        accum.feed_logits_frame(&frame_1.ten()?, layer_count, None)?;

        let best_fit = accum.frame_relations.get(1).unwrap().best_match().unwrap();
        assert_eq!(best_fit.1, Position::new(-50, -32));

        Ok(())
    }
}
