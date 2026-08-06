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


Notes from running live:
    - When we lose frames, the window may disconnect from the global estimate, probably want to match against the global afterall?
    - The area_radius value tanks performance.
    - Area radius trick also doesn't work to clean up detections that happened once, since we don't actually see anything around there so we
      end up not clearing it. Should we just do a circle around the center of the screen? But still fit with everything?


    - should try; updated_map = (new_frame - map) * update_rate + map
*/

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
struct FrameMatch {
    pub frame_index: StableFrameId,
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
pub struct BufferedMergeConfig {
    pub min_observations: usize,
    pub area_radius: usize,
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub struct RatioUpdateMergeConfig {
    pub update_rate: f64,
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub enum MergeMode {
    // All logits are softmax'd when received by the Accumulator.
    // Position correlation is only done on the first channel (normal walls)
    /// Update only in a buffered local area.
    ///
    /// For each sliding window processing:
    ///  - Allocate processing_values (f32) and processing_counts (i64) of appropriate size
    ///  - For frames in the sliding window:
    ///     - Determine local vicinity boolean mask for this frame by convoluting with circle of area_radius.
    ///     - Add values in local vicinity to processing_values at appropriate position.
    ///     - Increment processing_counts by one for the local vicinity.
    ///  - Merge processing_values and processing_counts into global accumulated values at correct position.
    ///
    /// Extract final result by:
    ///  - Create boolean mask where global counts not zero.
    ///  - Divide global values in the mask by counts in the mask.
    ///
    /// Notes:
    ///  - Effective for walls segmented just one or two frames like near an exit.
    ///  - Flipside of that is that it can't clear up false positives because they create a local vicinity once, which
    ///    leads to values being populated, but because there's never a wall in the local vicinity again it never clears
    ///    false positives.
    ///  - Very effective against obscurations like inventory panel.
    ///
    /// Todo: maybe pair this with an area that's like; "only accept within", possibly based on the map-view-distance?
    Buffered(BufferedMergeConfig),

    /// Combine each frame using a update rate,
    /// updated_map = (new_frame - map) * update_rate + map
    ///
    /// Can recover from false positive sections.
    ///
    /// Notes:
    ///   - Very clean results, good false positive rejection.
    ///   - Does not handle obscurations well, because it updates everything in view, so inventory panel is problematic.
    ///   - Introduces lag in information usage, resulting in map parts only seen a short while near the exit missing.
    RatioUpdate(RatioUpdateMergeConfig),
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub struct AccumulationConfig {
    pub fit_against_previous_frames: usize,
    pub layer_count: usize,
    pub merge_mode: MergeMode,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Accumulation {
    #[serde(with = "crate::serde_tensor::tensor")]
    accumulation_values: Tensor,
    #[serde(with = "crate::serde_tensor::tensor")]
    accumulation_counts: Tensor,

    accumulation_position: Position,

    config: AccumulationConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LocalizedFrame {
    id: StableFrameId,
    #[serde(with = "crate::serde_tensor::tensor")]
    frame: Tensor,
    pyramid: Pyramid,
    frame_relation: FrameRelation,
    /// This is the accumulated global position, determined against the best matching of the frame relations.
    global_pos: Position,
}
impl LocalizedFrame {
    pub fn into_device(self, device: &fp::Device) -> StableTorchResult<Self> {
        Ok(Self {
            id: self.id,
            frame: self.frame.to(&flash_powder::factory::ToOptions {
                device: Some(*device),
                ..Default::default()
            })?,
            pyramid: self.pyramid.into_device(*device)?,
            frame_relation: self.frame_relation,
            global_pos: self.global_pos,
        })
    }
}

use std::collections::BTreeMap;

#[derive(Debug, Copy, Clone, Deserialize, Serialize, Ord, PartialOrd, Eq, PartialEq)]
pub struct StableFrameId(pub usize);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Accumulator {
    localized_frames: BTreeMap<StableFrameId, LocalizedFrame>,
    id_counter: usize,
    accumulation: Option<Accumulation>,
}
impl Accumulator {
    pub fn new() -> Self {
        Self {
            localized_frames: Default::default(),
            accumulation: None,
            id_counter: 0,
        }
    }
    pub fn enable_accumulator(&mut self, config: AccumulationConfig) -> StableTorchResult<()> {
        self.accumulation = Some(Accumulation {
            accumulation_values: Tensor::zeros(&[], &Default::default())?,
            accumulation_counts: Tensor::zeros(&[], &Default::default())?,
            accumulation_position: Position::origin(),

            config,
        });
        Ok(())
    }

    pub fn accumulate_merge_buffered(
        &mut self,
        frames: &[(Position, Tensor)],
        config: &BufferedMergeConfig,
    ) -> StableTorchResult<()> {
        let (values, counts, bottom_left_corner) =
            Self::combine_frames(frames, config.area_radius)?;
        // First see if this is hte first frame.
        let accumulation_mut = self.accumulation.as_mut().unwrap();
        if accumulation_mut.accumulation_counts.dim() == 0 {
            accumulation_mut.accumulation_counts = counts;
            accumulation_mut.accumulation_values = values;
            accumulation_mut.accumulation_position = bottom_left_corner.into();
        } else {
            // Create a grid for the accumulation so far.
            let mut grid = GridOverlay::new();
            let orig_id = grid.add_tensor(
                &accumulation_mut.accumulation_counts,
                accumulation_mut.accumulation_position,
            );

            let start_extent = grid.extent();

            // Next, add the new tensors.
            let new_id = grid.add_tensor(&values, bottom_left_corner);

            if start_extent != grid.extent() {
                println!(
                    "accumulation_mut.accumulation_counts shape: {:?}",
                    accumulation_mut.accumulation_counts.shape()
                );
                // Allocate new tensors, then copy the data.
                // I should add that 'derivate of' new zero.
                let (w, h) = grid.full_size();
                let options = flash_powder::factory::TensorOptions {
                    dtype: Some(accumulation_mut.accumulation_counts.dtype()),
                    device: Some(accumulation_mut.accumulation_counts.device()),
                    ..Default::default()
                };
                let mut new_counts: Tensor = Tensor::zeros(&[1, h, w], &options)?;
                let options = flash_powder::factory::TensorOptions {
                    dtype: Some(accumulation_mut.accumulation_values.dtype()),
                    device: Some(accumulation_mut.accumulation_values.device()),
                    ..Default::default()
                };
                let c = accumulation_mut.accumulation_values.isize(-3);
                let mut new_values: Tensor = Tensor::zeros(&[c, h, w], &options)?;

                // Next, do the blitting.
                let (ax, ay) = grid.full_grid_irange(orig_id);
                new_values
                    .i_mut((.., ay, ax))?
                    .copy_from_tensor(&accumulation_mut.accumulation_values)?;
                let (ax, ay) = grid.full_grid_irange(orig_id);
                new_counts
                    .i_mut((.., ay, ax))?
                    .copy_from_tensor(&accumulation_mut.accumulation_counts)?;
                accumulation_mut.accumulation_counts = new_counts;
                accumulation_mut.accumulation_values = new_values;

                // We need to correct the position when this happens.
                let grid_pos = grid.full_position();
                accumulation_mut.accumulation_position = grid_pos.into();
            }

            // Now the grid is always the correct size, and we can do the addition thing.

            let (ax, ay) = grid.full_grid_irange(new_id);

            accumulation_mut
                .accumulation_values
                .i_mut((.., ay.clone(), ax.clone()))?
                .add_assign(&values.squeeze()?)?;

            // And repeat for counts;
            let (ax, ay) = grid.full_grid_irange(new_id);

            accumulation_mut
                .accumulation_counts
                .i_mut((.., ay.clone(), ax.clone()))?
                .add_assign(&counts.squeeze()?)?;
        }

        Ok(())
    }

    pub fn accumulate_ratio_update(
        &mut self,
        frames: &[(Position, Tensor)],
        config: &RatioUpdateMergeConfig,
    ) -> StableTorchResult<()> {
        let mut grid = GridOverlay::new();
        let (position, values) = frames
            .first()
            .ok_or(anyhow::format_err!("no frames provided"))?;
        grid.add_tensor(values, *position);

        let first_frame = &frames.first().unwrap().1;
        // let channels = first_frame.isize(-3);

        // let fdtype = first_frame.dtype();
        let values = values.squeeze()?;

        // let options = flash_powder::factory::TensorOptions {
        //     dtype: Some(fdtype),
        //     device: Some(first_frame.device()),
        //     ..Default::default()
        // };
        let (w, h) = grid.full_size();
        let counts: Tensor = Tensor::ones(
            &[1, h, w],
            &flash_powder::factory::TensorOptions {
                dtype: Some(fp::DType::I64),
                device: Some(first_frame.device()),
                ..Default::default()
            },
        )?;
        let bottom_left_corner = grid.full_position();
        // First see if this is hte first frame.
        let accumulation_mut = self.accumulation.as_mut().unwrap();
        if accumulation_mut.accumulation_counts.dim() == 0 {
            accumulation_mut.accumulation_counts = counts;
            accumulation_mut.accumulation_values = values.lazy_clone()?;
            accumulation_mut.accumulation_position = bottom_left_corner.into();
        } else {
            // Create a grid for the accumulation so far.
            let mut grid = GridOverlay::new();
            let orig_id = grid.add_tensor(
                &accumulation_mut.accumulation_counts,
                accumulation_mut.accumulation_position,
            );

            let start_extent = grid.extent();

            // Next, add the new tensors.
            let new_id = grid.add_tensor(&values, bottom_left_corner);

            if start_extent != grid.extent() {
                println!(
                    "accumulation_mut.accumulation_counts shape: {:?}",
                    accumulation_mut.accumulation_counts.shape()
                );
                // Allocate new tensors, then copy the data.
                // I should add that 'derivate of' new zero.
                let (w, h) = grid.full_size();
                let options = flash_powder::factory::TensorOptions {
                    dtype: Some(accumulation_mut.accumulation_counts.dtype()),
                    device: Some(accumulation_mut.accumulation_counts.device()),
                    ..Default::default()
                };
                let mut new_counts: Tensor = Tensor::zeros(&[1, h, w], &options)?;
                let options = flash_powder::factory::TensorOptions {
                    dtype: Some(accumulation_mut.accumulation_values.dtype()),
                    device: Some(accumulation_mut.accumulation_values.device()),
                    ..Default::default()
                };
                let c = accumulation_mut.accumulation_values.isize(-3);
                let mut new_values: Tensor = Tensor::zeros(&[c, h, w], &options)?;

                // Next, do the blitting.
                let (ax, ay) = grid.full_grid_irange(orig_id);
                new_values
                    .i_mut((.., ay, ax))?
                    .copy_from_tensor(&accumulation_mut.accumulation_values)?;
                let (ax, ay) = grid.full_grid_irange(orig_id);
                new_counts
                    .i_mut((.., ay, ax))?
                    .copy_from_tensor(&accumulation_mut.accumulation_counts)?;
                accumulation_mut.accumulation_counts = new_counts;
                accumulation_mut.accumulation_values = new_values;

                // We need to correct the position when this happens.
                let grid_pos = grid.full_position();
                accumulation_mut.accumulation_position = grid_pos.into();
            }

            // Now the grid is always the correct size, and we can do the addition thing.

            let (ax, ay) = grid.full_grid_irange(new_id);

            //  updated_map = (new_frame - map) * update_rate + map
            let new_frame = values.squeeze()?;
            let old_map = accumulation_mut
                .accumulation_values
                .i((.., ay.clone(), ax.clone()))?
                .to_owned()?;
            let scalar: Tensor = config.update_rate.try_into()?;
            let combined = (new_frame.sub(&old_map)?).mul(&scalar.to(&old_map.device().into())?)?;
            accumulation_mut
                .accumulation_values
                .i_mut((.., ay.clone(), ax.clone()))?
                .add_assign(&combined)?;

            // And repeat for counts;
            let (ax, ay) = grid.full_grid_irange(new_id);

            accumulation_mut
                .accumulation_counts
                .i_mut((.., ay.clone(), ax.clone()))?
                .add_assign(&counts.squeeze()?)?;
        }
        Ok(())
    }

    pub fn accumulate_logits_frame(&mut self, frame: &Ten<'_>) -> StableTorchResult<()> {
        if self.accumulation.is_none() {
            anyhow::bail!("trying to accumulate without config");
        }

        let config = self.accumulation.as_ref().map(|a| a.config).unwrap();

        self.feed_logits_frame(frame, config.layer_count, None)?;

        if self.localized_frames.len() >= config.fit_against_previous_frames {
            // Stack 'n' frames together, filter them, and merge them into the main accumulation.
            let mut frames = vec![];
            for localized_frame in self
                .localized_frames
                .values()
                .take(config.fit_against_previous_frames)
            {
                frames.push((
                    localized_frame.global_pos,
                    localized_frame.frame.lazy_clone()?,
                ));
            }

            match config.merge_mode {
                MergeMode::Buffered(buffered_merge_config) => {
                    self.accumulate_merge_buffered(&frames, &buffered_merge_config)?
                }
                MergeMode::RatioUpdate(ratio_update_merge_config) => {
                    self.accumulate_ratio_update(&frames, &ratio_update_merge_config)?
                }
            }

            // Pop the frame from the front.
            let _ = self.localized_frames.pop_first().unwrap();
        }

        Ok(())
    }

    fn accumulate_buffered(
        &self,
        accumulate: &Accumulation,
        config: &BufferedMergeConfig,
        debug_dir: Option<&Path>,
    ) -> StableTorchResult<Tensor> {
        // Now that we are here, we can divide the actual values by the counts... and we should get a pristine image.
        // Acounts contains zeros, which is a problem as it blows up the values to nan, so on the locations where the
        // count is actually zero, we just add one, this shouldn't matter as the values there should be zero.
        let zero: Tensor = 0i64.try_into()?;
        let positions_with_zero = accumulate
            .accumulation_counts
            .eq(&zero)?
            .squeeze()?
            .to_owned()?;
        let zeros_made_into_ones_but_counts_else =
            accumulate.accumulation_counts.add(&positions_with_zero)?;

        if let Some(debug_dir) = debug_dir {
            zeros_made_into_ones_but_counts_else
                .image_scale_to_domain()?
                .save_image(debug_dir.join(format!("zeros_made_into_ones_but_counts_else.png")))?;
        }

        let r = accumulate.accumulation_values.div(
            &zeros_made_into_ones_but_counts_else
                .to(&accumulate.accumulation_values.dtype().into())?,
        )?;

        // We can do this to further clean it up, accepting only data where 'n' observations were present.
        let two: Tensor = (config.min_observations as i64).try_into()?;
        let positions_with_twos = accumulate
            .accumulation_counts
            .ge(&two)?
            .squeeze()?
            .to_owned()?;
        let r = r.mul(&positions_with_twos)?;
        return Ok(r);
    }
    pub fn accumulate_postprocess(&self, debug_dir: Option<&Path>) -> StableTorchResult<Tensor> {
        if self.accumulation.is_none() {
            anyhow::bail!("trying to accumulate without config");
        }
        let config = self.accumulation.as_ref().map(|a| a.config).unwrap();
        let accumulate = self.accumulation.as_ref().unwrap();

        match config.merge_mode {
            MergeMode::Buffered(buffered_merge_config) => {
                self.accumulate_buffered(&accumulate, &buffered_merge_config, debug_dir)
            }
            MergeMode::RatioUpdate(_ratio_update_merge_config) => {
                accumulate.accumulation_values.lazy_clone()
            }
        }
    }

    pub fn feed_logits_frame(
        &mut self,
        frame: &Ten<'_>,
        layer_count: usize,
        debug_dir: Option<&Path>,
    ) -> StableTorchResult<()> {
        let new_frame_index = self.id_counter;
        self.id_counter += 1;
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
        for (frame_id, base_frame) in self.localized_frames.iter() {
            let other_pyramid = &base_frame.pyramid;

            let this_frame_prefix = format!("{}", frame_id.0);
            let (value, pos) = multi_res_stack.pyramid_aligner(
                &other_pyramid,
                debug_dir.as_ref().map(|a| (*a, this_frame_prefix.as_str())),
            )?;
            println!("Frame {this_frame_prefix} with new: {value:?}, at {pos:?}");
            // Record the alignment of this new frame against the existing frame `i`.
            matches.push(FrameMatch {
                frame_index: base_frame.id,
                value,
                pos,
            });
        }
        let id = StableFrameId(new_frame_index);
        let frame_relation = FrameRelation { matches };
        let best_match = frame_relation.best_match();
        let global_pos = if let Some((_score, _pos, _framematch)) = best_match {
            let best_frame = self.localized_frames.get(&_framematch.frame_index).unwrap();
            best_frame.global_pos + _pos
        } else {
            Position::origin()
        };
        println!("Inserting frame with {id:?} at {global_pos:?}");
        self.localized_frames.insert(
            id,
            LocalizedFrame {
                id,
                frame,
                pyramid: multi_res_stack,
                frame_relation,
                global_pos,
            },
        );

        Ok(())
    }

    fn create_grid(&self) -> StableTorchResult<GridOverlay> {
        // Iterate over all the frames, collect the best scoring one, then create the composite.
        let mut grid = GridOverlay::new();
        let mut grid_ids = BTreeMap::new();
        for (i, base) in self.localized_frames.iter() {
            println!("frame {i:?}");

            let best_entry: Option<(f32, Position, FrameMatch)> = base.frame_relation.best_match();

            if let Some((score, p, frame_match)) = best_entry {
                println!(" score: {score}");
                println!(" pos  : {p:?}");
                println!(" to   : {:?}", frame_match.frame_index);

                let parent_pos = grid.grid_position(grid_ids[&frame_match.frame_index]);
                println!(
                    " parent_pos   : {:?}    (parent_pos + frame_match.pos): {:?}",
                    parent_pos,
                    (parent_pos + frame_match.pos)
                );

                grid_ids.insert(
                    *i,
                    grid.add_tensor(&base.frame.ten()?, (parent_pos + frame_match.pos).into()),
                );
            } else {
                grid_ids.insert(*i, grid.add_tensor(&base.frame.ten()?, Position::origin()));
            }
        }

        Ok(grid)
    }

    pub fn debug_use_accumulation(&self, debug_dir: &Path) -> StableTorchResult<()> {
        let grid = self.create_grid()?;
        // Stack the grid, round robin channel.
        let (w, h) = grid.full_size();

        let options = flash_powder::factory::TensorOptions {
            dtype: Some(
                self.localized_frames
                    .first_key_value()
                    .map(|(_, a)| a.frame.dtype())
                    .unwrap(),
            ),
            device: Some(
                self.localized_frames
                    .first_key_value()
                    .map(|(_, a)| a.frame.device())
                    .unwrap(),
            ),
            ..Default::default()
        };
        let mut canvas: Tensor = Tensor::zeros(&[3, h, w], &options)?;

        let mut channel = 0;
        for (grid_id, (_stable_id, localized_frame)) in grid.ids().zip(self.localized_frames.iter())
        {
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
    ) -> StableTorchResult<(Tensor, Tensor, Position)> {
        let mut grid = GridOverlay::new();
        for (position, t) in frames.iter() {
            grid.add_tensor(t, *position);
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
            let presence_mask_with_ones = inflated_non_zero_present(frame.ten()?, area_radius)?;
            counts
                .i_mut((.., ay.clone(), ax.clone()))?
                .add_assign(&presence_mask_with_ones)?;

            // Next, we can add the values, but we also multiply those by the mask we just created.
            // Such that we only consider points in the vicinity.

            values
                .i_mut((.., ay.clone(), ax.clone()))?
                .add_assign(&frame.squeeze()?.mul(&presence_mask_with_ones)?)?;
        }

        Ok((values, counts, grid.full_position().into()))
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
        for localized_frame in self.localized_frames.values() {
            let offset = localized_frame
                .frame_relation
                .best_match()
                .map(|(_, p, _)| p)
                .unwrap_or(Position::origin());
            current_pos = current_pos + offset;
            frames.push((current_pos, localized_frame.frame.lazy_clone()?));
        }

        let (values, counts, _position) = Self::combine_frames(&frames, area_radius)?;

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

    pub fn into_device(self, device: fp::Device) -> StableTorchResult<Self> {
        let mut localized_frames = BTreeMap::new();
        for (id, p) in self.localized_frames.into_iter() {
            localized_frames.insert(id, p.into_device(&device)?);
        }

        Ok(Self {
            id_counter: self.id_counter,
            localized_frames,
            accumulation: self.accumulation,
        })
    }
    pub fn frame_count(&self) -> usize {
        self.localized_frames.len()
    }

    pub fn pop_left(&mut self) -> Option<(Tensor, FrameRelation, Pyramid)> {
        if self.frame_count() > 1 {
            let (_, r) = self.localized_frames.pop_first()?;
            let LocalizedFrame {
                id: _,
                frame,
                pyramid,
                frame_relation,
                global_pos: _,
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
        let other = circle_image(h, w, cx, cy, r)?;

        let zero: Tensor = 0.0f32.try_into()?;
        let one: Tensor = 3.0f32.try_into()?;
        let base = other.eq(&zero)?.mul(&one)?;

        let half: Tensor = 0.01f32.try_into()?;
        let black = other.mul(&half)?;

        fp::torch::stack(&[base.ten()?, other.ten()?, black.ten()?], 0)?
            .unsqueeze(0)?
            .to_owned()
    }

    #[test]
    fn test_accumulator() -> StableTorchResult<()> {
        let frame_0 = make_circle_logits(30, 30)?;
        let frame_1 = make_circle_logits(30 + 50, 30 + 32)?;

        let mut accum = Accumulator::new();

        let layer_count = 4;
        accum.feed_logits_frame(&frame_0.ten()?, layer_count, None)?;
        accum.feed_logits_frame(&frame_1.ten()?, layer_count, None)?;

        let best_fit = accum
            .localized_frames
            .last_key_value()
            .unwrap()
            .1
            .frame_relation
            .best_match()
            .unwrap();
        assert_eq!(best_fit.1, Position::new(-50, -32));

        Ok(())
    }

    #[test]
    fn test_accumulator_online() -> StableTorchResult<()> {
        let frame_0 = make_circle_logits(30, 30)?;
        let frame_1 = make_circle_logits(30 + 10, 30 + 10)?;
        let frame_2 = make_circle_logits(30 + 20, 30 + 20)?;
        let frame_3 = make_circle_logits(30 + 30, 30 + 30)?;
        let frame_4 = make_circle_logits(30 + 40, 30 + 40)?;

        let mut accumulator = Accumulator::new();
        let config = AccumulationConfig {
            fit_against_previous_frames: 3,
            layer_count: 3,
            merge_mode: MergeMode::Buffered(BufferedMergeConfig {
                min_observations: 0,
                area_radius: 15,
            }),
        };
        accumulator.enable_accumulator(config)?;

        accumulator.accumulate_logits_frame(&frame_0.ten()?)?;
        accumulator.accumulate_logits_frame(&frame_1.ten()?)?;
        accumulator.accumulate_logits_frame(&frame_2.ten()?)?;
        accumulator.accumulate_logits_frame(&frame_3.ten()?)?;
        accumulator.accumulate_logits_frame(&frame_4.ten()?)?;

        let r = accumulator.accumulate_postprocess(None)?;

        let zero: Tensor = 0.0f32.try_into()?;
        let r_nonzero = r.i((2, 1..128, 0..128))?.ne(&zero)?;
        let z: usize = r_nonzero.bools_ref()?.iter().map(|b| *b as usize).sum();

        // Hardcoded value from visual inspection, if the spheres are not aligned this number is incorrect.
        assert_eq!(z, 2969);

        Ok(())
    }
}
