use flash_powder as fp;
use flash_powder::prelude::*;
use flash_powder::{Device, Ten, Tensor, nn};
use flash_powder_image::prelude::*;
use std::path::Path;
pub mod grid;
use grid::{GridOverlay, Position};
pub mod pyramid;
use pyramid::Pyramid;

use fp::StableTorchResult;

#[derive(Debug, Copy, Clone)]
pub struct AccumulatorConfig {
    pub base_roi_size: Option<(usize, usize)>,
    pub frame_roi_size: (usize, usize),
}
impl Default for AccumulatorConfig {
    fn default() -> Self {
        Self {
            base_roi_size: None,
            frame_roi_size: (512, 512),
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct FrameMatch {
    pub frame_index: usize,
    pub value: f32,
    pub pos: Position,
}

#[derive(Debug, Clone)]
struct FrameRelation {
    pub frame_index: usize,
    pub matches: Vec<FrameMatch>,
}

#[derive(Debug, Clone)]
pub struct Accumulator {
    frames: Vec<Tensor>,
    pyramids: Vec<Pyramid>,
    frame_relations: Vec<FrameRelation>,
    config: AccumulatorConfig,
}
impl Accumulator {
    pub fn new() -> Self {
        Self {
            frames: vec![],
            pyramids: vec![],
            frame_relations: vec![],
            config: AccumulatorConfig::default(),
        }
    }
    pub fn feed_logits_frame(
        &mut self,
        frame: &Ten<'_>,

        debug_dir: Option<&Path>,
    ) -> StableTorchResult<()> {
        dbg!();
        let frame_width = frame.shape()[3];
        let frame_height = frame.shape()[2];
        let new_frame_index = self.frames.len();
        let frame = frame.to(&fp::DType::F32.into())?;

        // Should we do a softmax here? or do we just run with the raw logits?
        let frame = nn::functional::softmax_int(&frame, 1, None)?;

        let multi_res_stack = pyramid::Pyramid::new(frame.i((0, 1, .., ..))?, 3)?;

        let new_frame_prefix = format!("{new_frame_index}");

        multi_res_stack
            .dump_pyramid(debug_dir.as_ref().map(|a| (*a, new_frame_prefix.as_str())))?;

        // Currently, 0 background, 1 foreground.
        // Which means foreground+foreground adds, but background+foreground is not penalized.
        // Does that matter?
        //
        // Currently we pick the new frame center... but that's sub par ideally we'd select the area of the current frame
        // where we have a lot of information.

        // Compare it against all the existing frames, because why not.
        let mut matches = vec![];
        for (i, base_frame) in self.frames.iter().enumerate() {
            dbg!();
            let bf_w = base_frame.shape()[3];
            let bf_h = base_frame.shape()[2];
            // Run the convolution, lets just do it by shape for now.
            // input;  minibatch, in_channels, iH, iW
            // weight; out_channels, in_channels/groups, kh, kw
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
        self.frame_relations.push(FrameRelation {
            frame_index: new_frame_index,
            matches,
        });

        Ok(())
    }

    pub fn debug_use_accumulation(&self, debug_dir: &Path) -> StableTorchResult<()> {
        println!("{:#?}", self.frame_relations);

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

        // Stack the grid, round robin channel.
        let (w, h) = grid.full_size();

        let options = flash_powder::factory::TensorOptions {
            dtype: Some(self.frames[0].dtype()),
            device: Some(self.frames[0].device()),
            ..Default::default()
        };
        let mut canvas: Tensor = Tensor::zeros(&[3, h, w], &options)?;

        let mut channel = 0;
        for (grid_id, frame) in grid_ids.iter().zip(self.frames.iter()) {
            let (ax, ay) = grid.full_grid_irange(*grid_id);
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
}
