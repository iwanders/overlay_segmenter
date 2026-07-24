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

    pub fn debug_use_accumulation(&self, debug_dir: &Path) -> StableTorchResult<()> {
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

    pub fn write_file<Q: AsRef<Path>>(&mut self, output_path: Q) -> StableTorchResult<()> {
        // self.pyramids.clear();
        // self.frame_relations.clear();
        // self.frames.clear();
        let data = postcard::to_allocvec(self)?;
        println!("dta is {:?}", data.len());
        std::fs::write(output_path, &data).map_err(|e| anyhow::format_err!(e))
    }
}
