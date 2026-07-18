use flash_powder as fp;
use flash_powder::prelude::*;
use flash_powder::{Device, Ten, Tensor, nn};
use flash_powder_image::prelude::*;
use std::path::Path;

use fp::StableTorchResult;

pub struct AccumulatorConfig {
    pub base_roi_size: (usize, usize),
    pub frame_roi_size: (usize, usize),
}
impl Default for AccumulatorConfig {
    fn default() -> Self {
        Self {
            base_roi_size: (768, 512),
            frame_roi_size: (256, 256),
        }
    }
}

struct PositionedFrame {
    pub frame_index: usize,
    pub x: i32,
    pub y: i32,
}

pub struct Accumulator {
    frames: Vec<Tensor>,
    frame_locations: Vec<PositionedFrame>,
    config: AccumulatorConfig,
}
impl Accumulator {
    pub fn new() -> Self {
        Self {
            frames: vec![],
            frame_locations: vec![],
            config: AccumulatorConfig::default(),
        }
    }
    pub fn feed_logits_frame(
        &mut self,
        frame: &Ten<'_>,

        debug_dir: Option<&Path>,
    ) -> StableTorchResult<()> {
        let frame_width = frame.shape()[3];
        let frame_height = frame.shape()[2];
        let new_frame_index = self.frames.len();
        // Should we do a softmax here? or do we just run with the raw logits?
        // Compare it against all the existing frames, because why not.
        let mut best_match: Option<usize> = None;
        for (i, other_frame) in self.frames.iter().enumerate() {
            // Run the convolution, lets just do it by shape for now.
            // input;  minibatch, in_channels, iH, iW
            // weight; out_channels, in_channels/groups, kh, kw

            let base_xrange = ((frame_width / 2) - self.config.base_roi_size.0 / 2) as isize
                ..((frame_width / 2) + self.config.base_roi_size.0 / 2) as isize;
            let base_yrange = ((frame_height / 2) - self.config.base_roi_size.1 / 2) as isize
                ..((frame_height / 2) + self.config.base_roi_size.1 / 2) as isize;

            let frame_xrange = ((frame_width / 2) - self.config.frame_roi_size.0 / 2) as isize
                ..((frame_width / 2) + self.config.frame_roi_size.0 / 2) as isize;
            let frame_yrange = ((frame_height / 2) - self.config.frame_roi_size.1 / 2) as isize
                ..((frame_height / 2) + self.config.frame_roi_size.1 / 2) as isize;

            let other_bg = other_frame
                .i((0, 1..2, base_yrange, base_xrange))?
                .unsqueeze(0)?
                .to(&fp::DType::F32.into())?;

            let new_bg = frame
                .i((0, 1..2, frame_yrange, frame_xrange))?
                .unsqueeze(0)?
                .to(&fp::DType::F32.into())?;
            println!("other shape: {:?}", other_bg.shape());
            println!("new_bg shape: {:?}", new_bg.shape());
            let options = nn::functional::Conv2dOptions {
                // padding: (10, 10),
                ..Default::default()
            };
            println!("running conv");
            let conv2 = nn::functional::conv2d(&other_bg, &new_bg, None, &options)?;
            // Get the peak, check if the peak is higher than current best match.

            if let Some(output_dir) = debug_dir {
                let img_norm = conv2.image_scale_to_domain()?;
                let path = output_dir.join(&format!(
                    "old_frame_{i}_new_frame_{new_frame_index:?}_conv.png"
                ));
                img_norm.save_image(&path)?;
                other_bg
                    .image_scale_to_domain()?
                    .save_image(&output_dir.join(&format!(
                        "old_frame_{i}_new_frame_{new_frame_index:?}_other_bg.png"
                    )))?;
                new_bg
                    .image_scale_to_domain()?
                    .save_image(&output_dir.join(&format!(
                        "old_frame_{i}_new_frame_{new_frame_index:?}_new_bg.png"
                    )))?;
            }

            println!("shape conv2: {:?}", conv2.shape());
            println!("conv2: {:?}", conv2);
        }

        if best_match.is_none() {
            self.frames.push(frame.to_owned()?);
        }

        Ok(())
    }
}
