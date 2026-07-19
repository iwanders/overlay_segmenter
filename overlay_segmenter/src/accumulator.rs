use flash_powder as fp;
use flash_powder::prelude::*;
use flash_powder::{Device, Ten, Tensor, nn};
use flash_powder_image::prelude::*;
use std::path::Path;

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
    pub x: isize,
    pub y: isize,
}

#[derive(Debug, Clone)]
struct FrameRelation {
    pub frame_index: usize,
    pub matches: Vec<FrameMatch>,
}

#[derive(Debug, Clone)]
pub struct Accumulator {
    frames: Vec<Tensor>,
    frame_relations: Vec<FrameRelation>,
    config: AccumulatorConfig,
}
impl Accumulator {
    pub fn new() -> Self {
        Self {
            frames: vec![],
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

            let (base_xrange, base_yrange) =
                if let Some(base_roi_size) = self.config.base_roi_size.as_ref() {
                    let base_xrange = ((bf_w / 2) - base_roi_size.0 / 2) as isize
                        ..((bf_w / 2) + base_roi_size.0 / 2) as isize;
                    let base_yrange = ((bf_h / 2) - base_roi_size.1 / 2) as isize
                        ..((bf_h / 2) + base_roi_size.1 / 2) as isize;
                    (base_xrange, base_yrange)
                } else {
                    (0..bf_w as isize, 0..bf_h as isize)
                };
            dbg!(&base_xrange);
            dbg!(&base_yrange);

            let frame_xrange = ((frame_width / 2) - self.config.frame_roi_size.0 / 2) as isize
                ..((frame_width / 2) + self.config.frame_roi_size.0 / 2) as isize;
            let frame_yrange = ((frame_height / 2) - self.config.frame_roi_size.1 / 2) as isize
                ..((frame_height / 2) + self.config.frame_roi_size.1 / 2) as isize;

            let base_bg = base_frame
                .i((0, 1..2, base_yrange.clone(), base_xrange.clone()))?
                .unsqueeze(0)?;

            let base_bg = base_bg.image_resize(
                [base_yrange.len() / 2, base_xrange.len() / 2],
                nn::functional::InterpolateAlgorithm::Bilinear,
            )?;

            let new_bg = frame
                .i((0, 1..2, frame_yrange, frame_xrange))?
                .unsqueeze(0)?
                .to(&fp::DType::F32.into())?;

            let new_bg = new_bg.image_resize(
                [
                    self.config.frame_roi_size.1 / 2,
                    self.config.frame_roi_size.0 / 2,
                ],
                nn::functional::InterpolateAlgorithm::Bilinear,
            )?;

            println!("base shape: {:?}", base_bg.shape());
            println!("new_bg shape: {:?}", new_bg.shape());
            let options = nn::functional::Conv2dOptions {
                // padding: (10, 10),
                ..Default::default()
            };
            println!("running conv");
            let conv2 = nn::functional::conv2d(&base_bg, &new_bg, None, &options)?;
            // Get the peak, check if the peak is higher than current best match.

            if let Some(output_dir) = debug_dir {
                let path = output_dir.join(&format!("new_frame_{new_frame_index:?}_orig.png"));
                frame.i((0, 1, .., ..))?.save_image(&path)?;

                let img_norm = conv2.image_scale_to_domain()?;
                let path = output_dir.join(&format!(
                    "new_frame_{new_frame_index:?}_old_frame_{i}_conv.png"
                ));
                img_norm.save_image(&path)?;
                base_bg
                    .image_scale_to_domain()?
                    .save_image(&output_dir.join(&format!(
                        "new_frame_{new_frame_index:?}_old_frame_{i}_other_bg.png"
                    )))?;
                new_bg
                    .image_scale_to_domain()?
                    .save_image(&output_dir.join(&format!(
                        "new_frame_{new_frame_index:?}_old_frame_{i}_new_bg.png"
                    )))?;
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
            //
            // But that's x at the ROI... and we want X for the full frame.
            //
            //
            let x = x as isize - self.config.frame_roi_size.0 as isize - base_xrange.start;
            let y = y as isize - self.config.frame_roi_size.1 as isize - base_yrange.start;

            println!("shape conv2: {:?}", conv2.shape());
            //println!("conv2: {:?}", conv2);
            println!("values: {:?} {:?} {value}", values.shape(), values);
            println!("indices: {:?} peak at {x},{y}", indices);
            matches.push(FrameMatch {
                frame_index: i,
                value: value,
                x,
                y,
            });
        }

        self.frames.push(frame);
        self.frame_relations.push(FrameRelation {
            frame_index: new_frame_index,
            matches,
        });

        Ok(())
    }

    pub fn debug_use_accumulation(&self, debug_dir: &Path) -> StableTorchResult<()> {
        println!("{:#?}", self.frame_relations);
        // Lets just iterate over it all and add the pairs.
        for base in self.frame_relations.iter() {
            let base_frame = &self.frames[base.frame_index];
            let t: Tensor = 0.5.try_into()?;
            let base_bool = base_frame.ge(&t)?;
            for other in base.matches.iter() {
                let other_frame = &self.frames[other.frame_index];
                let other_bool = other_frame.ge(&t)?;

                // Lets combine these...
                // We need to make a canvas using the offset... and size of both frames.
            }
        }
        Ok(())
    }
}
