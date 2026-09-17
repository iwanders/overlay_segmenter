use flash_powder as fp;
use flash_powder::nn;
use flash_powder::prelude::*;
use flash_powder::{StableTorchResult, Ten, Tensor};
use serde::{Deserialize, Serialize};

use crate::accumulator::grid::GridWindow;
use crate::accumulator::grid::Position;

/// Create a distinguishing kernel
///
/// Input is tile_index x channel x height x width
/// It creates a new tensor that for each tile index considers the other tile indices' negative information.
/// stacked_tensor: Stacked labels.
/// channel_weights: Weighting per channel, such that more important distinguishing channels can be raised in weight. Must be equal length to channels.
pub fn make_distinguishing_kernel(
    stacked_tensor: &Ten<'_>,
    channel_weights: &[f32],
) -> StableTorchResult<Tensor> {
    // println!(
    //     "stacked_tensor shape: {:?}, t {:?}",
    //     stacked_tensor.shape(),
    //     stacked_tensor.dtype()
    // );

    let _tile_count = stacked_tensor.size(0);
    let channels = stacked_tensor.size(1);
    let h = stacked_tensor.size(2);
    let w = stacked_tensor.size(3);

    let channel_weighting: Tensor = channel_weights.try_into()?;
    let channel_weighting = channel_weighting.view(&[channels, 1, 1])?;
    let channel_weighting = channel_weighting.to(&stacked_tensor.device().into())?;

    let mut r = Tensor::zeros(
        stacked_tensor.sizes(),
        &flash_powder::factory::TensorOptions {
            dtype: stacked_tensor.dtype().into(),
            device: stacked_tensor.device().into(),
            ..Default::default()
        },
    )?;

    for this_tile in 0..stacked_tensor.size(0) {
        let this_mask = stacked_tensor
            .i((this_tile as isize, .., .., ..))?
            .squeeze()?;
        let mut the_others = Tensor::zeros(
            &[channels, h, w],
            &flash_powder::factory::TensorOptions {
                dtype: stacked_tensor.dtype().into(),
                device: stacked_tensor.device().into(),
                ..Default::default()
            },
        )?;

        for other_tile in 0..stacked_tensor.size(0) {
            if this_tile == other_tile {
                continue;
            }
            the_others.add_assign(&stacked_tensor.i((other_tile as isize, .., .., ..))?)?;
        }

        // next, create a mask for the others.
        let zero: Tensor = 0.0.try_into()?;
        let negative_scaling = -1.0;
        let minus_one: Tensor = negative_scaling.try_into()?;
        let has_value = the_others.ne(&zero)?;
        // Thats a boolean mask... but we don't want it where this mask has values... so we remove that.
        let has_value_but_not_self_values = has_value.mul(&this_mask.eq(&zero)?)?;
        // Next, scale that mask with a negative float.
        let negative_value_accounting = minus_one
            .mul(&has_value_but_not_self_values)?
            .mul(&channel_weighting)?;
        r.i_mut((this_tile as isize, .., .., ..))?
            .add_assign(&negative_value_accounting)?;

        let positive_scaling = 1.0;
        let positive_scalar: Tensor = positive_scaling.try_into()?;
        // Next, we need to overwrite the values that this tile has populated with a positive value.
        let positive_value_accounting = positive_scalar.mul(&this_mask)?.mul(&channel_weighting)?;
        r.i_mut((this_tile as isize, .., .., ..))?
            .add_assign(&positive_value_accounting)?;
    }

    Ok(r)
}

/// Returns the `(value, dx, dy)` of a correlation's peak, where `dx`/`dy` are offsets from
/// the centre of the correlation output.
fn conv_peak(conv: &fp::Ten<'_>) -> anyhow::Result<(f32, isize, isize)> {
    let (values, indices) = conv
        .flatten_using_ints(0, None)?
        .topk(1, &Default::default())?;
    let value = *values.cpu()?.as_f32()?;
    let index = *indices.cpu()?.as_i64()? as isize;
    let width = conv.isize(-1) as isize;
    // let height = conv.isize(-2) as isize;
    let (x, y) = (index % width, index / width);
    Ok((value, x, y))
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct TileScore {
    /// Index of this tile in the tiles for consideration.
    pub index: usize,
    /// Position of where the kernel should be position on the mask for optimal match (so top left corner)
    pub position: Position,
    /// Score at this position.
    pub score: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScoredConv {
    #[serde(with = "crate::serde_tensor::tensor")]
    pub conv2d_values: Tensor,
    pub scores: Vec<TileScore>,
}

impl ScoredConv {
    pub fn highest(&self) -> Option<&TileScore> {
        self.scores
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
    }
}

/// Convolute a mask with a distinguishing kernel.
///
/// Mask: The 'n' channel mask from segmentation.
/// Kernel; The 't' x 'n' x h x w, dinstinguishing kernel with 't' tiles and 'n' channels.
/// Roi; The region of interest (for the top left corner) of the mask. It will _NOT_ position the kernel anywhere outside
///      of the mask.
pub fn conv_mask_with_distinguishing_kernel(
    mask: &Ten<'_>,
    kernel: &Ten<'_>,
    roi: Option<GridWindow>,
) -> StableTorchResult<ScoredConv> {
    let options = nn::functional::Conv2dOptions {
        padding: (0, 0),
        ..Default::default()
    };

    if mask.dim() != 4 {
        anyhow::bail!("mask is not 4 dimensional, expecting b (1) x c x h x w")
    }
    if kernel.dim() != 4 {
        anyhow::bail!("kernel is not 4 dimensional, expecting t (1+) x c x h x w")
    }

    let mask_w = mask.isize(-1) as isize;
    let mask_h = mask.isize(-2) as isize;
    let kernel_w = kernel.isize(-1) as isize;
    let kernel_h = kernel.isize(-2) as isize;
    let (yo, xo, mask) = if let Some(roi) = roi {
        (
            roi.position.y,
            roi.position.x,
            mask.i((
                ..,
                ..,
                (roi.position.y as isize)
                    ..(roi.position.y
                        + (roi.size.h as isize + kernel_h as isize).min(mask_h - roi.position.y)),
                (roi.position.x as isize)
                    ..(roi.position.x
                        + (roi.size.w as isize + kernel_w as isize).min(mask_w - roi.position.x)),
            ))?,
        )
    } else {
        (0, 0, mask.ten()?)
    };
    if false {
        use flash_powder_image::TensorToImage;

        mask.save_image("/tmp/test_conv_tile_matcher_kernel_maskthing.png")?;
    }

    let conv2 = nn::functional::conv2d(&mask, kernel, None, &options)?;
    let conv2 = conv2.to(&fp::DType::F32.into())?;

    let mut scores = vec![];
    for candidate_slice in 0..conv2.size(1) {
        let this_slice = conv2.i((0, candidate_slice as isize, .., ..))?;
        let (this_score, dx, dy) = conv_peak(&this_slice)?;

        scores.push(TileScore {
            index: candidate_slice,
            position: Position {
                x: dx + xo,
                y: dy + yo,
            },
            score: this_score,
        });
    }

    let r = ScoredConv {
        conv2d_values: conv2,
        scores,
    };
    Ok(r)
}

#[cfg(test)]
mod test {
    use flash_powder_image::TensorToImage as _;

    use crate::accumulator::pyramid::circle_image;

    use super::*;

    #[test]
    fn test_conv_tile_matcher() -> StableTorchResult<()> {
        // Lets make the kernel with just a filled circle.
        let kernel = circle_image(8, 8, 4, 4, 4)?;
        // Lets make an image, with two circles, one at the top right that has a 1.0 kernel, and one at the bottom
        // left with 0.5 kernel.
        let mut mask = Tensor::zeros(&[32, 32], &Default::default())?;
        *mask.f32_mut(&[0, 0])? = 1.0;

        mask.i_mut((8isize..16, 16isize..24))?.add_assign(&kernel)?;

        let f32_0_5: Tensor = 0.5.try_into()?;
        mask.i_mut((24isize..32, 4isize..12))?
            .add_assign(&kernel.mul(&f32_0_5)?)?;

        // Write to disk for inspection.
        mask.save_image("/tmp/test_conv_tile_matcher_mask.png")?;
        kernel.save_image("/tmp/test_conv_tile_matcher_kernel.png")?;

        // It needs to have channels and such, but they can be 1 size.
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
        let kernel = kernel.unsqueeze(0)?.unsqueeze(0)?;

        dbg!();
        // Calculate the convolution and best fit for each.
        let r = conv_mask_with_distinguishing_kernel(&mask.ten()?, &kernel.ten()?, None)?;
        let highest = r.highest();
        assert!(highest.is_some());
        let highest = highest.unwrap();
        assert_eq!(highest.position.x, 16);
        assert_eq!(highest.position.y, 8);
        assert_eq!(highest.score, 47.0);
        assert_eq!(highest.index, 0);
        dbg!();

        // Next, search in the bottom left window for the weaker signal...
        let roi = GridWindow::rect_at((16, 16).into(), (0, 16).into());
        let r = conv_mask_with_distinguishing_kernel(&mask.ten()?, &kernel.ten()?, Some(roi))?;
        println!("r: {r:#?}");
        let highest = r.highest();
        assert!(highest.is_some());
        let highest = highest.unwrap();
        assert_eq!(highest.position.x, 4);
        assert_eq!(highest.position.y, 24);
        assert_eq!(highest.score, 47.0 * 0.5);
        assert_eq!(highest.index, 0);

        dbg!();
        // WHat is we have an ROI that is smaller than the actual image... we need to grow the view into the mask then
        // because having an ROI that's smaller than the kernel is very well possible if we already have a precise estimate.
        let roi = GridWindow::rect_at((4, 4).into(), (2, 22).into());
        let r = conv_mask_with_distinguishing_kernel(&mask.ten()?, &kernel.ten()?, Some(roi))?;
        println!("r: {r:#?}");
        let highest = r.highest();
        assert!(highest.is_some());
        let highest = highest.unwrap();
        assert_eq!(highest.position.x, 4);
        assert_eq!(highest.position.y, 24);
        assert_eq!(highest.score, 47.0 * 0.5);
        assert_eq!(highest.index, 0);

        Ok(())
    }
}
