use flash_powder as fp;
use flash_powder::nn;
use flash_powder::prelude::*;
use flash_powder::{StableTorchResult, Ten, Tensor};
use serde::{Deserialize, Serialize};

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
    index: usize,
    /// Position of where the kernel should be position on the mask for optimal match (so top left corner)
    position: Position,
    /// Score at this position.
    score: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScoredConv {
    #[serde(with = "crate::serde_tensor::tensor")]
    pub conv2d_values: Tensor,
    pub scores: Vec<TileScore>,
}
pub fn conv_mask_with_distinguishing_kernel(
    mask: &Ten<'_>,
    mask_scale: isize,
    kernel: &Ten<'_>,
) -> StableTorchResult<ScoredConv> {
    let options = nn::functional::Conv2dOptions {
        padding: (0, 0),
        ..Default::default()
    };
    let conv2 = nn::functional::conv2d(mask, kernel, None, &options)?;
    let conv2 = conv2.to(&fp::DType::F32.into())?;
    println!("conv2.shape: {:?}", conv2.shape());

    let mask_center_w = mask.isize(-1) / 2;
    let mask_center_h = mask.isize(-2) / 2;
    let kernel_center_w = kernel.isize(-1) / 2;
    let kernel_center_h = kernel.isize(-2) / 2;

    let mut scores = vec![];
    for candidate_slice in 0..conv2.size(0) {
        let this_slice = conv2.i((candidate_slice as isize, .., ..))?;
        let (this_score, dx, dy) = conv_peak(&this_slice)?;
        let dx = dx;
        let dy = dy;

        scores.push(TileScore {
            index: candidate_slice,
            position: Position { x: dx, y: dy },
            score: this_score,
        });
    }

    let r = ScoredConv {
        conv2d_values: conv2,
        scores,
    };
    Ok(r)
}
