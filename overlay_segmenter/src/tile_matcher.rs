use flash_powder::prelude::*;
use flash_powder::{StableTorchResult, Ten, Tensor};

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
