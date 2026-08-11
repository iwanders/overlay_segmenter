use flash_powder::Tensor;
use flash_powder::prelude::*;
// -------------- hsv_to_rgb --------------
// https://github.com/python/cpython/blob/0fff6bd86cf0224152c509e295d3cbbd209098f3/Lib/colorsys.py#L145

// Converted by qwen3.5:9b

/// Converts HSV color values to RGB using float representation.
///
/// # Arguments
/// * `h`: Hue (H) in range [0.0, 1.0)
/// * `s`: Saturation (S) in range [0.0, 1.0]
/// * `v`: Value (V) in range [0.0, 1.0]
///
/// # Returns
/// A tuple containing (R, G, B) values in the same range as V.
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    // Handle grayscale (zero saturation)
    if s == 0.0 {
        return (v, v, v);
    }

    // Calculate the sector index i
    // Python's int() truncates; Rust's (val as u32) does the same for positive values
    let i = (h * 6.0) as u32;

    // Calculate the fractional part f
    // Note: i is cast back to f32 for the subtraction to preserve float precision
    let f = (h * 6.0) - (i as f32);

    // Calculate intermediate values
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    // Wrap i to be within [0, 5]
    let i = i % 6;

    // Return the appropriate channel combination based on the sector
    match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        5 => (v, p, q),
        _ => unreachable!(), // Should never happen as i % 6 is 0..=5
    }
}

pub fn generate_color_palette(class_count: usize) -> Result<Tensor, anyhow::Error> {
    let mut colors = vec![[0.0, 0.0, 0.0]];
    let class_count = class_count - 1; // -1 because black was already added
    for i in 0..class_count {
        // 1. Divide hue space evenly between 0.0 and 1.0
        let hue = (i as f32) / (class_count as f32);

        // 2. Keep saturation and value high for vibrant, distinct colors
        let saturation = 1.0;
        let value = 1.0;

        colors.push(hsv_to_rgb(hue, saturation, value).into());
    }
    Tensor::from(&colors[..])
}

pub fn apply_pallette(
    palette: &flash_powder::Ten<'_>,
    n_channel_tensor: &flash_powder::Ten<'_>,
) -> Result<Tensor, anyhow::Error> {
    let pixel_index = n_channel_tensor.argmax(Some(0), Some(true))?;
    palette.index_tensor(&[pixel_index])?.squeeze()?.to_owned()
}
pub fn unapply_pallette(
    class_count: usize,
    n_channel_tensor: &flash_powder::Ten<'_>,
) -> Result<Tensor, anyhow::Error> {
    // Lets allocate the tensor
    let mut d = Tensor::zeros(
        &[
            class_count,
            n_channel_tensor.size(0),
            n_channel_tensor.size(1),
        ],
        &n_channel_tensor.device().into(),
    )?;
    // Next, generate the palette.
    let palette = generate_color_palette(class_count)?;
    // Next, sweep over the entries in the pallette, obtain a boolean mask, and set the values in d to one.
    for i in 0..class_count {
        let this_color = palette.i((i as isize, ..))?;
        let mask = n_channel_tensor.eq(&this_color)?.all_dim(2, None)?;
        d.i_mut((i as isize, .., ..))?
            .copy_from_tensor(&mask.ten()?)?;
    }

    Ok(d)
}

#[cfg(test)]
mod test {
    use super::*;
    use flash_powder::StableTorchResult;

    #[test]
    fn test_palette_roundtrip() -> StableTorchResult<()> {
        let palette = generate_color_palette(5)?;

        let mut d = Tensor::zeros(&[5, 6, 6], &Default::default())?;
        for i in 0..5 {
            d.i_mut((i, .., i))?.fill_f64(1.0)?;
        }
        // Oh, yeah lets also set the last column to ones in the background mask since it is black.
        d.i_mut((0, .., 5))?.fill_f64(1.0)?;
        let color_per_pixel = apply_pallette(&palette.ten()?, &d.ten()?)?;
        let color_per_pixel = color_per_pixel.permute(&[2, 0, 1])?.contiguous()?;
        // color_per_pixel
        //     .save_image("/tmp/test_overlay_segmenter_palette_applied.png")
        //     .unwrap();

        let channel_consecutive = color_per_pixel.permute(&[1, 2, 0])?.contiguous()?;
        let back_to_probs = unapply_pallette(5, &channel_consecutive.ten()?)?;

        assert!(back_to_probs.is_equal(&d)?);

        Ok(())
    }
}
