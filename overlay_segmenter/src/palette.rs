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

#[cfg(test)]
mod test {
    use super::*;
    use flash_powder::StableTorchResult;
    use flash_powder::prelude::*;
    use flash_powder_image::prelude::*;

    #[test]
    fn test_palette_roundtrip() -> StableTorchResult<()> {
        let mut d = Tensor::zeros(&[5, 6, 6], &Default::default())?;
        d.i_mut((0..3, 0..3))?.fill_f64(1.0)?;
        d.save_image("/tmp/fp_greyscale_f32.png").unwrap();

        Ok(())
    }
}
