use flash_powder::prelude::*;
use flash_powder::{StableTorchResult, Tensor};
use flash_powder_image::prelude::*;

pub fn load_mask_image<P: AsRef<std::path::Path>>(
    v: P,
    value_to_depth_index: &[(u8, usize)],
) -> StableTorchResult<Tensor> {
    let p: &std::path::Path = &v.as_ref();

    // Read the greyscale image
    let img = Tensor::read_image(p)?;

    let class_count = value_to_depth_index
        .iter()
        .map(|(_, a)| *a)
        .max()
        .ok_or(anyhow::anyhow!("should have more than one class"))?;

    // Next, we can create a tensor of the appropriate size.
    let mut d = Tensor::zeros(
        &[class_count + 1, img.isize(1), img.isize(2)],
        &flash_powder::DType::F16.into(),
    )?;

    // Next, iterate over the classess and assign into the correct layer.
    for (greyscale_value, class_index) in value_to_depth_index {
        let value: Tensor = (*greyscale_value).try_into()?;
        // Create the boolean mask.
        let m = img.eq(&value)?;
        d.i_mut((*class_index as isize, .., ..))?
            .copy_from_tensor(&m.squeeze()?)?;
    }

    Ok(d)
}

#[cfg(test)]
mod test {
    use super::*;
    use flash_powder::StableTorchResult;

    #[test]
    fn test_mask_loader() -> StableTorchResult<()> {
        let palette = crate::palette::generate_color_palette(5)?;

        let mut greyscale = Tensor::zeros(&[1, 6, 6], &flash_powder::DType::U8.into())?;
        let mut d = Tensor::zeros(&[5, 6, 6], &Default::default())?;
        let mut mapping = vec![];
        for i in 0..5 {
            greyscale.i_mut((.., .., i))?.fill_with(i as u8)?;
            mapping.push((i as u8, i as usize));
            d.i_mut((i, .., i))?.fill_f64(1.0)?;
        }
        d.i_mut((0, .., 5))?.fill_with(1.0)?;

        let color_per_pixel = crate::palette::apply_pallette(&palette.ten()?, &d.ten()?)?;
        let color_per_pixel = color_per_pixel.permute(&[2, 0, 1])?.contiguous()?;
        color_per_pixel
            .save_image("/tmp/test_mask_loader_palette.png")
            .unwrap();
        greyscale
            .save_image("/tmp/test_mask_loader_greyscale.png")
            .unwrap();

        let and_back = load_mask_image(&"/tmp/test_mask_loader_greyscale.png", &mapping)?;

        let channel_consecutive = color_per_pixel.permute(&[1, 2, 0])?.contiguous()?;
        let back_to_probs = crate::palette::unapply_pallette(5, &channel_consecutive.ten()?)?;

        assert!(and_back.is_equal(&back_to_probs)?);

        Ok(())
    }
}
