use anyhow::bail;
use flash_powder as fp;
use flash_powder::{Ten, Tensor, nn, prelude::*};
use nn::module::Module;

pub mod model;
use model::{UNet, UNetOptions};

// -------------- image::DynamicImage to fp::Tensor --------------
/// Convert dynamic image into [1, 3, h, w] Tensor as floats.
fn image_to_float_tensor(
    image: &image::DynamicImage,
    use_cuda: bool,
) -> Result<fp::Tensor, anyhow::Error> {
    let img = image.to_rgb8();

    // Lets first just tensorify the image, first create an empty tensor.
    let mut t = fp::Tensor::zeros(
        &[img.height() as usize, img.width() as usize, 3],
        &fp::factory::TensorOptions {
            dtype: Some(fp::DType::U8),
            ..Default::default()
        },
    )?;
    // Copy in the data.
    t.data_mut()?.copy_from_slice(img.as_raw().as_slice());

    // Convert that into a float tensor and multiply it by 255.0
    let img_float = t.to(&fp::factory::ToOptions {
        dtype: Some(fp::DType::F32),
        ..Default::default()
    })?;
    let divisor: Tensor = 255.0.try_into()?;
    let img_tensor_ready = img_float.div(&divisor)?;

    let channels_stacked = img_tensor_ready.permute(&[2, 0, 1])?;
    if use_cuda {
        Ok(channels_stacked.to(&fp::Device::CUDA.into())?)
    } else {
        Ok(channels_stacked.to_owned()?)
    }
}

fn tensor_to_image(ten: &Ten<'_>) -> Result<image::DynamicImage, anyhow::Error> {
    if ten.dim() != 2 && ten.dim() != 3 {
        bail!(
            "Tensor dimension of 2 or 3 was expected, got {:?}",
            ten.shape()
        )
    }

    if ten.dim() == 2 {
        let mut t = ten.to_owned()?;
        t = t.to(&fp::DType::F32.into())?;
        // Greyscale image.
        let width = ten.size(1);
        let height = ten.size(0);

        if ten.dtype() == fp::DType::F16 {
            let t255: Tensor = 255.0.try_into()?;
            let t255: Tensor = t255.to(&fp::DType::F16.into())?;
            t = t.mul(&t255)?;
            t = t.to(&fp::DType::U8.into())?;
            let mut raw_pixels: Vec<u8> = vec![0; width * height];
            raw_pixels.copy_from_slice(t.u8s_ref()?);
            let img = image::GrayImage::from_raw(width as u32, height as u32, raw_pixels)
                .expect("container is not the right size for width and height");
            Ok(image::DynamicImage::ImageLuma8(img))
        } else if ten.dtype() == fp::DType::I64 {
            t = t.to(&fp::DType::U8.into())?;
            let mut raw_pixels: Vec<u8> = vec![0; width * height];
            raw_pixels.copy_from_slice(t.u8s_ref()?);
            let img = image::GrayImage::from_raw(width as u32, height as u32, raw_pixels)
                .expect("container is not the right size for width and height");
            Ok(image::DynamicImage::ImageLuma8(img))
        } else {
            todo!("data type {:?} is not implemented yet", ten.dtype());
        }
    } else if ten.dim() == 3 {
        let width = ten.size(1);
        let height = ten.size(0);
        let channels = ten.size(2);
        if channels != 3 {
            todo!("images must have 3 channels");
        }

        let mut t = ten.to_owned()?;
        // Interpret as rgb.
        if t.dtype() == fp::DType::F16 || t.dtype() == fp::DType::F32 || t.dtype() == fp::DType::F64
        {
            // It is a float, so move it to the 255.0 space.
            t = t.to(&fp::DType::F64.into())?; // widen to full resolution.
            let t255: Tensor = 255.0.try_into()?;
            let t255: Tensor = t255.to(&fp::DType::F64.into())?;
            t = t.mul(&t255)?;
            // Now it is float 0.0-255.0
        }
        // Which is converted to u8 here.
        t = t.to(&fp::DType::U8.into())?;
        t = t.contiguous()?;
        let mut raw_pixels: Vec<u8> = vec![0; width * height * 3];
        raw_pixels.copy_from_slice(t.u8s_ref()?);
        let img = image::RgbImage::from_raw(width as u32, height as u32, raw_pixels)
            .expect("Container is not the right size for width and height");
        Ok(image::DynamicImage::ImageRgb8(img))
    } else {
        todo!("image with more than one channel are not yet handled")
    }
}

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

fn generate_color_palette(class_count: usize) -> Result<Tensor, anyhow::Error> {
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

pub fn main() -> Result<(), anyhow::Error> {
    use std::path::PathBuf;

    let args = std::env::args().collect::<Vec<String>>();

    let safetensors_path = if let Some(path) = args.get(1) {
        path.to_owned()
    } else {
        bail!("missing safetensors argument")
    };
    // Verify weights exist, if not give a nice warning.
    let weights = PathBuf::from(safetensors_path);
    if !weights.is_file() {
        eprintln!(
            "Missing {:?}, path should be to safetensors file.",
            weights.display()
        );
        bail!("missing necessary file, bailing out")
    }

    // Load safetensors and wrap
    let data = std::fs::read(weights).expect("Unable to read file");
    let tensors = flash_powder_safetensors::safetensors::SafeTensors::deserialize(&data)?;
    let our_safetensor = flash_powder_safetensors::SafetensorReader::from_safetensors(&tensors);

    // Instantiate the network and load its weights.
    let mut unet = UNet::new(&UNetOptions::default())?;

    if false {
        let dummy_image = Tensor::zeros(&[1, 3, 256, 256], &Default::default())?;
        let _r = unet.forward(&dummy_image.ten()?)?;
    }

    unet.load_state_dict(&our_safetensor)?;

    // Move to cuda if available.
    let use_cuda = fp::torch::cuda::is_available();
    println!("cuda available? {use_cuda:?}");
    if use_cuda {
        unet.to(&fp::Device::CUDA.into())?
    }

    println!("unet channels out: {:?}", unet.channels_out());
    let palette = generate_color_palette(unet.channels_out())?;
    const COLOR_MASK_OUTPUT: bool = true;

    // Iterate over the input arguments and run the network.
    for argument in std::env::args().skip(2) {
        let img = image::ImageReader::open(&argument)?.decode()?;
        let channels_stacked = image_to_float_tensor(&img, use_cuda)?;
        let channels_stacked = channels_stacked.to(&unet.dtype().into())?;
        let dimension = (.., 64..(896 + 64), 128..1792); // 1664x832
        let indexed = channels_stacked.i(dimension.clone())?;
        let image = indexed.unsqueeze(0)?;

        let start = std::time::Instant::now();
        let r = unet.forward(&image.ten()?)?;
        println!("r: \n{:?}", r);
        let r = r.to(&flash_powder::factory::ToOptions {
            device: Some(fp::Device::CPU),
            ..Default::default()
        })?;

        let mut mask_image = Tensor::zeros(
            &[
                unet.channels_out(),
                img.height() as usize,
                img.width() as usize,
            ],
            &Default::default(),
        )?;
        mask_image.i_mut(dimension)?.copy_(&r.squeeze()?)?;
        let duration = (std::time::Instant::now() - start).as_secs_f64();
        println!("{argument}: {duration:.2}s"); // First 0.29s, subseq 0.18

        let img;
        if COLOR_MASK_OUTPUT {
            let pixel_index = mask_image.argmax(Some(0), Some(true))?;
            let color_per_pixel = palette
                .index_tensor(&[pixel_index])?
                .squeeze()?
                .to_owned()?;
            println!("color_per_pixel shape: {:?}", color_per_pixel.shape());

            img = tensor_to_image(&color_per_pixel.ten()?)?;
        } else {
            let t255: Tensor = 255.try_into()?;
            let max = mask_image.argmax(Some(0), Some(true))?.mul(&t255)?;
            let max_squeezed = max.squeeze()?;
            img = tensor_to_image(&max_squeezed.ten()?)?;
        }

        img.save("/tmp/first_light.png")?;
    }

    Ok(())
}
