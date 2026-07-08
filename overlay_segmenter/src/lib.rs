use anyhow::bail;
use flash_powder as fp;
use flash_powder::{Ten, Tensor, nn, prelude::*};
use flash_powder_image::{TensorFromImage, TensorToImage};
use nn::module::Module;

pub mod model;
use model::{UNet, UNetOptions};

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
    let device = if use_cuda {
        fp::Device::CUDA
    } else {
        fp::Device::CPU
    };

    println!("unet channels out: {:?}", unet.channels_out());
    let palette = generate_color_palette(unet.channels_out())?;
    const COLOR_MASK_OUTPUT: bool = true;

    let f32_255: Tensor = 255.0.try_into()?;
    // Iterate over the input arguments and run the network.
    for argument in std::env::args().skip(2) {
        let img = Tensor::read_image(&argument)?
            .to(&fp::DType::F32.into())?
            .to(&device.into())?
            .div(&f32_255)?;
        let channels_stacked = img.to(&unet.dtype().into())?;
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
            &[unet.channels_out(), img.size(1), img.size(2)],
            &Default::default(),
        )?;
        mask_image
            .i_mut(dimension)?
            .copy_from_tensor(&r.squeeze()?)?;
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

            //img = tensor_to_image(&color_per_pixel.ten()?)?;
            let color_per_pixel = color_per_pixel.permute(&[2, 0, 1])?.contiguous()?;
            img = color_per_pixel.to_dynamic_image()?;
        } else {
            let t255: Tensor = 255.try_into()?;
            let max = mask_image.argmax(Some(0), Some(true))?.mul(&t255)?;
            let max_squeezed = max.squeeze()?;
            // img = tensor_to_image(&max_squeezed.ten()?)?;
            img = max_squeezed.to_dynamic_image()?;
        }

        img.save("/tmp/first_light.png")?;
    }

    Ok(())
}
