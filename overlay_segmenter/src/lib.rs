use anyhow::bail;
pub use flash_powder;
use flash_powder as fp;
use flash_powder::{Tensor, nn, prelude::*};
use flash_powder_image::prelude::*;
use nn::module::Module;
pub mod palette;

pub mod model;
use model::{UNet, UNetOptions};

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

    let load_options = nn::StateDictLoadOptions {
        // Blow away the current tensor sizes and take whatever is in the safetensors.
        assign: true,
        ..Default::default()
    };
    unet.load_state_dict(&our_safetensor, &load_options)?;

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
    let palette = palette::generate_color_palette(unet.channels_out())?;
    const COLOR_MASK_OUTPUT: bool = true;

    let palette = palette.to(&device.into())?;
    // Iterate over the input arguments and run the network.
    for argument in std::env::args().skip(2) {
        let path = std::path::PathBuf::from(&argument);
        if argument.contains("_mask.png")
            || argument.contains("_values.png")
            || argument.contains("_batch.png")
        {
            // println!("  Ignoring {argument:?} because it looks like our output");
            continue;
        }
        let img = Tensor::read_image(&argument)?.image_floatify(&device.into())?;
        let channels_stacked = img.to(&unet.dtype().into())?;
        let dimension = (.., 64..(896 + 64), 128..1792); // 1664x832
        let indexed = channels_stacked.i(dimension.clone())?;
        let image = indexed.unsqueeze(0)?;

        let start = std::time::Instant::now();
        let r = unet.forward(&image.ten()?)?;

        let mut mask_image = Tensor::zeros(
            &[unet.channels_out(), img.size(1), img.size(2)],
            &Default::default(),
        )?;
        mask_image
            .i_mut(dimension)?
            .copy_from_tensor(&r.squeeze()?)?;
        let duration = (std::time::Instant::now() - start).as_secs_f64();
        println!("{argument}: {duration:.2}s"); // First 0.29s, subseq 0.18

        let img = if COLOR_MASK_OUTPUT {
            let pixel_index = mask_image.argmax(Some(0), Some(true))?;
            let color_per_pixel = palette
                .index_tensor(&[pixel_index])?
                .squeeze()?
                .to_owned()?;

            //img = tensor_to_image(&color_per_pixel.ten()?)?;
            let color_per_pixel = color_per_pixel.to(&fp::Device::CPU.into())?;
            let color_per_pixel = color_per_pixel.permute(&[2, 0, 1])?.contiguous()?;
            color_per_pixel.to_dynamic_image()?
        } else {
            let t255: Tensor = 255.try_into()?;
            let max = mask_image.argmax(Some(0), Some(true))?.mul(&t255)?;
            let max_squeezed = max.squeeze()?;
            // img = tensor_to_image(&max_squeezed.ten()?)?;
            let max_squeezed = max_squeezed.to(&fp::Device::CPU.into())?;
            max_squeezed.to_dynamic_image()?
        };

        let stem = path.file_stem().unwrap().display();
        img.save(format!("/tmp/{}_mask.png", stem))?;
        // Lets also make value, to do that, we make mask_image from [C, H, W] into [C, 1, H, W].
        let batch_of_values = mask_image.unsqueeze(1)?;
        let batch_of_values_normalized = batch_of_values.image_scale_to_domain()?;
        batch_of_values_normalized.save_image(format!("/tmp/{}_values.png", stem))?;
    }

    Ok(())
}
