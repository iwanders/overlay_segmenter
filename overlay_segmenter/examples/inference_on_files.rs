use flash_powder as fp;
use flash_powder::Tensor;
use flash_powder::prelude::*;
use flash_powder_image::prelude::*;
use overlay_segmenter::common_setup;

use clap::Parser;

use std::path::PathBuf;

/// Run inference on files.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the safetensors file.
    #[arg(short, long)]
    model: PathBuf,

    /// Paths to the input images.
    #[arg()]
    files: Vec<PathBuf>,

    /// Output directory, default is to output adjacent to the images.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Write values
    #[arg(short, long)]
    write_values: bool,

    /// Run the accumulator
    #[arg(short, long)]
    accumulate: bool,
}

pub fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let (unet, device, palette) = common_setup(&args.model)?;

    let mut accumulator = if args.accumulate {
        Some(overlay_segmenter::accumulator::Accumulator::new())
    } else {
        None
    };

    let palette = palette.to(&device.into())?;
    // Iterate over the input arguments and run the network.
    for path in args.files.iter() {
        let stem = path
            .file_stem()
            .unwrap()
            .to_str()
            .expect("files passed should have a stem, else its a dir");
        if stem.contains("_mask") || stem.contains("_values") || stem.contains("_batch") {
            // println!("  Ignoring {argument:?} because it looks like our output");
            continue;
        }

        let output_path = args
            .output
            .as_ref()
            .cloned()
            .unwrap_or(path.parent().unwrap().to_owned());

        let img = Tensor::read_image(&path)?.image_floatify(&device.into())?;
        let channels_stacked = img.to(&unet.dtype().into())?;
        let dimension = (.., 64..(896 + 64), 128..1792); // 1664x832
        let indexed = channels_stacked.i(dimension.clone())?;
        let image = indexed.unsqueeze(0)?;

        let start = std::time::Instant::now();
        let r = unet.forward(&image.ten()?)?;

        if let Some(accumulator) = accumulator.as_mut() {
            accumulator.feed_logits_frame(&r.ten()?, Some(&output_path))?;
        }
        println!("done accumulating");

        let mut mask_image = Tensor::zeros(
            &[unet.channels_out(), img.size(1), img.size(2)],
            &Default::default(),
        )?;
        mask_image
            .i_mut(dimension)?
            .copy_from_tensor(&r.squeeze()?)?;
        let duration = (std::time::Instant::now() - start).as_secs_f64();
        println!("{path:?}: {duration:.2}s"); // First 0.29s, subseq 0.18

        println!("output_path: {output_path:?}");

        // Next apply the color mask.

        let pixel_index = mask_image.argmax(Some(0), Some(true))?;
        let color_per_pixel = palette
            .index_tensor(&[pixel_index])?
            .squeeze()?
            .to_owned()?;

        //img = tensor_to_image(&color_per_pixel.ten()?)?;
        let color_per_pixel = color_per_pixel.to(&fp::Device::CPU.into())?;
        let color_per_pixel = color_per_pixel.permute(&[2, 0, 1])?.contiguous()?;
        let img = color_per_pixel.to_dynamic_image()?;

        let stem = path.file_stem().unwrap().display();
        img.save(output_path.join(&format!("{}_mask.png", stem)))?;

        if args.write_values {
            // Lets also make value, to do that, we make mask_image from [C, H, W] into [C, 1, H, W].
            let batch_of_values = mask_image.unsqueeze(1)?;
            let batch_of_values_normalized = batch_of_values.image_scale_to_domain()?;
            batch_of_values_normalized
                .save_image(output_path.join(&format!("{}_values.png", stem)))?;
        }
    }

    if let Some(accumulator) = accumulator.as_mut() {
        if let Some(path) = args.output.as_ref() {
            accumulator.debug_use_accumulation(&path)?;
        }
    }

    Ok(())
}
