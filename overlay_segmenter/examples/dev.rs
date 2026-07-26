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
    /// Path to the postcard file.
    #[arg()]
    postcard: PathBuf,
    /// Path to write the debug files to.
    #[arg()]
    debug_dir: Option<PathBuf>,
    /// Path to write final file to
    #[arg(short, long, default_value = "/tmp/combined.png")]
    output_path: PathBuf,
}

pub fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let accumulator = overlay_segmenter::accumulator::Accumulator::read_postcard(args.postcard)?;

    let device = if fp::torch::cuda::is_available() {
        fp::Device::CUDA
    } else {
        fp::Device::CPU
    };
    println!("Device used: {device:?}");
    let accumulator = accumulator.into_device(device)?;

    if let Some(debug_dir) = args.debug_dir.as_ref() {
        accumulator.debug_use_accumulation(&debug_dir)?;
    }

    let debug_dir: Option<&std::path::Path> = args.debug_dir.as_ref().map(|a| a.as_path());
    let combined = accumulator.combined_avg(5, 1, debug_dir)?;
    println!("combined: {combined:?}");
    if let Some(debug_dir) = args.debug_dir.as_ref() {
        combined
            .i((1, .., ..))?
            .image_scale_to_domain()?
            .save_image(debug_dir.join(format!("combined_avg.png")))?;

        // let img = combined.to_dynamic_image()?;
        // img.save(debug_out.join(format!("combined_raw.png")))?;
        combined
            .unsqueeze(1)?
            .image_scale_to_domain()?
            .save_image(debug_dir.join(format!("combined_full_batch.png")))?;
    }
    let palette = overlay_segmenter::palette::generate_color_palette(combined.size(0))?;
    let palette = palette.to(&device.into())?;

    let pixel_index = combined.argmax(Some(0), Some(true))?;
    let color_per_pixel = palette
        .index_tensor(&[pixel_index])?
        .squeeze()?
        .to_owned()?;

    //img = tensor_to_image(&color_per_pixel.ten()?)?;
    let color_per_pixel = color_per_pixel.permute(&[2, 0, 1])?.contiguous()?;
    let img = color_per_pixel.to_dynamic_image()?;
    img.save(args.output_path)?;
    Ok(())
}
