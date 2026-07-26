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
}

pub fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let accumulator = overlay_segmenter::accumulator::Accumulator::read_postcard(args.postcard)?;
    //println!("accumulator: {accumulator:?}");

    let debug_out = std::path::PathBuf::from(&"/tmp/debug_out/");
    accumulator.debug_use_accumulation(&debug_out)?;

    let combined = accumulator.combined_avg(&debug_out)?;
    println!("combined: {combined:?}");
    combined
        .i((1, .., ..))?
        .image_scale_to_domain()?
        .save_image(debug_out.join(format!("combined_avg.png")))?;
    // let img = combined.to_dynamic_image()?;
    // img.save(debug_out.join(format!("combined_raw.png")))?;
    combined
        .unsqueeze(1)?
        .image_scale_to_domain()?
        .save_image(debug_out.join(format!("combined_full_batch.png")))?;

    let palette = overlay_segmenter::palette::generate_color_palette(combined.size(0))?;

    let pixel_index = combined.argmax(Some(0), Some(true))?;
    let color_per_pixel = palette
        .index_tensor(&[pixel_index])?
        .squeeze()?
        .to_owned()?;

    //img = tensor_to_image(&color_per_pixel.ten()?)?;
    let color_per_pixel = color_per_pixel.permute(&[2, 0, 1])?.contiguous()?;

    let img = color_per_pixel.to_dynamic_image()?;
    img.save(debug_out.join(format!("combined_mask.png")))?;
    Ok(())
}
