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
    /// Paths to the input images.
    #[arg()]
    postcard: PathBuf,
}

pub fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let mut accumulator =
        overlay_segmenter::accumulator::Accumulator::read_postcard(args.postcard)?;
    println!("accumulator: {accumulator:?}");
    Ok(())
}
