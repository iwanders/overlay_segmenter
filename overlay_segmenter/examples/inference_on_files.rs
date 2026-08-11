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
    /// Use the combined accumulation
    #[arg(short, long)]
    enable: bool,

    /// Path to write the aligned data to.
    #[arg(long)]
    accumulate_write: Option<PathBuf>,

    /// Downscale raw images by this factor.
    #[arg(short, long)]
    downscale: Option<usize>,

    /// Ratio'd updated
    #[arg(short, long)]
    update_ratio: Option<f64>,
}

pub fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let (unet, device, palette) = common_setup(&args.model)?;

    let mut accumulator = if args.accumulate {
        let mut accum = overlay_segmenter::accumulator::Accumulator::new();
        Some(if args.enable {
            let merge_mode = if let Some(ratio) = args.update_ratio {
                overlay_segmenter::accumulator::MergeMode::RatioUpdate(
                    overlay_segmenter::accumulator::RatioUpdateMergeConfig { update_rate: ratio },
                )
            } else {
                overlay_segmenter::accumulator::MergeMode::Buffered(
                    overlay_segmenter::accumulator::BufferedMergeConfig {
                        min_observations: 1,
                        area_radius: 20,
                    },
                )
            };
            let config = overlay_segmenter::accumulator::AccumulationConfig {
                fit_against_previous_frames: 3,

                layer_count: 3,
                merge_mode,
            };
            accum.enable_accumulator(config)?;
            accum
        } else {
            accum
        })
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
        std::fs::create_dir_all(&output_path)?;

        let img = Tensor::read_image(&path)?;
        let img = img.image_floatify(&device.into())?;
        let img_orig_res = img.shape();
        let channels_stacked = img.to(&unet.dtype().into())?;
        let dimension = (.., 64..(896 + 64), 128..1792); // 1664x832
        let indexed = channels_stacked.i(dimension.clone())?;

        println!("indexed before: {:?}", indexed.shape());
        let (orig_crop_w, orig_crop_h) = (indexed.isize(-1), indexed.isize(-2));
        let indexed = if let Some(downscale) = args.downscale {
            let w = orig_crop_w / downscale;
            let h = orig_crop_h / downscale;
            indexed
                .image_resize(
                    [h, w],
                    flash_powder::nn::functional::InterpolateAlgorithm::Nearest,
                )?
                .squeeze()?
                .to_owned()?
        } else {
            indexed.to_owned()?
        };
        println!("indexed size: {:?}", indexed.shape());
        let image = indexed.unsqueeze(0)?;
        println!("image size: {:?}", image.shape());

        let start = std::time::Instant::now();
        let r = unet.forward(&image.ten()?)?;

        if let Some(accumulator) = accumulator.as_mut() {
            if args.enable {
                accumulator.accumulate_logits_frame(&r.ten()?)?;
            } else {
                accumulator.feed_logits_frame(
                    &r.ten()?,
                    4 - args.downscale.unwrap_or(1).ilog2() as usize,
                    Some(&output_path),
                )?;
            }
        }
        println!("done accumulating");

        let mut mask_image = Tensor::zeros(
            &[unet.channels_out(), img_orig_res[1], img_orig_res[2]],
            &Default::default(),
        )?;
        let r = if let Some(_downscale) = args.downscale {
            r.image_resize(
                [orig_crop_h, orig_crop_w],
                flash_powder::nn::functional::InterpolateAlgorithm::Nearest,
            )?
            .squeeze()?
            .to_owned()?
        } else {
            r
        };
        mask_image
            .i_mut(dimension)?
            .copy_from_tensor(&r.squeeze()?)?;
        let duration = (std::time::Instant::now() - start).as_secs_f64();
        println!("{path:?}: {duration:.2}s"); // First 0.29s, subseq 0.18

        println!("output_path: {output_path:?}");

        // Next apply the color mask.
        let color_per_pixel =
            overlay_segmenter::palette::apply_pallette(&palette.ten()?, &mask_image.ten()?)?;

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
            if args.enable {
                let r = accumulator.accumulate_postprocess(Some(&path))?; // Next apply the color mask.
                let color_per_pixel =
                    overlay_segmenter::palette::apply_pallette(&palette.ten()?, &r.ten()?)?;

                //img = tensor_to_image(&color_per_pixel.ten()?)?;
                let color_per_pixel = color_per_pixel.to(&fp::Device::CPU.into())?;
                let color_per_pixel = color_per_pixel.permute(&[2, 0, 1])?.contiguous()?;
                let img = color_per_pixel.to_dynamic_image()?;

                img.save("/tmp/supercombine.png")?;
            } else {
                accumulator.debug_use_accumulation(&path)?;
            }
        }
        if let Some(write_path) = args.accumulate_write.as_ref() {
            accumulator.write_postcard(&write_path)?;
        }
    }

    Ok(())
}
