use fp::prelude::*;
use overlay_segmenter::flash_powder as fp;

use flash_powder_image::prelude::*;

/*
reloader.html;
<html>
<img id="my-image" src="http://localhost:8000/output_mask.png" alt="Live Feed"><script> function reloadFast() {
  const img = document.getElementById('my-image');
    img.src = 'output_mask.png?v=' + new Date().getTime();
  }
  setInterval(reloadFast, 10);
</script>
</html>

python3 -m http.server
*/

use clap::Parser;

use std::path::PathBuf;

/// Run inference on files.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the safetensors file.
    #[arg(short, long)]
    model: PathBuf,

    /// Run the accumulator
    #[arg(short, long)]
    accumulate: bool,
    /// Use the combined accumulation
    #[arg(short, long)]
    enable: bool,

    /// Downscale raw images by this factor.
    #[arg(short, long)]
    downscale: Option<usize>,

    /// Ratio'd updated
    #[arg(short, long)]
    update_ratio: Option<f64>,
}

pub fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let (unet, device, palette) = overlay_segmenter::common_setup(&args.model)?;

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
                        min_observations: 3,
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

    // Next, create the grabber
    let mut grabber = screen_capture::capture()?;

    let res = grabber.resolution();
    println!("Capture reports resolution of: {:?}", res);

    // 64 : (896 + 64), 128:1792
    let display = 0;
    let width = 1792 - 128;
    let height = 896 - 64;
    //let width = 512;
    //let height = 512;
    let x: u32 = 1920 / 2 - width / 2;
    let y: u32 = 1080 / 2 - height / 2;
    grabber.prepare_capture(display, x, y, width, height)?;

    let output_path = "/tmp/screen_section.png";
    use std::time::{Duration, Instant};

    let interval: f32 = 0.001;

    const WRITE_RGB_TO_DISK: bool = true;
    const PRINT_DURATIONS: bool = true;
    let palette = palette.to(&device.into())?;

    let global_start = Instant::now();
    let mut loop_counter = 0;

    loop {
        loop_counter += 1;
        let start = Instant::now();

        grabber.capture_image()?;
        let img = grabber.image()?;

        let img_flat = img.as_flat_samples();
        let img_as_flat_tensor = img_flat.as_ten()?;
        let img_on_device = img_as_flat_tensor.to(&device.into())?;

        // Now we do the channel shuffle, and lets also drop that alpha.
        let rgb_without_dummy_a = img_on_device.narrow(2, 0, 3)?;
        let img_channels_grouped = rgb_without_dummy_a.permute(&[2, 0, 1])?;

        // This is BGR need RGB, flip on the channel direction.
        let colors_correct = img_channels_grouped.flip(&[0])?;

        // Save image to disk, just fo clarity.
        if WRITE_RGB_TO_DISK {
            colors_correct.save_image(&output_path)?;
        }

        let img = colors_correct.image_floatify(&device.into())?;
        let channels_stacked = img.to(&unet.dtype().into())?;
        //let dimension = (.., 64..(896 + 64), 128..1792); // 1664x832
        //let indexed = channels_stacked.i(dimension.clone())?;
        let indexed = channels_stacked.ten()?;

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

        let r = unet.forward(&image.ten()?)?;
        let duration = (std::time::Instant::now() - start).as_secs_f64();
        if PRINT_DURATIONS {
            println!(" Acquisition and prep {duration:.4}s");
        }

        if let Some(accumulator) = accumulator.as_mut() {
            if args.enable {
                accumulator.accumulate_logits_frame(&r.ten()?)?;
            } else {
                accumulator.feed_logits_frame(
                    &r.ten()?,
                    4 - args.downscale.unwrap_or(1).ilog2() as usize,
                    None,
                )?;
            }
        }

        /*
        let r = r.squeeze()?;
        let output = if USE_SOFTMAX_THRESHOLDING {
            let sm = fp::nn::functional::softmax_int(&output, 0, None)?;
            let threshold: Tensor = 0.3.try_into()?;
            let above = sm.ge(&threshold)?;

            sm.mul(&above)?
        } else {
            output.to_owned()?
        };

        let pixel_index = output.argmax(Some(0), Some(true))?;
        let color_per_pixel = palette
            .index_tensor(&[pixel_index])?
            .squeeze()?
            .to_owned()?;

        //img = tensor_to_image(&color_per_pixel.ten()?)?;
        let color_per_pixel = color_per_pixel.permute(&[2, 0, 1])?.contiguous()?;

        if true {
            let img = color_per_pixel.to_dynamic_image()?;
            img.save("/tmp/output_mask.png")?;
        }
        */
        let time_taken = (Instant::now() - start).as_secs_f32();
        let global_time_taken = (Instant::now() - global_start).as_secs_f64() / loop_counter as f64;
        let global_fps = 1.0 / global_time_taken;

        if PRINT_DURATIONS {
            println!(
                "Saved {output_path:?} took {time_taken:.4}s   avg: {global_time_taken:.4}s {global_fps:.1} fps   ",
            );
        }

        if let Some(accumulator) = accumulator.as_mut() {
            let r = accumulator.accumulate_postprocess(None)?; // Next apply the color mask.
            if r.dim() == 0 {
                continue;
            }
            let color_per_pixel =
                overlay_segmenter::palette::apply_pallette(&palette.ten()?, &r.ten()?)?;

            //img = tensor_to_image(&color_per_pixel.ten()?)?;
            let color_per_pixel = color_per_pixel.to(&fp::Device::CPU.into())?;
            let color_per_pixel = color_per_pixel.permute(&[2, 0, 1])?.contiguous()?;
            let img = color_per_pixel.to_dynamic_image()?;

            img.save("/tmp/output_mask.png")?;
        }

        let remaining_sleep = (interval - time_taken).max(0.0);
        std::thread::sleep(Duration::from_secs_f32(remaining_sleep));
    }
}
