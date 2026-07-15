use anyhow::bail;
use fp::prelude::*;
use fp::{Device, Tensor};
use overlay_segmenter::flash_powder as fp;
use overlay_segmenter::model::{UNet, UNetOptions};

use flash_powder_image::prelude::*;
use overlay_segmenter::generate_color_palette;

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

    // Next, create the grabber
    let mut grabber = screen_capture::capture()?;

    let res = grabber.resolution();
    println!("Capture reports resolution of: {:?}", res);

    let display = 0;
    let width = 512;
    let height = 512;
    let x = 1920 / 2 - width / 2;
    let y = 1080 / 2 - height / 2;
    grabber.prepare_capture(display, x, y, width, height)?;

    let output_path = "/tmp/screen_section.png";
    use std::time::{Duration, Instant};

    let interval: f32 = 0.001;

    const WRITE_RGB_TO_DISK: bool = true;
    const PRINT_DURATIONS: bool = true;
    const USE_SOFTMAX_THRESHOLDING: bool = true;
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
        let image = channels_stacked.unsqueeze(0)?;

        let r = unet.forward(&image.ten()?)?;
        let duration = (std::time::Instant::now() - start).as_secs_f64();
        if PRINT_DURATIONS {
            println!(" Acquisition and prep {duration:.4}s");
        }

        let output = r.squeeze()?;

        let output = if USE_SOFTMAX_THRESHOLDING {
            let sm = fp::functional::softmax_int(&output, 0, None)?;
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

        let time_taken = (Instant::now() - start).as_secs_f32();
        let global_time_taken = (Instant::now() - global_start).as_secs_f64() / loop_counter as f64;
        let global_fps = 1.0 / global_time_taken;

        if PRINT_DURATIONS {
            println!(
                "Saved {output_path:?} took {time_taken:.4}s   avg: {global_time_taken:.4}s {global_fps:.1} fps  ({:?})",
                color_per_pixel.shape()
            );
        }
        let remaining_sleep = (interval - time_taken).max(0.0);
        std::thread::sleep(Duration::from_secs_f32(remaining_sleep));
    }
}
