pub use flash_powder;
use flash_powder as fp;
use flash_powder::{Device, Tensor, nn};
use nn::module::Module;
pub mod palette;

pub mod model;
use model::{UNet, UNetOptions};

use anyhow::Context;
pub fn create_unet(safetensors_path: &std::path::Path) -> Result<UNet, anyhow::Error> {
    // Load safetensors and wrap
    let data = std::fs::read(&safetensors_path)
        .with_context(|| format!("Could not open safetensors path; {safetensors_path:?}"))?;
    let tensors = flash_powder_safetensors::safetensors::SafeTensors::deserialize(&data)?;
    let our_safetensor = flash_powder_safetensors::SafetensorReader::from_safetensors(&tensors);

    // Instantiate the network and load its weights.
    let mut unet = UNet::new(&UNetOptions::default())?;

    let load_options = fp::nn::StateDictLoadOptions {
        // Blow away the current tensor sizes and take whatever is in the safetensors.
        assign: true,
        ..Default::default()
    };
    unet.load_state_dict(&our_safetensor, &load_options)?;
    Ok(unet)
}

pub fn common_setup(
    safetensors_path: &std::path::Path,
) -> Result<(UNet, Device, Tensor), anyhow::Error> {
    // Instantiate the network and load its weights.
    let mut unet = create_unet(&safetensors_path)?;

    // Move to cuda if available.
    let device = if fp::torch::cuda::is_available() {
        fp::Device::CUDA
    } else {
        fp::Device::CPU
    };
    println!("Device used: {device:?}");
    unet.to(&device.into())?;

    println!("unet channels out: {:?}", unet.channels_out());
    let palette = palette::generate_color_palette(unet.channels_out())?;

    Ok((unet, device, palette))
}
