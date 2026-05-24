use safetensors::SafeTensors;

use anyhow::bail;
use flash_powder as fp;
use flash_powder::{Ten, Tensor, functional, nn, prelude::*};
use nn::module::{Module, ModuleTensors, ModuleTensorsMut};

#[derive(Debug)]
pub struct UNet {
    encoder_conv_1: nn::Sequential,
    encoder_conv_2: nn::Sequential,
    encoder_conv_3: nn::Sequential,
    encoder_conv_4: nn::Sequential,
    bottleneck: nn::Sequential,
    maxpool2x2: nn::MaxPool2d,
    decoder_up_level4: Box<dyn Module>,
    decoder_conv_4: nn::Sequential,
    decoder_up_level3: Box<dyn Module>,
    decoder_conv_3: nn::Sequential,
    decoder_up_level2: Box<dyn Module>,
    decoder_conv_2: nn::Sequential,
    decoder_up_level1: Box<dyn Module>,
    decoder_conv_1: nn::Sequential,
    last_conv: nn::Conv2d,
}

#[derive(Copy, Clone, Debug)]
pub struct UNetOptions {
    channels_in: usize,
    channels_out: usize,
    use_upconv: bool,
}
impl Default for UNetOptions {
    fn default() -> Self {
        Self {
            channels_in: 3,
            channels_out: 2,
            use_upconv: true,
        }
    }
}

impl UNet {
    pub fn new(options: &UNetOptions) -> Result<Self, anyhow::Error> {
        let maxpool2x2 = nn::MaxPool2d {
            kernel_size: (2, 2),
            options: Default::default(),
        };
        if !options.use_upconv {
            todo!("upsample flavour not yet supported")
        }

        fn conv_block(
            in_channels: usize,
            out_channels: usize,
        ) -> Result<nn::Sequential, anyhow::Error> {
            let mut features: nn::Sequential = Default::default();
            let conv2d_options = functional::Conv2dOptions {
                padding: (1, 1),
                ..Default::default()
            };
            let layer = nn::Conv2d::new(in_channels, out_channels, (3, 3), conv2d_options)?;
            features.push(layer);
            features.push(nn::ReLU);
            let layer = nn::Conv2d::new(out_channels, out_channels, (3, 3), conv2d_options)?;
            features.push(layer);
            features.push(nn::ReLU);
            Ok(features)
        }

        let encoder_conv_1 = conv_block(options.channels_in, 64)?;
        let encoder_conv_2 = conv_block(64, 128)?;
        let encoder_conv_3 = conv_block(128, 256)?;
        let encoder_conv_4 = conv_block(256, 512)?;
        let bottleneck = conv_block(512, 1024)?;

        let conv_transpose2d_options = functional::ConvTranspose2dOptions {
            stride: (2, 2),
            ..Default::default()
        };

        let decoder_up_level4 =
            nn::ConvTranspose2d::new(1024, 1024, (2, 2), conv_transpose2d_options)?.into_boxed();
        let decoder_conv_4 = conv_block(512 + 1024, 512)?;

        let decoder_up_level3 =
            nn::ConvTranspose2d::new(512, 512, (2, 2), conv_transpose2d_options)?.into_boxed();
        let decoder_conv_3 = conv_block(256 + 512, 256)?;

        let decoder_up_level2 =
            nn::ConvTranspose2d::new(256, 256, (2, 2), conv_transpose2d_options)?.into_boxed();
        let decoder_conv_2 = conv_block(128 + 256, 128)?;

        let decoder_up_level1 =
            nn::ConvTranspose2d::new(128, 128, (2, 2), conv_transpose2d_options)?.into_boxed();
        let decoder_conv_1 = conv_block(64 + 128, 64)?;

        let last_conv = nn::Conv2d::new(64, options.channels_out, (1, 1), Default::default())?;
        Ok(UNet {
            maxpool2x2,
            encoder_conv_1,
            encoder_conv_2,
            encoder_conv_3,
            encoder_conv_4,
            bottleneck,
            decoder_up_level4,
            decoder_conv_4,
            decoder_up_level3,
            decoder_conv_3,
            decoder_up_level2,
            decoder_conv_2,
            decoder_up_level1,
            decoder_conv_1,
            last_conv,
        })
    }
}
impl nn::Module for UNet {
    // https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L65
    fn forward(&self, input: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
        let encoded_level_1 = self.encoder_conv_1.forward(input)?;
        let input_encode_level_2 = self.maxpool2x2.forward(&encoded_level_1.ten()?)?;

        let encoded_level_2 = self.encoder_conv_2.forward(&input_encode_level_2.ten()?)?;
        let input_encode_level_3 = self.maxpool2x2.forward(&encoded_level_2.ten()?)?;

        let encoded_level_3 = self.encoder_conv_3.forward(&input_encode_level_3.ten()?)?;
        let input_encode_level_4 = self.maxpool2x2.forward(&encoded_level_3.ten()?)?;

        let encoded_level_4 = self.encoder_conv_4.forward(&input_encode_level_4.ten()?)?;

        let input_bottleneck = self.maxpool2x2.forward(&encoded_level_4.ten()?)?;
        let output_bottleneck = self.bottleneck.forward(&input_bottleneck.ten()?)?;

        // Now the decoder, here we concatenate with the skip levels.
        let upsample_for_decode_level_4 =
            self.decoder_up_level4.forward(&output_bottleneck.ten()?)?;

        let input_decode_level_4 =
            Tensor::cat(&[&upsample_for_decode_level_4, &encoded_level_4], 1)?;
        let decoded_level_4 = self.decoder_conv_4.forward(&input_decode_level_4.ten()?)?;

        // And then we repeat that...
        let upsample_for_decode_level_3 =
            self.decoder_up_level3.forward(&decoded_level_4.ten()?)?;
        let input_decode_level_3 =
            Tensor::cat(&[&upsample_for_decode_level_3, &encoded_level_3], 1)?;
        let decoded_level_3 = self.decoder_conv_3.forward(&input_decode_level_3.ten()?)?;
        // And then we repeat that...
        let upsample_for_decode_level_2 =
            self.decoder_up_level2.forward(&decoded_level_3.ten()?)?;
        let input_decode_level_2 =
            Tensor::cat(&[&upsample_for_decode_level_2, &encoded_level_2], 1)?;
        let decoded_level_2 = self.decoder_conv_2.forward(&input_decode_level_2.ten()?)?;
        // And then we repeat that...
        let upsample_for_decode_level_1 =
            self.decoder_up_level1.forward(&decoded_level_2.ten()?)?;
        let input_decode_level_1 =
            Tensor::cat(&[&upsample_for_decode_level_1, &encoded_level_1], 1)?;
        let decoded_level_1 = self.decoder_conv_1.forward(&input_decode_level_1.ten()?)?;

        // And then the last classifier head
        let output = self.last_conv.forward(&decoded_level_1.ten()?)?;
        Ok(output)
    }

    fn tensors(&self) -> ModuleTensors<'_> {
        ModuleTensors::new()
            .with_namespaced("encoder_conv_1", self.encoder_conv_1.tensors())
            .with_namespaced("encoder_conv_2", self.encoder_conv_2.tensors())
            .with_namespaced("encoder_conv_3", self.encoder_conv_3.tensors())
            .with_namespaced("encoder_conv_4", self.encoder_conv_4.tensors())
            .with_namespaced("bottleneck", self.bottleneck.tensors())
            .with_namespaced("decoder_up_level4", self.decoder_up_level4.tensors())
            .with_namespaced("decoder_conv_4", self.decoder_conv_4.tensors())
            .with_namespaced("decoder_up_level3", self.decoder_up_level3.tensors())
            .with_namespaced("decoder_conv_3", self.decoder_conv_3.tensors())
            .with_namespaced("decoder_up_level2", self.decoder_up_level2.tensors())
            .with_namespaced("decoder_conv_2", self.decoder_conv_2.tensors())
            .with_namespaced("decoder_up_level1", self.decoder_up_level1.tensors())
            .with_namespaced("decoder_conv_1", self.decoder_conv_1.tensors())
            .with_namespaced("last_conv", self.last_conv.tensors())
    }

    fn tensors_mut(&mut self) -> ModuleTensorsMut<'_> {
        ModuleTensorsMut::new()
            .with_namespaced("encoder_conv_1", self.encoder_conv_1.tensors_mut())
            .with_namespaced("encoder_conv_2", self.encoder_conv_2.tensors_mut())
            .with_namespaced("encoder_conv_3", self.encoder_conv_3.tensors_mut())
            .with_namespaced("encoder_conv_4", self.encoder_conv_4.tensors_mut())
            .with_namespaced("bottleneck", self.bottleneck.tensors_mut())
            .with_namespaced("decoder_up_level4", self.decoder_up_level4.tensors_mut())
            .with_namespaced("decoder_conv_4", self.decoder_conv_4.tensors_mut())
            .with_namespaced("decoder_up_level3", self.decoder_up_level3.tensors_mut())
            .with_namespaced("decoder_conv_3", self.decoder_conv_3.tensors_mut())
            .with_namespaced("decoder_up_level2", self.decoder_up_level2.tensors_mut())
            .with_namespaced("decoder_conv_2", self.decoder_conv_2.tensors_mut())
            .with_namespaced("decoder_up_level1", self.decoder_up_level1.tensors_mut())
            .with_namespaced("decoder_conv_1", self.decoder_conv_1.tensors_mut())
            .with_namespaced("last_conv", self.last_conv.tensors_mut())
    }
}

// -------------- Safetensor interface to flash powder's state dict loading --------------
/// Converter to go from safetnsors DType to flash_powder DType
fn safetensor_dtype_to_scalar_type(v: safetensors::Dtype) -> fp::DType {
    match v {
        safetensors::Dtype::F16 => fp::DType::F16,
        safetensors::Dtype::F32 => fp::DType::F32,
        safetensors::Dtype::F64 => fp::DType::F64,
        _ => todo!("todo handle {v:?}"),
    }
}

/// Convert a tensor by `name` from `tensors` into a flash powder Tensor.
fn safetensor_to_tensor(tensors: &SafeTensors, name: &str) -> Result<fp::Tensor, anyhow::Error> {
    if let Ok(tensor_view) = tensors.tensor(name) {
        // Create a tensor of the correct shape and type
        let mut v = fp::Tensor::zeros(
            tensor_view.shape(),
            &fp::factory::TensorOptions {
                dtype: Some(safetensor_dtype_to_scalar_type(tensor_view.dtype())),
                ..Default::default()
            },
        )?;

        // Copy the bytes.
        v.data_mut()?.copy_from_slice(tensor_view.data());
        Ok(v)
    } else {
        bail!("could not find safetensor {name}")
    }
}

#[derive(Copy, Clone, Debug)]
pub struct OurSafeTensors<'a, 'd> {
    st: &'a SafeTensors<'d>,
}
impl<'a, 'd> OurSafeTensors<'a, 'd> {
    pub fn new(st: &'a SafeTensors<'d>) -> Self {
        OurSafeTensors { st }
    }
}

impl<'a, 'd> nn::StateDictAdaptor for OurSafeTensors<'a, 'd> {
    fn tensor(&self, name: &str) -> Option<Tensor> {
        safetensor_to_tensor(self.st, name).ok()
    }
}
impl<'a, 'd> nn::StateDictReader for OurSafeTensors<'a, 'd> {
    fn inner(&self) -> &dyn nn::StateDictAdaptor {
        self
    }
}

// -------------- image::DynamicImage to fp::Tensor --------------
/// Convert dynamic image into [1, 3, h, w] Tensor as floats.
fn image_to_float_tensor(
    image: &image::DynamicImage,
    use_cuda: bool,
) -> Result<fp::Tensor, anyhow::Error> {
    let img = image.to_rgb8();

    // Lets first just tensorify the image, first create an empty tensor.
    let mut t = fp::Tensor::zeros(
        &[img.height() as usize, img.width() as usize, 3],
        &fp::factory::TensorOptions {
            dtype: Some(fp::DType::U8),
            ..Default::default()
        },
    )?;
    // Copy in the data.
    t.data_mut()?.copy_from_slice(img.as_raw().as_slice());

    // Convert that into a float tensor and multiply it by 255.0
    let img_float = t.to(&fp::factory::ToOptions {
        dtype: Some(fp::DType::F32),
        ..Default::default()
    })?;
    let divisor: Tensor = (255.0,).try_into()?;
    let img_tensor_ready = img_float.div(&divisor)?;

    let w = img_tensor_ready.shape()[1];
    let h = img_tensor_ready.shape()[0];

    let channels_stacked = img_tensor_ready.permute(&[2, 0, 1])?;
    // let with_batch = channels_stacked.view(&[1, 3, h, w])?.to_owned()?;
    if use_cuda {
        Ok(channels_stacked.to(&fp::Device::CUDA.into())?)
    } else {
        Ok(channels_stacked.to_owned()?)
    }
}

fn tensor_to_image(ten: &Ten<'_>) -> Result<image::DynamicImage, anyhow::Error> {
    println!("image: {:?}", ten.shape());
    println!("image: {:?}", ten.dtype());

    let mut t = ten.to_owned()?;
    t = t.to(&fp::DType::F32.into())?;
    if ten.dim() != 3 {
        bail!("Expected tensor of dimension 3")
    }

    let width = ten.size(2);
    let height = ten.size(1);

    if ten.size(0) == 1 {
        // Greyscale image.
        if ten.dtype() == fp::DType::F16 {
            let t255: Tensor = (255.0,).try_into()?;
            let t255: Tensor = t255.to(&fp::DType::F16.into())?;
            t = t.mul(&t255)?;
            t = t.to(&fp::DType::U8.into())?;
            let mut raw_pixels: Vec<u8> = vec![0; (width * height * 1) as usize];
            raw_pixels.copy_from_slice(t.u8s_ref()?);
            let img = image::GrayImage::from_raw(width as u32, height as u32, raw_pixels)
                .expect("Container is not the right size for width and height");
            return Ok(image::DynamicImage::ImageLuma8(img));
        } else if ten.dtype() == fp::DType::I64 {
            t = t.to(&fp::DType::U8.into())?;
            let mut raw_pixels: Vec<u8> = vec![0; (width * height * 1) as usize];
            raw_pixels.copy_from_slice(t.u8s_ref()?);
            let img = image::GrayImage::from_raw(width as u32, height as u32, raw_pixels)
                .expect("Container is not the right size for width and height");
            return Ok(image::DynamicImage::ImageLuma8(img));
        } else {
            todo!("data type {:?} is not implemented yet", ten.dtype());
        }
    }

    todo!()
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
            "Run this binary from the 'example_vgg' directory, it looks for  \
            {}, if that doesn't exist:\n Download it from https://download.pytorch.org/models/vgg11-8a719046.pth,\
            convert it to safetensors with ./convert_pth.py",
            weights.display()
        );
        bail!("missing necessary file, bailing out")
    }

    // Load safetensors and wrap
    let data = std::fs::read(weights).expect("Unable to read file");
    let tensors = SafeTensors::deserialize(&data)?;
    let our_safetensor = OurSafeTensors { st: &tensors };

    // Instantiate vgg network and load its weights.
    let mut unet = UNet::new(&UNetOptions::default())?;

    if false {
        let dummy_image = Tensor::zeros(&[1, 3, 256, 256], &Default::default())?;
        let r = unet.forward(&dummy_image.ten()?)?;
    }

    unet.load_state_dict(&our_safetensor)?;

    // Move to cuda if available.
    let use_cuda = fp::torch::cuda::is_available();
    println!("cuda available? {use_cuda:?}");
    if use_cuda {
        unet.to(&fp::Device::CUDA.into())?
    }

    // Iterate over the input arguments and run the network.
    for argument in std::env::args().skip(2) {
        let img = image::ImageReader::open(&argument)?.decode()?;
        let channels_stacked = image_to_float_tensor(&img, use_cuda)?;
        let channels_stacked = channels_stacked.to(&fp::DType::F16.into())?;
        println!("channels_stacked shape: {:?}", channels_stacked.shape());
        let indexed = channels_stacked.i((.., 64..(896 + 64), 128..1792))?;
        let image = indexed.unsqueeze(0)?;
        println!("image shape: {:?}", image.shape());

        let r = unet
            .forward(&image.ten()?)?
            .to(&flash_powder::factory::ToOptions {
                device: Some(fp::Device::CPU),
                ..Default::default()
            })?;
        let t255: Tensor = (255,).try_into()?;
        let max = r.squeeze()?.argmax(Some(0), Some(true))?.mul(&t255)?;
        let img = tensor_to_image(&max.ten()?)?;

        img.save("/tmp/first_light.png")?;
    }

    Ok(())
}
