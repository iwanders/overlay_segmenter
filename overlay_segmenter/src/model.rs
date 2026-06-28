//! UNet Model definition
//!
//! This should be kept in sync to the Python counterpart.

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

        let decoder_up_level4 = if options.use_upconv {
            nn::ConvTranspose2d::new(1024, 1024, (2, 2), conv_transpose2d_options)?.into_boxed()
        } else {
            todo!("upsample flavour not yet supported")
        };
        let decoder_conv_4 = conv_block(512 + 1024, 512)?;

        let decoder_up_level3 = if options.use_upconv {
            nn::ConvTranspose2d::new(512, 512, (2, 2), conv_transpose2d_options)?.into_boxed()
        } else {
            todo!("upsample flavour not yet supported")
        };
        let decoder_conv_3 = conv_block(256 + 512, 256)?;

        let decoder_up_level2 = if options.use_upconv {
            nn::ConvTranspose2d::new(256, 256, (2, 2), conv_transpose2d_options)?.into_boxed()
        } else {
            todo!("upsample flavour not yet supported")
        };
        let decoder_conv_2 = conv_block(128 + 256, 128)?;

        let decoder_up_level1 = if options.use_upconv {
            nn::ConvTranspose2d::new(128, 128, (2, 2), conv_transpose2d_options)?.into_boxed()
        } else {
            todo!("upsample flavour not yet supported")
        };
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

    pub fn channels_out(&self) -> usize {
        self.last_conv.weight.size(0)
    }
}
impl nn::Module for UNet {
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
