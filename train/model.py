#!/usr/bin/env python3
import torch
import torch.nn as nn

"""
| -> | | 64   layer 1
       V
       | -> | | 128  layer 2
              V
              | -> | |  256  layer 3
                     V
                     | -> | | 512 l4  C+ -> | |   And so on, back up.
                            V           ^
                            |      -> | | 
                                 Bottleneck
                     """


class Unet(nn.Module):
    def __init__(self, channels_in=3, channels_out=2):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        # Encoder first.
        self.encoder_conv_1 = conv_block(channels_in, 64)
        # Now a maxpool
        self.maxpool2x2 = nn.MaxPool2d(2)
        self.encoder_conv_2 = conv_block(64, 128)
        # another maxpool, but no weights in that, so can reuse it.
        self.encoder_conv_3 = conv_block(128, 256)
        self.encoder_conv_4 = conv_block(256, 512)

        # Done with encoder

        # Now we're at the bottleneck at the bottom.
        self.bottleneck = conv_block(512, 1024)

        # Decoder next

        # concat with the skip layer

        # Now we have up-conv 2x2, but some people just use upsample?
        # self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # self.decoder_up_level4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_up_level4 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.decoder_conv_4 = conv_block(512 + 1024, 512)

        self.decoder_up_level3 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.decoder_conv_3 = conv_block(256 + 512, 256)

        self.decoder_up_level2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.decoder_conv_2 = conv_block(128 + 256, 128)

        self.decoder_up_level1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.decoder_conv_1 = conv_block(64 + 128, 64)

        # And then we have a conv 1x1, why does this have 2 channels out??
        self.last_conv = nn.Conv2d(
            64,
            out_channels=channels_out,
            kernel_size=1,  # , padding=1
            # Probably don't want that padding, since it makes the output image a different size than the input image.
        )

    def forward(self, x):
        # First encoder
        encoded_level_1 = self.encoder_conv_1(x)
        input_encode_level_2 = self.maxpool2x2(encoded_level_1)

        encoded_level_2 = self.encoder_conv_2(input_encode_level_2)
        input_encode_level_2 = self.maxpool2x2(encoded_level_2)

        encoded_level_3 = self.encoder_conv_3(input_encode_level_2)
        input_encode_level_4 = self.maxpool2x2(encoded_level_3)

        encoded_level_4 = self.encoder_conv_4(input_encode_level_4)

        input_bottleneck = self.maxpool2x2(encoded_level_4)
        output_bottleneck = self.bottleneck(input_bottleneck)

        # Now the decoder, here we concatenate with the skip levels.
        upsample_for_decode_level_4 = self.decoder_up_level4(output_bottleneck)
        input_decode_level_4 = torch.cat(
            [upsample_for_decode_level_4, encoded_level_4], dim=1
        )
        decoded_level_4 = self.decoder_conv_4(input_decode_level_4)

        # And then we repeat that...
        upsample_for_decode_level_3 = self.decoder_up_level3(decoded_level_4)
        input_decode_level_3 = torch.cat(
            [upsample_for_decode_level_3, encoded_level_3], dim=1
        )
        decoded_level_3 = self.decoder_conv_3(input_decode_level_3)

        # And then we repeat that...
        upsample_for_decode_level_2 = self.decoder_up_level2(decoded_level_3)
        input_decode_level_2 = torch.cat(
            [upsample_for_decode_level_2, encoded_level_2], dim=1
        )
        decoded_level_2 = self.decoder_conv_2(input_decode_level_2)

        # And then we repeat that...
        upsample_for_decode_level_1 = self.decoder_up_level1(decoded_level_2)
        input_decode_level_1 = torch.cat(
            [upsample_for_decode_level_1, encoded_level_1], dim=1
        )
        decoded_level_1 = self.decoder_conv_1(input_decode_level_1)

        # And then the last classifier head
        #
        output = self.last_conv(decoded_level_1)

        return output


if __name__ == "__main__":
    batch_size = 1
    input_channels = 3
    output_channels = 2
    model = Unet(channels_in=input_channels, channels_out=output_channels)
    image_width = 512
    image_height = 512 * 2
    im = torch.randn(batch_size, input_channels, image_width, image_height)
    x = model(im)
    print(f"Input shape: {im.shape}", im[:, 0:3, 0:3])
    print(f"Output shape: {x.shape}", x[:, 0:3, 0:3])

    # torch.Size([4, 3, 584, 565])
    model = Unet(channels_in=3, channels_out=1)
    # im = torch.randn(4, 3, 584, 565)
    im = torch.randn(4, 3, 512, 512)
    x = model(im)
    print(f"Input shape: {im.shape}", im[:, 0:3, 0:3])
    print(f"Output shape: {x.shape}", x[:, 0:3, 0:3])
