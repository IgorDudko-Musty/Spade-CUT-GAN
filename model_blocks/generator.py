import torch.nn as nn

from .blocks import DownsampleBlock, SpadeResBlock, UpsampleBlock


class Generator(nn.Module):
    def __init__(
        self,
        ds_in_ch=3,
        ds_out_ch=[32, 64, 128, 256],
        ds_kernels=[7, 3, 3, 3],
        ds_strides=[1, 2, 2, 2],
        feat_layers=[5, 8, 11],
        sp_latent=256,
        sp_cmap_in_ch=3,
        sp_cmap_out_ch=[64, 128, 256],
        sp_count=6,
        up_in_channel=256,
        out_out_channels=[128, 64, 32, 1],
        up_kernels=[3, 3, 3, 3],
        upsample_sizes=[2, 2, 2, 1],
    ):
        super().__init__()

        self.downsample = DownsampleBlock(
            in_channel=ds_in_ch,
            out_channels=ds_out_ch,
            kernel=ds_kernels,
            strides=ds_strides,
        )
        blocks = []
        for _ in range(sp_count):
            blocks += [
                SpadeResBlock(
                    latent_ch=sp_latent,
                    condmap_in_ch=sp_cmap_in_ch,
                    condmap_out_ch=sp_cmap_out_ch,
                )
            ]
        self.spaderes = nn.Sequential(*blocks)
        self.upsample = UpsampleBlock(
            in_channel=up_in_channel,
            out_channels=out_out_channels,
            kernel=up_kernels,
            upsample=upsample_sizes,
        )

        self.feat_layers = feat_layers

    def encoder_feat(self, x, feat_layers=[5, 8, 11]):
        features = []
        for num, layer in enumerate(self.downsample):
            x = layer(x)
            if num in feat_layers:
                features.append(x)
        return features

    def forward(self, x, for_discriminator=False):
        if for_discriminator:
            x = self.downsample(x)
            for block in self.spaderes:
                latent = block(latent, x)
            thermal_image = self.upsample(latent)
            return thermal_image
        else:
            features_src = self.encoder_feat(x, self.feat_layers)

            latent = features_src[-1]
            for block in self.spaderes:
                latent = block(latent, x)

            thermal_image = self.upsample(latent)
            features_trt = self.encoder_feat(thermal_image.repeat(1, 3, 1, 1), self.feat_layers)
            return thermal_image, features_src, features_trt
