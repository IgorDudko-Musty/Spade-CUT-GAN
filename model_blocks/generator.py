import torch.nn as nn

from blocks import DownsampleBlock, SpadeResBlock, UpsampleBlock


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample = DownsampleBlock(
            in_channel=3,
            out_channels=[32, 64, 128, 256],
            kernel=[7, 3, 3, 3],
            strides=[1, 2, 2, 2],
        )
        blocks = []
        for _ in range(6):
            blocks += [
                SpadeResBlock(
                    latent_ch=256, condmap_in_ch=3, condmap_out_ch=[64, 128, 256]
                )
            ]
        self.spaderes = nn.Sequential(*blocks)
        self.upsample = UpsampleBlock(
            in_channel=256, out_channels=[128, 64, 32, 1], k=3, upsample=[2, 2, 2, 1]
        )

    def encoder_feat(self, x, feat_layers=[5, 8, 11]):
        features = []
        for num, layer in enumerate(self.downsample):
            x = layer(x)
            if num in feat_layers:
                features.append(x)
        return features

    def forward(self, x):
        features_src = self.encoder_feat(x)

        latent = features_src[-1]
        for block in self.spaderes:
            latent = block(latent, x)

        thermal_image = self.upsample(latent)
        features_trt = self.encoder_feat(thermal_image)
        return thermal_image, features_src, features_trt
