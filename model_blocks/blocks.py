import torch.nn as nn


class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channels=[32, 64, 128],
        kernel=[7, 3, 3],
        strides=[1, 2, 2],
    ):
        super().__init__()
        layers = []
        for out_channel, stride, k in zip(out_channels, strides, kernel):
            layers += [
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=(k, k),
                    stride=stride,
                    padding=k // 2,
                    padding_mode="reflect",
                ),
                nn.InstanceNorm2d(out_channel),
                nn.SiLU(),
            ]

            in_channel = out_channel
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def __iter__(self):
        return iter(self.layers)


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channel=128,
        out_channels=[64, 32, 1],
        kernel=[3, 3, 3],
        upsample=[2, 2, 1],
    ):
        super().__init__()
        layers = []
        for out_channel, up_smpl, k in zip(out_channels, upsample, kernel):
            if up_smpl != 1:
                layers += [
                    nn.Upsample(scale_factor=up_smpl, mode="nearest"),
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=(k, k),
                        stride=1,
                        padding=k // 2,
                        padding_mode="reflect",
                    ),
                    nn.SiLU(),
                ]
            else:
                layers += [
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=(k, k),
                        stride=1,
                        padding=k // 2,
                        padding_mode="reflect",
                    ),
                    nn.Tanh(),
                ]
            in_channel = out_channel
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SpadeResBlock(nn.Module):
    def __init__(self, latent_ch, condmap_in_ch, condmap_out_ch=[32, 64, 128]):
        super().__init__()

        self.norm_2d = nn.InstanceNorm2d(latent_ch)
        layers = []
        for out_channel in condmap_out_ch:
            layers += [
                nn.Conv2d(
                    in_channels=condmap_in_ch,
                    out_channels=out_channel,
                    kernel_size=(3, 3),
                    stride=2,
                    padding=1,
                    padding_mode="reflect",
                ),
                nn.SiLU(),
            ]
            condmap_in_ch = out_channel
        self.cond_map = nn.Sequential(*layers)
        self.gamma_beta = nn.Sequential(
            nn.Conv2d(
                in_channels=condmap_in_ch,
                out_channels=latent_ch * 2,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.SiLU(),
        )

    def forward(self, x, condmap):
        x_norm = self.norm_2d(x)
        condmap = self.cond_map(condmap)

        gamma_beta = self.gamma_beta(condmap)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        return x_norm * (1 + gamma) + beta + x
