import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=[64, 128, 256, 512, 1],
        kernel_size=[4, 4, 4, 4, 4],
        strides=[2, 2, 2, 1, 1],
        prob=0.2,
    ):
        super().__init__()
        layers = []
        for channel, k, stride in zip(out_channels, kernel_size, strides):
            layers += [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channel,
                    kernel_size=(k, k),
                    stride=stride,
                    padding=1,
                    padding_mode="zeros",
                )
            ]
            if in_channels == 1:
                layers += [
                    nn.LeakyReLU(prob),
                ]
            elif channel != 1:
                layers += [
                    nn.InstanceNorm2d(channel),
                    nn.LeakyReLU(prob),
                ]
            in_channels = channel
        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)
