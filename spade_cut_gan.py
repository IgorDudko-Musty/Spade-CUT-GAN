import torch.nn as nn
import yaml

from model_blocks.discriminator import Discriminator
from model_blocks.generator import Generator
from model_blocks.patchnce_head import PatchNCE
from model_losses.gan_loss import GANLoss
from model_losses.patchnce_loss import PatchNCELoss


class SpadeCUTGAN(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        with open(config, "r") as file:
            config = yaml.safe_load(file)
        self.generator = Generator(
            ds_in_ch=config["generator"]["ds_in_ch"],
            ds_out_ch=config["generator"]["ds_out_ch"],
            ds_kernels=config["generator"]["ds_kernels"],
            ds_strides=config["generator"]["ds_strides"],
            sp_latent=config["generator"]["sp_latent"],
            sp_cmap_in_ch=config["generator"]["sp_cmap_in_ch"],
            sp_cmap_out_ch=config["generator"]["sp_cmap_out_ch"],
            sp_count=config["generator"]["sp_count"],
            up_in_channel=config["generator"]["up_in_channel"],
            out_out_channels=config["generator"]["out_out_channels"],
            up_kernels=config["generator"]["up_kernels"],
            upsample_sizes=config["generator"]["upsample_sizes"],
        )
        self.discriminator = Discriminator(
            in_channels=config["discriminator"]["in_channels"],
            out_channels=config["discriminator"]["out_channels"],
            kernel_size=config["discriminator"]["kernel_size"],
            strides=config["discriminator"]["strides"],
            prob=config["discriminator"]["prob"],
        )
        self.patchnce = PatchNCE(mlp_in=config["patchnce"]["mlp_in"], mlp_out=config["patchnce"]["mlp_out"])
        self.gan_loss = GANLoss()
        self.patchnce_loss = PatchNCELoss(config["patchnceLoss"]["temperature"], config["patchnceLoss"]["patch_num"])

        self.lambda_gan = config["lambda_gan"]
        self.lambda_nce = config["lambda_nce"]

    def generator_out(self, img_rgb, for_discriminator=False):
        if for_discriminator:
            fake = self.generator(img_rgb, for_discriminator)
            return fake
        else:
            fake, src_feats, trt_feats = self.generator(img_rgb, for_discriminator)
            return fake, src_feats, trt_feats

    def discriminator_out(self, img):
        prob_map = self.discriminator(img)
        return prob_map

    def PNCE_out(self, src_feats, trt_feats):
        src_feats_mlp = self.patchnce(src_feats)
        trt_feats_mlp = self.patchnce(trt_feats)
        return src_feats_mlp, trt_feats_mlp

    def forward_G(self, img_rgb):
        fake, src_feats, trt_feats = self.generator_out(img_rgb, for_discriminator=False)

        prob_map_fake = self.discriminator_out(fake)

        src_feats_mlp, trt_feats_mlp = self.PNCE_out(src_feats, trt_feats)

        loss_G = self.gan_loss.G_loss(prob_map_fake)
        loss_PNCE = self.patchnce_loss(src_feats_mlp, trt_feats_mlp)
        loss_G_tot = loss_G * self.lambda_gan + loss_PNCE * self.lambda_nce
        return loss_G_tot, loss_G, loss_PNCE, fake

    def forward_D(self, img_rgb, img_tir):
        fake = self.generator_out(img_rgb, for_discriminator=True)

        prob_map_fake = self.discriminator_out(fake.detach())
        prob_map_real = self.discriminator_out(img_tir)

        loss_D = self.gan_loss.D_loss(prob_map_real, prob_map_fake)
        return loss_D
