import torch.nn as nn
import torch.nn.functional as F
import yaml

from model_blocks.discriminator import Discriminator
from model_blocks.generator import Generator
from model_blocks.patchnce_head import PatchNCE
from model_losses.feat_match_loss import FeatureMatchLoss
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
        self.discriminator = nn.ModuleList(
            [
                Discriminator(
                    in_channels=config["discriminator"]["in_channels"],
                    out_channels=config["discriminator"]["out_channels"],
                    kernel_size=config["discriminator"]["kernel_size"],
                    strides=config["discriminator"]["strides"],
                    prob=config["discriminator"]["prob"],
                )
                for _ in range(config["discriminator"]["amount"])
            ]
        )
        self.patchnce = PatchNCE(
            mlp_in=config["patchnce"]["mlp_in"], mlp_out=config["patchnce"]["mlp_out"]
        )
        self.gan_loss = GANLoss()
        self.patchnce_loss = PatchNCELoss(
            config["patchnceLoss"]["temperature"], config["patchnceLoss"]["patch_num"]
        )
        self.fm_loss = FeatureMatchLoss()

        self.lambda_gan = config["lambda_gan"]
        self.lambda_nce = config["lambda_nce"]
        self.lambda_fm = config["lambda_fm"]

    def generator_out(self, img_rgb, for_discriminator=False):
        if for_discriminator:
            fake = self.generator(img_rgb, for_discriminator)
            return fake
        else:
            fake, src_feats, trt_feats = self.generator(img_rgb, for_discriminator)
            return fake, src_feats, trt_feats

    def discriminator_out(self, img):
        prob_map_list, feats_list = [], []
        for i, D in enumerate(self.discriminator):
            scale = 2**i
            prob_map, feats = D(
                F.interpolate(
                    img,
                    scale_factor=1 / scale,
                    mode="bilinear",
                    align_corners=False,
                ),
            )
            prob_map_list.append(prob_map)
            feats_list.append(feats)
        return prob_map_list, feats_list

    def PNCE_out(self, src_feats, trt_feats):
        src_feats_mlp = self.patchnce(src_feats)
        trt_feats_mlp = self.patchnce(trt_feats)
        return src_feats_mlp, trt_feats_mlp

    def forward_G(self, img_rgb, img_tir):
        fake, src_feats, trt_feats = self.generator_out(
            img_rgb, for_discriminator=False
        )

        prob_map_fake, feats_fake_list = self.discriminator_out(fake)
        prob_map_ir, feats_real_list = self.discriminator_out(img_tir.detach())

        src_feats_mlp, trt_feats_mlp = self.PNCE_out(src_feats, trt_feats)

        loss_G = 0
        for prob_map in prob_map_fake:
            loss_G += self.gan_loss.G_loss(prob_map)
        loss_G /= len(prob_map_fake)

        loss_PNCE = self.patchnce_loss(src_feats_mlp, trt_feats_mlp)

        loss_FM = 0
        for feats_fake, feats_ir in zip(feats_fake_list, feats_real_list):
            loss_FM += self.fm_loss(feats_ir, feats_fake)
        loss_FM /= len(feats_fake_list)

        loss_G_tot = (
            loss_G * self.lambda_gan
            + loss_PNCE * self.lambda_nce
            + loss_FM * self.lambda_fm
        )
        return loss_G_tot, loss_G, loss_PNCE, loss_FM, fake

    def forward_D(self, img_rgb, img_tir):
        fake = self.generator_out(img_rgb, for_discriminator=True)

        prob_map_fake, _ = self.discriminator_out(fake.detach())
        prob_map_real, _ = self.discriminator_out(img_tir)

        loss_D = 0.0
        for real, fake in zip(prob_map_real, prob_map_fake):
            loss_D += self.gan_loss.D_loss(real, fake)
        return loss_D / len(prob_map_real)
