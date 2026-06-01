import os

import cv2
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset.dataset import RGB2TIRData
from spade_cut_gan import SpadeCUTGAN


def train(
    data_path,
    model_save_path,
    img_save_path=r"./results",
    config=r"./config.yaml",
    epochs=100,
    decay_start=10,
    batch_size=1,
    num_workers=4,
    lr_G=2e-4,
    lr_D=2e-4,
    device="cpu",
):

    dataset = RGB2TIRData(data_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = SpadeCUTGAN(config)
    model.train()

    opt_G = torch.optim.Adam(model.generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(
        model.discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999)
    )

    scheduler_G = LambdaLR(
        opt_G,
        lr_lambda=lambda epoch: 1.0
        - max(0, epoch - decay_start) / (epochs - decay_start),
    )
    scheduler_D = LambdaLR(
        opt_D,
        lr_lambda=lambda epoch: 1.0
        - max(0, epoch - decay_start) / (epochs - decay_start),
    )

    for epoch in range(1, epochs + 1):

        dataset.shuffle_ds()

        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

        for rgb, tir in pbar:
            rgb = rgb.permute(0, 3, 1, 2).to(torch.float32).to(device) / 255 * 2 - 1
            tir = tir.unsqueeze(1).to(torch.float32).to(device) / 255 * 2 - 1

            opt_D.zero_grad()
            loss_D = model.forward_D(rgb, tir)
            loss_D.backward()
            scheduler_D.step()

            opt_G.zero_grad()
            loss_G_tot, loss_G, loss_PNCE, fake = model.forward_G(rgb)
            loss_G_tot.backward()
            scheduler_G.step()

            pbar.set_postfix(
                {"D": loss_D.item(), "G": loss_G.item(), "NCE": loss_PNCE.item()}
            )
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model.generator.state_dict(), f"checkpoints/epoch_{epoch}_G.pth")
        torch.save(model.discriminator.state_dict(), f"checkpoints/epoch_{epoch}_D.pth")

        os.makedirs(img_save_path, exist_ok=True)
        cv2.imwrite(
            img_save_path + rf"fake_{epoch}.jpg",
            ((fake[0].squeeze(0) + 1) / 2 * 255).to(torch.uint8),
        )
        rgb = ((rgb[0].permute(1, 2, 0) + 1) / 2 * 255).to(torch.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            img_save_path + rf"rgb_{epoch}.jpg",
            rgb,
        )
        cv2.imwrite(
            img_save_path + rf"tir_{epoch}.jpg",
            ((tir.squeeze(0, 1) + 1) / 2 * 255).to(torch.uint8),
        )


if __name__ == "__main__":

    model = SpadeCUTGAN(r"./config.yaml")

    aaa = (torch.rand(1, 3, 640, 512, dtype=torch.float32) - 0.5) * 2
    bbb = (torch.rand(1, 1, 640, 512, dtype=torch.float32) - 0.5) * 2

    loss_D = model.forward_D(aaa, bbb)

    loss_G_tot, loss_G, loss_PNCE, fm_loss, fake = model.forward_G(aaa, bbb)

    a = 9

    train(
        data_path=r"/mnt/storage/datasets",
        model_save_path=r"",
        img_save_path=r"./results/",
        config=r"./config.yaml",
        epochs=100,
        decay_start=10,
        batch_size=1,
        num_workers=8,
        lr_G=2e-4,
        lr_D=2e-4,
        device="cuda:0",
    )
