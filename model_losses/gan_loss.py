import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def D_loss(self, D_real, D_fake):
        loss_real = self.loss(D_real, torch.ones_like(D_real))
        loss_fake = self.loss(D_fake.detach(), torch.zeros_like(D_fake))
        return 0.5 * (loss_real + loss_fake)

    def G_loss(self, D_fake):
        return self.loss(D_fake, torch.ones_like(D_fake))
