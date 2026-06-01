import torch.nn as nn


class FeatureMatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fm_loss = nn.L1Loss()

    def forward(self, feats_real, feats_fake):
        fm_loss = 0
        for feat_real, feat_fake in zip(feats_real, feats_fake):
            fm_loss += self.fm_loss(feat_real, feat_fake)
        return fm_loss / len(feats_real)
