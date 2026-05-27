import torch.nn as nn


class PatchNCE(nn.Module):
    def __init__(self, mlp_in=[256, 256, 256], mlp_out=512):
        super().__init__()

        self.mlps = nn.ModuleList()

        for in_ch in mlp_in:
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(in_ch, mlp_out),
                    nn.SiLU(),
                    nn.Linear(mlp_out, mlp_out),
                )
            )

    def forward(self, features):
        output = []
        for feat, mlp in zip(features, self.mlps):
            B, C, _, _ = feat.shape
            feat = feat.reshape(B, C, -1).transpose(1, 2)
            output.append(mlp(feat))
        return output
