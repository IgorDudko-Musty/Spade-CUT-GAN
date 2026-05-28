import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07, patch_num=512):
        super().__init__()
        self.temperature = temperature
        self.patch_num = patch_num
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, src_feats, trt_feats, patch_num=512):
        total_loss = 0.0
        n_layers = len(src_feats)

        for src, trt in zip(src_feats, trt_feats):

            patch_inds = torch.randint(
                low=0,
                high=src.shape[1],
                size=(patch_num,),
                dtype=torch.int64,
                device=src.device,
            )

            src = src[:, patch_inds, :].squeeze(0)
            trt = trt[:, patch_inds, :].squeeze(0)

            src = F.normalize(src, p=2, dim=-1)
            trt = F.normalize(trt, p=2, dim=-1)

            logits = trt @ src.transpose(0, 1)
            logits /= self.temperature

            targets = torch.arange(patch_num, dtype=torch.int64, device=src.device)

            loss = self.cross_entropy_loss(logits, targets)
            total_loss += loss
        return total_loss / n_layers
