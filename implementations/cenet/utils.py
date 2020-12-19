import torch
import torch.nn as nn


class dice_loss(torch.nn.Module):
    def init(self):
        super(diceloss, self).init()
    def forward(self,pred, target):
        smooth = 1.

        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + smooth) / (
                    pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

        return loss.mean()