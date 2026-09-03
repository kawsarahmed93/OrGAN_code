import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/_functional.py
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/focal.py
# https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
# https://arxiv.org/pdf/1708.02002.pdf   
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        logpt = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-logpt)
        focal_term = (1.0 - pt).pow(self.gamma)
        loss = focal_term * logpt
        if self.alpha != 0:
            loss *= self.alpha * target + (1 - self.alpha) * (1 - target)
        return loss.mean()
