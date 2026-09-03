from .functional import revgrad
import torch
from torch import nn

class GradientReversal(nn.Module):
    """Gradient reversal layer with a settable, schedulable alpha
    (Brion et al. 2021, Comput Biol Med 131:104269, Eq. 3-4)."""

    def __init__(self, alpha=0.):
        super().__init__()
        self.alpha = float(alpha)

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    def forward(self, x):
        alpha = torch.tensor(self.alpha, device=x.device, dtype=x.dtype)
        return revgrad(x, alpha)