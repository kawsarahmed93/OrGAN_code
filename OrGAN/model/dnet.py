import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ============================================================
# Deterministic reflection-padded convolution
# ============================================================

class ReflectConv2d(nn.Conv2d):
    """nn.Conv2d(padding_mode="reflect") with a deterministic backward.

    PyTorch's native reflection_pad2d_backward_cuda scatters gradients with
    atomicAdd, so it has no deterministic implementation: under
    torch.use_deterministic_algorithms(True, warn_only=True) it warns once
    per epoch and silently falls back to the nondeterministic kernel. That
    was the sole remaining source of run-to-run variance in train-D and
    train-C_D (train-C has no discriminator and was already reproducible),
    and it is why setting a seed alone never made those two configs repeat.

    Expressing the same padding as slice + flip + cat keeps the arithmetic
    identical - the forward output is bit-identical to padding_mode="reflect"
    - while routing the backward through cat/flip, whose gradients are plain
    deterministic slicing. Gradients agree with the native kernel to ~2e-7
    (float32 summation order), i.e. this is the same architecture, not a
    substitute for it; switching to padding_mode="zeros" would have been a
    real architecture change and is not what this does.

    Subclassing nn.Conv2d (rather than prepending a padding module) keeps the
    parameter names - and therefore state_dict keys - unchanged, so existing
    dnet.ckpt files load without modification.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, **kwargs):
        kwargs.pop("padding_mode", None)
        # The conv itself does no padding; we pad explicitly in forward().
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=0, bias=bias, **kwargs)
        self.reflect_pad = padding

    def forward(self, x):
        p = self.reflect_pad
        if p:
            x = torch.cat([x[..., 1:p + 1].flip(-1), x, x[..., -p - 1:-1].flip(-1)], dim=-1)
            x = torch.cat([x[..., 1:p + 1, :].flip(-2), x, x[..., -p - 1:-1, :].flip(-2)], dim=-2)
        return self._conv_forward(x, self.weight, self.bias)


# ============================================================
# PatchGAN with Spectral Normalization
# ============================================================

class PatchGAN(nn.Module):

    def __init__(self, in_channels=1, features=[32, 64, 128, 256]):
        super().__init__()

        layers = []

        # First layer: no normalization
        layers.append(
            nn.Sequential(
                spectral_norm(
                    ReflectConv2d(
                        in_channels,
                        features[0],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        in_feat = features[0]

        for i, feature in enumerate(features[1:]):

            # All except the last feature block use stride 2
            # Last feature block uses stride 1
            stride = 2 if i < len(features) - 2 else 1

            layers.append(
                nn.Sequential(
                    spectral_norm(
                        ReflectConv2d(
                            in_feat,
                            feature,
                            kernel_size=4,
                            stride=stride,
                            padding=1,
                            bias=False,
                        )
                    ),
                    nn.InstanceNorm2d(feature, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

            in_feat = feature

        # Final prediction layer
        layers.append(
            spectral_norm(
                ReflectConv2d(
                    in_feat,
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                )
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================
# Two-Scale PatchGAN
# D3 removed
# D1 has larger receptive field
# ============================================================

class CNN(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()

        # LARGE patch discriminator
        # Approx RF for 512x512 input: ~142x142
        self.D1 = PatchGAN(
            in_channels=in_channels,
            features=[32, 64, 128, 256, 512]
        )

        # MID patch discriminator
        # Applied after 2x downsampling
        self.D2 = PatchGAN(
            in_channels=in_channels,
            features=[32, 64, 128]
        )

        self.downsample = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            count_include_pad=False
        )

    def forward(self, x):

        out1 = self.D1(x)

        x2 = self.downsample(x)
        out2 = self.D2(x2)

        return [out1, out2]


# ============================================================
# GAN Loss
# ============================================================

class GANLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, target_is_real):

        total_loss = 0

        for pred in preds:

            if target_is_real:
                target = torch.ones_like(pred)
            else:
                target = torch.zeros_like(pred)

            total_loss += self.loss(pred, target)

        return total_loss / len(preds)


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    x = torch.randn(1, 1, 512, 512)

    D = CNN(in_channels=1)

    outputs = D(x)

    print("Number of outputs:", len(outputs))

    for i, out in enumerate(outputs):
        print(f"Scale {i+1} output shape:", out.shape)