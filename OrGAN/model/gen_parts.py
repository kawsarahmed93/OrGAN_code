""" Parts of the OrGAN model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gradient_reversal import GradientReversal

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DomainClassifier(nn.Module):
    """Domain classifier fed by the gradient reversal layer.

    Follows the sizing convention in Brion et al. 2021 (Comput Biol Med
    131:104269): a small strided-conv stack that halves/doubles channels per
    level down to a fixed-size dense head, rather than flattening the
    full-resolution tapped feature map directly into one huge FC layer.

    Widened/deepened (~5M params, vs ~460K originally) to test whether a
    classifier with more capacity gives a more informative reversed gradient
    - a weak classifier can be fooled by superficial cues that don't track
    the actual domain gap FID/KID pick up (closs plateaued near ln(2) even
    at GRL_LAMBDA_MAX=0.001, suggesting it wasn't working hard to tell the
    domains apart in the first place). Still kept far below the generator's
    own parameter count and below the ~105M single-FC-layer original this
    project moved away from, since an overpowered classifier risks the
    opposite failure mode: saturated (near 0/1) predictions vanish the NLL
    gradient before GRL even reverses it, which would starve rather than
    strengthen the adversarial signal into the encoder. The extra Dropout
    below is a cheap guard against exactly that overconfidence failure mode.
    """

    def __init__(self, out_channels, in_channels=4):
        super().__init__()
        self.grl = GradientReversal(alpha=0.)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channels),
            nn.LogSoftmax(dim=1),
        )

    def set_alpha(self, alpha):
        self.grl.set_alpha(alpha)

    def forward(self, x):
        x = self.grl(x)
        x = self.features(x)
        return self.classifier(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1)),
          nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

# """ Parts of the OrGAN model """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .gradient_reversal import GradientReversal


# def get_num_groups(num_channels, max_groups=8):
#     """
#     Choose a valid GroupNorm group number.
#     For example:
#         C=64  -> 8 groups
#         C=32  -> 8 groups
#         C=4   -> 4 groups
#     """
#     num_groups = min(max_groups, num_channels)

#     while num_channels % num_groups != 0:
#         num_groups -= 1

#     return num_groups


# class DoubleConv(nn.Module):
#     """(convolution => [GN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()

#         if not mid_channels:
#             mid_channels = out_channels

#         g1 = get_num_groups(mid_channels)
#         g2 = get_num_groups(out_channels)

#         self.double_conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 mid_channels,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False
#             ),
#             nn.GroupNorm(g1, mid_channels),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(
#                 mid_channels,
#                 out_channels,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False
#             ),
#             nn.GroupNorm(g2, out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class DomainClassifier(nn.Module):
#     """Original domain classifier kept unchanged"""

#     def __init__(self, out_channels):
#         super().__init__()

#         self.double_conv = nn.Sequential(
#             GradientReversal(alpha=1.),
#             nn.Linear(512 * 512 * 4, 100),
#             nn.ReLU(inplace=True),
#             nn.Linear(100, out_channels),
#             nn.LogSoftmax(dim=1)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         if bilinear:
#             self.up = nn.Upsample(
#                 scale_factor=2,
#                 mode='bilinear',
#                 align_corners=True
#             )
#             self.conv = DoubleConv(
#                 in_channels,
#                 out_channels,
#                 in_channels // 2
#             )
#         else:
#             self.up = nn.ConvTranspose2d(
#                 in_channels,
#                 in_channels // 2,
#                 kernel_size=2,
#                 stride=2
#             )
#             self.conv = DoubleConv(
#                 in_channels,
#                 out_channels
#             )

#     def forward(self, x1, x2):
#         x1 = self.up(x1)

#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(
#             x1,
#             [
#                 diffX // 2,
#                 diffX - diffX // 2,
#                 diffY // 2,
#                 diffY - diffY // 2
#             ]
#         )

#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(1, 1),
#                 stride=(1, 1)
#             ),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.conv(x)