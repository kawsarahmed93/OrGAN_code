import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import autocast

# # ============================================================
# # Classifier (unchanged)
# # ============================================================
# class Classifier(nn.Module):
#     def __init__(self, num_classes, in_features=1024):
#         super().__init__()

#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(in_features, num_classes)

#         nn.init.normal_(self.fc.weight, 0, 0.01)
#         nn.init.zeros_(self.fc.bias)

#     def forward(self, features):
#         x = self.pool(features)
#         x = x.flatten(1)
#         return self.fc(x)


# # ============================================================
# # Full Model
# # ============================================================
# class DenseNet121(nn.Module):
#     def __init__(self, num_classes: int, in_channels: int = 1):
#         super().__init__()

#         backbone = torchvision.models.densenet121(
#             weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
#         )

#         self.encoder = backbone.features

#         old_conv = self.encoder.conv0
#         new_conv = nn.Conv2d(
#             in_channels,
#             old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             padding=old_conv.padding,
#             bias=False
#         )

#         new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
#         self.encoder.conv0 = new_conv
        
#         self.classifier = Classifier(num_classes)

#     def forward(self, x):
#         features = self.encoder(x)
#         features = F.relu(features, inplace=False)
#         logits = self.classifier(features)

#         return {
#             "logits": logits
#         }

# ============================================================
# Full Model
# ============================================================
class DenseNet121(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()

        backbone = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        )

        f = backbone.features

        # --- Fix first conv for grayscale ---
        old_conv = f.conv0
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        f.conv0 = new_conv

        # --- Split DenseNet into blocks ---
        self.stem = nn.Sequential(
            f.conv0,
            f.norm0,
            f.relu0,
            f.pool0
        )

        self.denseblock1 = f.denseblock1
        self.transition1 = f.transition1

        self.denseblock2 = f.denseblock2
        self.transition2 = f.transition2

        self.denseblock3 = f.denseblock3
        self.transition3 = f.transition3

        self.denseblock4 = f.denseblock4
        self.norm5 = f.norm5
        
        # self.classifier = Classifier(num_classes)
        self.classifier = FNClassifier(num_classes)

    def forward(self, x):
        # --- stem ---
        x = self.stem(x)

        # --- block 1 ---
        x = self.denseblock1(x)
        x = self.transition1(x)

        # --- block 2 (28×28) ---
        x = self.denseblock2(x)
        x = self.transition2(x)

        # --- block 3 (14×14) ---
        x = self.denseblock3(x)
        x = self.transition3(x)

        # --- final ---
        x = self.denseblock4(x)
        x = self.norm5(x)
        x = F.relu(x, inplace=False)
        
        logits = self.classifier(x)

        return {
            "logits": logits
        }

# # ============================================================
# # Classifier (unchanged)
# # ============================================================
class Classifier(nn.Module):
    def __init__(self, num_classes, in_features=1024):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features, num_classes)

        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features):
        x = self.pool(features)
        x = x.flatten(1)
        return self.fc(x)


class FNClassifier(nn.Module):
    def __init__(self, num_classes, in_features=1024):
        super().__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2*in_features, num_classes)

        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features):
        x1 = self.avg(features)
        x2 = self.mx(features)
        x = torch.cat([x1,x2], dim=1)
        x = x.flatten(1)
        return self.fc(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        backbone = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        )

        f = backbone.features

        # --- Fix first conv for grayscale ---
        old_conv = f.conv0
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        f.conv0 = new_conv

        # --- Split DenseNet into blocks ---
        self.stem = nn.Sequential(
            f.conv0,
            f.norm0,
            f.relu0,
            f.pool0
        )

        self.denseblock1 = f.denseblock1
        self.transition1 = f.transition1

        self.denseblock2 = f.denseblock2
        self.transition2 = f.transition2

        self.denseblock3 = f.denseblock3
        self.transition3 = f.transition3

        self.denseblock4 = f.denseblock4
        self.norm5 = f.norm5

        # --- Gated fusion modules ---
        self.fusion1 = Attention(128)
        self.fusion2 = Attention(256)
        self.fusion3 = Attention(512)
        self.fusion4 = Attention(1024)

    def forward(self, x, l):
        # --- stem ---
        x = self.stem(x)

        # --- block 1 ---
        x = self.denseblock1(x) 
        x = self.transition1(x)
        x = self.fusion1(x, l)  

        # --- block 2 (28×28) ---
        x = self.denseblock2(x)  
        x = self.transition2(x)
        x = self.fusion2(x, l) 

        # --- block 3 (14×14) ---
        x = self.denseblock3(x)
        x = self.transition3(x)
        x = self.fusion3(x, l)   

        # --- final ---
        x = self.denseblock4(x)
        x = self.norm5(x)
        x = F.relu(x, inplace=False)
        x = self.fusion4(x, l) 

        return x

class Attention(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2*C, C, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 1),
            nn.Sigmoid()
        )
        
        self.norm_fx = nn.GroupNorm(32, C)
        self.norm_fused = nn.GroupNorm(32, C)

    def forward(self, fx, fl):
        fx=self.norm_fx(fx)

        fl = F.interpolate(fl, size=fx.shape[-2:], mode='bilinear', align_corners=False)
            
        
        MF = fx * fl

        G = self.gate(torch.cat([fx, MF], dim=1))
        # print(G.mean())
        # print(G.std())
        
        # fused = fx * (1 + G * (2*fl - 1)) # (1-G)*fx + 2*G*MF
        fused = fx * (1 + G * (3*fl - 1)) # (1-G)*fx + 3*G*MF
        # fused = fx * (1 + 2*G * (fl - 1)) # (1-2*G)*fx + 2*G*MF
        # fused = fx * (1 + 2* G *fl) # fx + 2*G*MF

        fused=self.norm_fused(fused)

        return fused

class FusionNet(nn.Module):
    def __init__(self, num_classes: int):

        super().__init__()

        in_features = 1024
        in_channels = 1

       
        self.encoder = Encoder(in_channels)

        self.classifier = FNClassifier(num_classes, in_features=1024)
        # self.classifier = Classifier(num_classes, in_features=1024)
        
    def forward(self, x, l):

        fused = self.encoder(x, l)

        logits = self.classifier(fused)

        return {
            "logits": logits
        }

if __name__ == "__main__":
    imgs = torch.zeros((2, 3, 224, 224))
    model = DenseNet121(num_classes=15)
    out = model(imgs)
    print(out["logits"].shape)  # torch.Size([2, 20])
