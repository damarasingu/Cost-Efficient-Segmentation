import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """
    Lightweight depthwise separable convolution block:
    3x3 depthwise conv -> 1x1 pointwise conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pw(self.dw(x))))

class FastSegNet(nn.Module):
    """
    Lightweight Fully Convolutional Network with encoder-decoder and skip connections.
    Designed for high accuracy salient object segmentation under strict CPU compute budgets (<50 ms).
    
    Architecture:
    - Input: (B, 3, 256, 256)
    - Initial stride-2 conv -> (B, 16, 128, 128)
    - Encoder stage 1: 24 channels (128x128) -> MaxPool -> 64x64
    - Encoder stage 2: 36 channels (64x64) -> MaxPool -> 32x32
    - Encoder stage 3: 64 channels (32x32) -> MaxPool -> 16x16
    - Bottleneck: 96 channels (16x16)
    - Decoder stage 3: 48 channels (32x32) with skip connection from Encoder 3
    - Decoder stage 2: 24 channels (64x64) with skip connection from Encoder 2
    - Decoder stage 1: 16 channels (128x128) with skip connection from Initial stage
    - Final stage: Bilinear upsample to 256x256 -> Conv -> 1-channel logit output
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Initial stride-2 downsampling to 128x128
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.enc1 = DepthwiseSeparableConv(16, 24)       # 128x128
        self.pool1 = nn.MaxPool2d(2, 2)                  # 64x64
        
        self.enc2 = DepthwiseSeparableConv(24, 36)       # 64x64
        self.pool2 = nn.MaxPool2d(2, 2)                  # 32x32
        
        self.enc3 = DepthwiseSeparableConv(36, 64)       # 32x32
        self.pool3 = nn.MaxPool2d(2, 2)                  # 16x16
        
        self.bottleneck = DepthwiseSeparableConv(64, 96) # 16x16
        
        # Decoder with skip connections
        self.dec3 = DepthwiseSeparableConv(96 + 64, 48)  # 32x32
        self.dec2 = DepthwiseSeparableConv(48 + 36, 24)  # 64x64
        self.dec1 = DepthwiseSeparableConv(24 + 16, 16)  # 128x128
        
        # Final refinement to 256x256
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, 3, 256, 256)
        x0 = self.init_conv(x)  # (B, 16, 128, 128)
        e1 = self.enc1(x0)      # (B, 24, 128, 128)
        
        p1 = self.pool1(e1)     # (B, 24, 64, 64)
        e2 = self.enc2(p1)      # (B, 36, 64, 64)
        
        p2 = self.pool2(e2)     # (B, 36, 32, 32)
        e3 = self.enc3(p2)      # (B, 64, 32, 32)
        
        p3 = self.pool3(e3)     # (B, 64, 16, 16)
        b = self.bottleneck(p3) # (B, 96, 16, 16)
        
        d3 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False) # 32x32
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False) # 64x64
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False) # 128x128
        d1 = self.dec1(torch.cat([d1, x0], dim=1))
        
        d0 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False) # 256x256
        out = self.final_conv(d0)
        return out

class DiceBCELoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Soft Dice Loss for robust segmentation.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, targets):
        bce_loss = self.bce(pred_logits, targets)
        
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * targets).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return bce_loss + dice_loss.mean()
