"""
U-Net Backbone with Skip Connection Gating
Implements Stem, 4-level Encoder with ASPP, and 4-level Decoder with skip gates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with rates 1, 2, 4, 8"""

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU()
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU()
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU()
        )

        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=8, dilation=8, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU()
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        # Global average pooling branch
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)

        # Concatenate all branches
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        # Final 1x1 convolution
        x = self.conv1x1(x)

        return x

class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm and SiLU activation"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)

class DownBlock(nn.Module):
    """Encoder downsampling block with MaxPool and ConvBlock"""

    def __init__(self, in_channels, out_channels, with_aspp=False):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        self.with_aspp = with_aspp
        if with_aspp:
            self.aspp = ASPP(out_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        if self.with_aspp:
            x = self.aspp(x)
        return x

class UpBlock(nn.Module):
    """Decoder upsampling block with skip connection gating"""

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # Upsampling followed by 1x1 conv to reduce channels
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        # Convolutional blocks after skip connection fusion
        self.conv = nn.Sequential(
            ConvBlock(2 * out_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

        # Skip connection gate (sigmoid gating)
        self.skip_gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, skip_x):
        # Upsample and reduce channels
        x = self.upsample(x)

        # Apply skip connection gate
        skip_gate = self.skip_gate(skip_x)
        gated_skip = skip_gate * skip_x

        # Concatenate and process
        x = torch.cat([x, gated_skip], dim=1)
        x = self.conv(x)

        return x

class UNetBackbone(nn.Module):
    """U-Net Backbone with Stem, Encoder, Decoder and Skip Connection Gating"""

    def __init__(self, input_channels=16, base_channels=64):
        super(UNetBackbone, self).__init__()
        self.base_channels = base_channels  # C=64

        # --------------------------
        # Stem
        # --------------------------
        self.stem = nn.Sequential(
            ConvBlock(input_channels, base_channels),  # Conv3x3(16→64)
            ConvBlock(base_channels, base_channels)    # Conv3x3(64→64)
        )

        # --------------------------
        # Encoder (4 levels)
        # --------------------------
        self.down1 = DownBlock(base_channels, 2 * base_channels)  # C→2C
        self.down2 = DownBlock(2 * base_channels, 4 * base_channels)  # 2C→4C
        self.down3 = DownBlock(4 * base_channels, 8 * base_channels, with_aspp=True)  # 4C→8C + ASPP
        self.down4 = DownBlock(8 * base_channels, 16 * base_channels)  # 8C→16C

        # --------------------------
        # Decoder (4 levels with skip gates)
        # --------------------------
        self.up4 = UpBlock(16 * base_channels, 8 * base_channels)  # 16C→8C
        self.up3 = UpBlock(8 * base_channels, 4 * base_channels)   # 8C→4C
        self.up2 = UpBlock(4 * base_channels, 2 * base_channels)   # 4C→2C
        self.up1 = UpBlock(2 * base_channels, base_channels)       # 2C→C

        # --------------------------
        # Multi-scale output heads
        # --------------------------
        self.out_1_4 = nn.Conv2d(4 * base_channels, base_channels, kernel_size=1)  # 1/4 scale
        self.out_1_2 = nn.Conv2d(2 * base_channels, base_channels, kernel_size=1)  # 1/2 scale
        self.out_1_1 = nn.Conv2d(base_channels, base_channels, kernel_size=1)      # 1/1 scale

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 16, H, W) - Gated fusion output

        Returns:
            multi_scale_features: dict containing {1/4, 1/2, 1/1} scale features
        """
        # --------------------------
        # Stem
        # --------------------------
        x = self.stem(x)  # (B, C, H, W)

        # --------------------------
        # Encoder (collect multi-scale features)
        # --------------------------
        e1 = self.down1(x)  # (B, 2C, H/2, W/2) - E1(1/2)
        e2 = self.down2(e1)  # (B, 4C, H/4, W/4) - E2(1/4)
        e3 = self.down3(e2)  # (B, 8C, H/8, W/8) - E3(1/8)
        e4 = self.down4(e3)  # (B, 16C, H/16, W/16) - E4(1/16)

        # --------------------------
        # Decoder (with skip gates)
        # --------------------------
        d4 = self.up4(e4, e3)  # (B, 8C, H/8, W/8)
        d3 = self.up3(d4, e2)  # (B, 4C, H/4, W/4)
        d2 = self.up2(d3, e1)  # (B, 2C, H/2, W/2)
        d1 = self.up1(d2, x)   # (B, C, H, W)

        # --------------------------
        # Multi-scale outputs
        # --------------------------
        out_1_4 = self.out_1_4(d3)  # 1/4 scale
        out_1_2 = self.out_1_2(d2)  # 1/2 scale
        out_1_1 = self.out_1_1(d1)  # 1/1 scale

        return {
            '1/4': out_1_4,
            '1/2': out_1_2,
            '1/1': out_1_1
        }