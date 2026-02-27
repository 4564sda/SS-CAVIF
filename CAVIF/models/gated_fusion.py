

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):


    def __init__(self, temporal_window=5, num_channels=16):

        super(GatedFusion, self).__init__()
        self.temporal_window = temporal_window  # W=5
        self.num_channels = num_channels        # 16 channels


        self.dwconv3d = nn.Conv3d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(temporal_window, 3, 3),
            groups=num_channels,
            padding=(0, 1, 1)  # No padding on temporal dimension
        )

        # GN + SiLU
        self.group_norm = nn.GroupNorm(num_groups=16, num_channels=num_channels)
        self.silu = nn.SiLU()

        # DWConv2D (3x3)
        self.dwconv2d = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            groups=num_channels,
            padding=1
        )

        # Conv1x1 (16→16)
        self.conv1x1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1
        )


        self.gate_conv = nn.Conv2d(
            in_channels=5,  # E has 5 channels
            out_channels=1,
            kernel_size=3,
            padding=1
        )


        self.channel_mlp = nn.Sequential(
            nn.Linear(2 * num_channels, num_channels),  # (16 + 16) → 16
            nn.SiLU(),
            nn.Linear(num_channels, num_channels),      # 16 → 16
            nn.Sigmoid()
        )

        self.channel_mlp[-2].weight.data.normal_(0, 0.01)
        self.channel_mlp[-2].bias.data.fill_(0.5)  # 初始偏置0.5使sigmoid输出接近0.73

    def compute_gate线索(self, features_window, batch_data):

        B, C, W, H, W_dim = features_window.shape


        residuals = features_window[:, 11:15, 2, :, :]  # (B, 4, H, W) - 中心位置的RGB残差
        r_min = torch.min(residuals, dim=1, keepdim=True)[0]  # (B, 1, H, W)

        grad_mag = features_window[:, 10:11, 2, :, :]  # (B, 1, H, W) - 中心位置的梯度幅值

        diff_flow = features_window[:, 4:6, 2, :, :]  # (B, 2, H, W) - 中心位置的差场
        d_mag = torch.sqrt(diff_flow[:, 0:1, :, :]**2 + diff_flow[:, 1:2, :, :]**2)  # (B, 1, H, W)

        optical_flows = batch_data['optical_flows'][:, 2, :, :, :]  # (B, 2, H, W) - 中心位置
        gyro_flows = batch_data['gyro_flows'][:, 2, :, :, :]        # (B, 2, H, W) - 中心位置

        optical_mag = torch.sqrt(optical_flows[:, 0:1, :, :]**2 + optical_flows[:, 1:2, :, :]**2)
        gyro_mag = torch.sqrt(gyro_flows[:, 0:1, :, :]**2 + gyro_flows[:, 1:2, :, :]**2)
        flow_diff = optical_mag - gyro_mag  # (B, 1, H, W)

        conf = batch_data['confidences'][:, 2, :, :, :]  # (B, 1, H, W) - 中心位置
        conf_inv = 1.0 - conf  # (B, 1, H, W)


        E = torch.cat([r_min, grad_mag, d_mag, flow_diff, conf_inv], dim=1)

        return E

    def forward(self, aligned_window, batch_data):

        B, C, W, H, W_dim = aligned_window.shape

        x = self.dwconv3d(aligned_window)


        x = x.squeeze(2)  # (B, 16, H, W)
        x = self.group_norm(x)
        x = self.silu(x)

        x = self.dwconv2d(x)  # (B, 16, H, W)
        Xmix = self.conv1x1(x)  # (B, 16, H, W)


        Xcenter = aligned_window[:, :, 2, :, :]  # (B, 16, H, W)


        E = self.compute_gate线索(aligned_window, batch_data)  # (B, 5, H, W)


        gspatial = torch.sigmoid(self.gate_conv(E))  # (B, 1, H, W)


        gap_x = F.adaptive_avg_pool2d(Xmix, (1, 1)).view(B, -1)  # (B, 16)


        gap_e = F.adaptive_avg_pool2d(E, (1, 1)).view(B, -1)  # (B, 5)
        gap_e = F.pad(gap_e, (0, C - 5))  # (B, 16) - 填充到16通道

        gap_concat = torch.cat([gap_x, gap_e], dim=1)  # (B, 32)
        gchannel = self.channel_mlp(gap_concat).view(B, C, 1, 1)  # (B, 16, 1, 1)


        x_center_part = gspatial * Xcenter
        x_mix_part = (1 - gspatial) * (gchannel * Xmix)
        Xgate = x_center_part + x_mix_part

        return Xgate,E