
import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputHead(nn.Module):


    def __init__(self, in_channels=64, delta_max=10.0):
        super(OutputHead, self).__init__()
        self.delta_max = delta_max  # δ_max hyperparameter


        self.alpha_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),  # Conv1x1 to 1 channel
            nn.Sigmoid()  # α ∈ (0,1)
        )


        self.delta_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 2, kernel_size=1),  # Conv1x1 to 2 channels (x,y)
            nn.Tanh()  # Tanh → [-1,1]
        )


        self.prior_gate = nn.Sequential(
            nn.Conv2d(5, 1, kernel_size=1),  # E has 5 channels
            nn.Sigmoid()  # α_0 ∈ (0,1)
        )


        self.scale_fusion = nn.Conv2d(3 * in_channels, in_channels, kernel_size=1)

    def forward(self, multi_scale_features, E, optical_flow, gyro_flow, flow_scale):

        feat_1_4 = multi_scale_features['1/4']  # (B, C, H/4, W/4)
        feat_1_2 = multi_scale_features['1/2']  # (B, C, H/2, W/2)
        feat_1_1 = multi_scale_features['1/1']  # (B, C, H, W)


        B, C, H, W = feat_1_1.shape
        feat_1_4 = F.interpolate(feat_1_4, size=(H, W), mode='bilinear', align_corners=True)
        feat_1_2 = F.interpolate(feat_1_2, size=(H, W), mode='bilinear', align_corners=True)


        fused_feat = torch.cat([feat_1_4, feat_1_2, feat_1_1], dim=1)  # (B, 3C, H, W)
        fused_feat = self.scale_fusion(fused_feat)  # (B, C, H, W)


        alpha_raw = self.alpha_head(fused_feat)  # (B, 1, H, W)


        alpha_0 = self.prior_gate(E)  # (B, 1, H, W)

        alpha_mix = 0.7 * alpha_raw + 0.3 * alpha_0

        alpha = torch.clamp(alpha_mix, min=0.1, max=0.9)

        delta_normalized = self.delta_head(fused_feat)


        delta = delta_normalized * (self.delta_max / flow_scale)


        u_out_normalized = alpha * optical_flow + (1 - alpha) * gyro_flow + delta


        Si = u_out_normalized * flow_scale

        return {
            'Si': Si,                # 最终稳定化流场 (像素域)
            'alpha': alpha,          # 融合权重图
            'delta': delta,          # 残差修正项
            'alpha_raw': alpha_raw,  # 原始融合权重 (用于分析)
            'alpha_0': alpha_0       # 先验门值 (用于分析)
        }