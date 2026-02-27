

import torch
import torch.nn as nn
from .feature_constructor import FeatureConstructor
from .temporal_alignment import TemporalAlignment
from .gated_fusion import GatedFusion
from .unet_backbone import UNetBackbone
from .output_head import OutputHead


class GFSNet(nn.Module):


    def __init__(self, frames_dir, original_height=1080, original_width=1920,
                 crop_top=4, crop_bottom=4, base_channels=64, delta_max=10.0):

        super(GFSNet, self).__init__()

        self.frames_dir = frames_dir
        self.height = original_height - crop_top - crop_bottom  # 1072
        self.width = original_width  # 1920


        self.feature_constructor = FeatureConstructor(
            frames_dir=frames_dir,
            original_height=original_height,
            original_width=original_width,
            crop_top=crop_top,
            crop_bottom=crop_bottom
        )

        self.temporal_alignment = TemporalAlignment()


        self.gated_fusion = GatedFusion(
            temporal_window=5,
            num_channels=16
        )


        self.unet_backbone = UNetBackbone(
            input_channels=16,
            base_channels=base_channels
        )

        self.output_head = OutputHead(
            in_channels=base_channels,
            delta_max=delta_max
        )

    def forward(self, batch_data, video_names):

        B, T, _, H, W = batch_data['optical_flows'].shape


        features_window, flow_scale = self.feature_constructor(
            batch_data, video_names
        )  # (B, 16, 5, H, W)


        aligned_features = self.temporal_alignment(
            features_window, batch_data, flow_scale
        )  # (B, C_align, H, W)


        fused_features, E = self.gated_fusion(
            aligned_features, batch_data
        )  # fused_features: (B, 16, H, W), E: (B, 5, H, W)


        backbone_features = self.unet_backbone(fused_features)  # dict with multi-scale features


        optical_flow_center = batch_data['optical_flows'][:, 2, :, :, :]
        gyro_flow_center = batch_data['gyro_flows'][:, 2, :, :, :]

        output_center = self.output_head(
            backbone_features,
            E,
            optical_flow_center,
            gyro_flow_center,
            flow_scale
        )


        Si_prev = batch_data['optical_flows'][:, 1, :, :, :] * flow_scale


        Si_next = batch_data['optical_flows'][:, 3, :, :, :] * flow_scale

        # ========== 7. 组装输出 ==========
        outputs = {
            'Si': output_center['Si'],
            'Si_prev': Si_prev,  #
            'Si_next': Si_next,  #
            'alpha': output_center['alpha'],
            'delta': output_center['delta'],
            'alpha_raw': output_center['alpha_raw'],
            'alpha_0': output_center['alpha_0'],
            'E': E
        }

        return outputs

    def inference(self, batch_data, video_names):

        with torch.no_grad():
            outputs = self.forward(batch_data, video_names)
            return outputs['Si']