
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAlignment(nn.Module):

    def __init__(self):
        super(TemporalAlignment, self).__init__()

    def warp_features(self, features, displacement_field):

        B, C, H, W = features.shape

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=features.device, dtype=torch.float32),
            torch.arange(W, device=features.device, dtype=torch.float32),
            indexing='ij'
        )

        # Add displacement field to grid
        x_new = x_grid.unsqueeze(0).repeat(B, 1, 1) + displacement_field[:, 0, :, :]
        y_new = y_grid.unsqueeze(0).repeat(B, 1, 1) + displacement_field[:, 1, :, :]

        # Normalize to [-1, 1] for grid_sample
        x_new = 2.0 * x_new / (W - 1) - 1.0
        y_new = 2.0 * y_new / (H - 1) - 1.0

        # Stack to form grid (B, H, W, 2)
        grid = torch.stack([x_new, y_new], dim=-1)

        # Warp using bilinear interpolation with border padding
        warped_features = F.grid_sample(
            features, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped_features

    def compute_future_displacement(self, flow_1, flow_2):

        if flow_2 is None:
            return flow_1

        # pos4 (k=2): compose flow_1 and flow_2
        warped_flow_2 = self.warp_features(flow_2, flow_1)  # warp flow_2 by flow_1
        composed_flow = flow_1 + warped_flow_2

        return composed_flow

    def forward(self, features_window, batch_data, flow_scale):

        B, C, W, H, W_dim = features_window.shape
        aligned_features = []

        # Process each position in temporal window
        for pos_idx in range(W):
            # Get features for current position
            current_features = features_window[:, :, pos_idx, :, :]  # (B, 16, H, W)

            # 1. Center position (pos2): already in frame_i coordinate system
            if pos_idx == 2:
                aligned_features.append(current_features)
                continue

            # 2. Past positions (pos0, pos1): use reverse flow
            elif pos_idx in [0, 1]:
                # Get optical flow from batch data (already normalized)
                flow = batch_data['optical_flows'][:, pos_idx, :, :, :]  # (B, 2, H, W)

                # Convert back to original scale and reverse direction
                displacement = -flow * flow_scale  # (B, 2, H, W)

                # Warp features to frame_i coordinate system
                warped_features = self.warp_features(current_features, displacement)
                aligned_features.append(warped_features)

            # 3. Future positions (pos3, pos4): use flow composition
            elif pos_idx == 3:  # frame_{i+1} → frame_i
                # Get flow from pos2 (i→i+1)
                flow_i_to_i1 = batch_data['optical_flows'][:, 2, :, :, :]  # (B, 2, H, W)

                # Convert back to original scale
                displacement = flow_i_to_i1 * flow_scale  # (B, 2, H, W)

                # Warp features to frame_i coordinate system
                warped_features = self.warp_features(current_features, displacement)
                aligned_features.append(warped_features)

            elif pos_idx == 4:  # frame_{i+2} → frame_i
                # Get flows for composition: pos2 (i→i+1) and pos3 (i+1→i+2)
                flow_i_to_i1 = batch_data['optical_flows'][:, 2, :, :, :]  # (B, 2, H, W)
                flow_i1_to_i2 = batch_data['optical_flows'][:, 3, :, :, :]  # (B, 2, H, W)

                # Convert back to original scale
                flow_i_to_i1 = flow_i_to_i1 * flow_scale
                flow_i1_to_i2 = flow_i1_to_i2 * flow_scale

                # Compute composed displacement field
                displacement = self.compute_future_displacement(flow_i_to_i1, flow_i1_to_i2)

                # Warp features to frame_i coordinate system
                warped_features = self.warp_features(current_features, displacement)
                aligned_features.append(warped_features)

        # Stack aligned features (B, 16, W, H, W)
        aligned_window = torch.stack(aligned_features, dim=2)

        return aligned_window