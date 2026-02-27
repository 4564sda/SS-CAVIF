
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

class FeatureConstructor(nn.Module):


    def __init__(self, frames_dir, original_height=1080, original_width=1920,
                 crop_top=4, crop_bottom=4):

        super(FeatureConstructor, self).__init__()
        self.frames_dir = frames_dir
        self.original_height = original_height
        self.original_width = original_width
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.height = original_height - crop_top - crop_bottom  # 1072
        self.width = original_width

        # Sobel kernels for gradient computation
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def compute_flow_normalization_scale(self, optical_flows, gyro_flows):
        B, T, _, H, W = optical_flows.shape

        optical_mag = torch.sqrt(optical_flows[:, :, 0] ** 2 + optical_flows[:, :, 1] ** 2)  # [B, T, H, W]
        gyro_mag = torch.sqrt(gyro_flows[:, :, 0] ** 2 + gyro_flows[:, :, 1] ** 2)  # [B, T, H, W]

        for dim_index, (dim_opt, dim_gyro) in enumerate(zip(optical_mag.shape, gyro_mag.shape)):
            if dim_opt != dim_gyro:
                print(f"[DEBUG] Sape match at  {dim_index}: optical_mag={dim_opt}, gyro_mag={dim_gyro}")
        try:
            sum_mag = optical_mag + gyro_mag  # [B, T, H, W]
        except RuntimeError as e:
            raise

        total_elements = B * T * H * W
        k = int(0.95 * total_elements)

        # 分批处理以避免内存问题
        if total_elements > 10000000:
            scales = []
            for b in range(B):
                batch_mag = sum_mag[b].flatten()  # [T*H*W]
                batch_k = int(0.95 * len(batch_mag))
                batch_scale = torch.kthvalue(batch_mag, batch_k)[0]
                scales.append(batch_scale)
            scale = torch.stack(scales).mean()
        else:
            # 使用topk来近似分位数（更快）
            flat_mag = sum_mag.flatten()
            k = max(1, min(k, len(flat_mag) - 1))
            scale = torch.kthvalue(flat_mag, k)[0]

        # 避免除零
        scale = torch.clamp(scale, min=1e-6)

        return scale

    def load_frame(self, video_name, frame_idx, split='testF'):

        import os

        parent_dir = os.path.dirname(self.frames_dir)

        frames_dir = os.path.join(parent_dir, split)
        # ======================================================

        frame_path = os.path.join(
            frames_dir,
            video_name,
            'frames',
            f'frame_{frame_idx:04d}.jpg'
        )

        frame = Image.open(frame_path).convert('RGB')
        frame = np.array(frame, dtype=np.float32) / 255.0

        # Crop
        frame = frame[self.crop_top:self.original_height - self.crop_bottom, :, :]

        # Convert to tensor (3, H, W)
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        return frame

    def rgb_to_grayscale(self, rgb):

        gray = 0.299 * rgb[:, 0:1, :, :] + \
               0.587 * rgb[:, 1:2, :, :] + \
               0.114 * rgb[:, 2:3, :, :]
        return gray

    def compute_gradient_magnitude(self, gray):

        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)

        # Compute magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        return grad_mag

    def warp_image(self, image, flow):
        """
        Warp image using optical flow

        Args:
            image: (B, C, H, W) image to warp
            flow: (B, 2, H, W) optical flow (dx, dy)

        Returns:
            warped: (B, C, H, W) warped image
        """
        B, C, H, W = image.shape

        # Create base grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=image.device, dtype=torch.float32),
            torch.arange(W, device=image.device, dtype=torch.float32),
            indexing='ij'
        )

        # Add flow to grid
        x_new = x_grid.unsqueeze(0).repeat(B, 1, 1) + flow[:, 0, :, :]
        y_new = y_grid.unsqueeze(0).repeat(B, 1, 1) + flow[:, 1, :, :]

        # Normalize to [-1, 1] for grid_sample
        x_new = 2.0 * x_new / (W - 1) - 1.0
        y_new = 2.0 * y_new / (H - 1) - 1.0

        # Stack to form grid (B, H, W, 2)
        grid = torch.stack([x_new, y_new], dim=-1)

        # Warp using bilinear interpolation
        warped = F.grid_sample(image, grid, mode='bilinear',
                              padding_mode='border', align_corners=True)

        return warped

    def construct_features_for_pair(self, frame_i, frame_i_plus_1,
                                   optical_flow, gyro_flow, confidence,
                                   flow_scale):
        """
        Construct 16-channel features for a single frame pair

        Args:
            frame_i: (B, 3, H, W) - first frame (center frame)
            frame_i_plus_1: (B, 3, H, W) - second frame
            optical_flow: (B, 2, H, W) - optical flow from i to i+1
            gyro_flow: (B, 2, H, W) - gyro flow from i to i+1
            confidence: (B, 1, H, W) - PDC confidence map
            flow_scale: scalar - normalization scale

        Returns:
            features: (B, 16, H, W) - 16-channel features
        """
        B = frame_i.shape[0]

        # Normalize flows
        optical_flow_norm = optical_flow / flow_scale  # (B, 2, H, W)
        gyro_flow_norm = gyro_flow / flow_scale        # (B, 2, H, W)

        # Channel 1-2: Normalized gyro flow
        ch_1_2 = gyro_flow_norm  # (B, 2, H, W)

        # Channel 3-4: Normalized optical flow
        ch_3_4 = optical_flow_norm  # (B, 2, H, W)

        # Channel 5-6: Difference field
        diff_flow = optical_flow_norm - gyro_flow_norm  # (B, 2, H, W)
        ch_5_6 = diff_flow

        # Channel 7: Magnitude of gyro flow
        gyro_mag = torch.sqrt(gyro_flow_norm[:, 0:1, :, :]**2 +
                             gyro_flow_norm[:, 1:2, :, :]**2 + 1e-8)  # (B, 1, H, W)
        ch_7 = gyro_mag

        # Channel 8: Magnitude of optical flow
        optical_mag = torch.sqrt(optical_flow_norm[:, 0:1, :, :]**2 +
                                optical_flow_norm[:, 1:2, :, :]**2 + 1e-8)  # (B, 1, H, W)
        ch_8 = optical_mag

        # Channel 9: Magnitude of difference field
        diff_mag = torch.sqrt(diff_flow[:, 0:1, :, :]**2 +
                             diff_flow[:, 1:2, :, :]**2 + 1e-8)  # (B, 1, H, W)
        ch_9 = diff_mag

        # Channel 10: Grayscale of frame_i
        gray_i = self.rgb_to_grayscale(frame_i)  # (B, 1, H, W)
        ch_10 = gray_i

        # Channel 11: Gradient magnitude of frame_i
        grad_mag_i = self.compute_gradient_magnitude(gray_i)  # (B, 1, H, W)
        ch_11 = grad_mag_i

        # Channel 12-15: Warp residuals
        # Warp frame_i to frame_i+1 using optical flow
        warped_i = self.warp_image(frame_i, optical_flow)  # (B, 3, H, W)

        # Compute residual (difference)
        residual = frame_i_plus_1 - warped_i  # (B, 3, H, W)
        ch_12_15 = residual  # Use RGB residual directly (3 channels)

        # We need 4 channels for 12-15, so add magnitude as 4th channel
        residual_mag = torch.sqrt(torch.sum(residual**2, dim=1, keepdim=True) + 1e-8)  # (B, 1, H, W)
        ch_12_15 = torch.cat([residual, residual_mag], dim=1)  # (B, 4, H, W)

        # Channel 16: PDC confidence
        ch_16 = confidence  # (B, 1, H, W)

        # Concatenate all channels
        features = torch.cat([
            ch_1_2,      # Channels 1-2
            ch_3_4,      # Channels 3-4
            ch_5_6,      # Channels 5-6
            ch_7,        # Channel 7
            ch_8,        # Channel 8
            ch_9,        # Channel 9
            ch_10,       # Channel 10
            ch_11,       # Channel 11
            ch_12_15,    # Channels 12-15
            ch_16        # Channel 16
        ], dim=1)  # (B, 16, H, W)

        return features

    def forward(self, batch_data, video_names):
        """
        Construct 16-channel features for all frame pairs in temporal window

        Args:
            batch_data: dict containing:
                - optical_flows: (B, W, 2, H, W) - W=5 frame pairs
                - gyro_flows: (B, W, 2, H, W)
                - confidences: (B, W, 1, H, W)
                - frame_pair_indices: list of B lists, each containing W tuples
                - split: str - 'train' or 'val' (新增字段)
            video_names: list of B video names

        Returns:
            features_window: (B, 16, W, H, W) - features for temporal window
            flow_scale: scalar - normalization scale (for later use)
        """

        optical_flows = batch_data['optical_flows']  # (B, W, 2, H, W)
        gyro_flows = batch_data['gyro_flows']  # (B, W, 2, H, W)
        confidences = batch_data['confidences']  # (B, W, 1, H, W)
        frame_pair_indices = batch_data['frame_pair_indices']

        split = batch_data.get('split', 'test')  #
        # ====================================================

        B, W, _, H, W_dim = optical_flows.shape

        # Compute flow normalization scale (95th percentile)
        flow_scale = self.compute_flow_normalization_scale(optical_flows, gyro_flows)

        # Construct features for each position in temporal window
        features_list = []
        split = batch_data.get('split', 'test')  # 默认 'train'
        for pos_idx in range(W):  # W=5 positions
            # Get flows and confidence for this position
            optical_flow = optical_flows[:, pos_idx, :, :, :]  # (B, 2, H, W)
            gyro_flow = gyro_flows[:, pos_idx, :, :, :]  # (B, 2, H, W)
            confidence = confidences[:, pos_idx, :, :, :]  # (B, 1, H, W)

            # Load frames for this position (batch processing)
            frames_i_list = []
            frames_i_plus_1_list = []

            for batch_idx in range(B):
                video_name = video_names[batch_idx]
                frame_idx_i, frame_idx_i_plus_1 = frame_pair_indices[batch_idx][pos_idx]


                frame_i = self.load_frame(video_name, frame_idx_i, split)
                frame_i_plus_1 = self.load_frame(video_name, frame_idx_i_plus_1, split)
                # ==========================================

                frames_i_list.append(frame_i)
                frames_i_plus_1_list.append(frame_i_plus_1)

            # Stack into batches
            frames_i = torch.stack(frames_i_list, dim=0).to(optical_flow.device)  # (B, 3, H, W)
            frames_i_plus_1 = torch.stack(frames_i_plus_1_list, dim=0).to(optical_flow.device)

            # Construct 16-channel features for this position
            features = self.construct_features_for_pair(
                frames_i, frames_i_plus_1,
                optical_flow, gyro_flow, confidence,
                flow_scale
            )  # (B, 16, H, W)

            features_list.append(features)

        # Stack features from all positions
        features_window = torch.stack(features_list, dim=2)  # (B, 16, W, H, W)

        return features_window, flow_scale