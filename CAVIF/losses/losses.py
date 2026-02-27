"""
Loss Functions for GFSNet Training
Implements self-supervised losses: photometric, SSIM, temporal smoothness, and acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1 loss variant)"""

    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        """
        Args:
            x, y: tensors to compare
        Returns:
            loss: scalar
        """
        diff = x - y
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)

class SSIM(nn.Module):
    """Structural Similarity Index"""

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma=1.5):
        """Create Gaussian kernel"""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        """Create 2D Gaussian window"""
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        """
        Args:
            img1, img2: (B, 3, H, W) images
        Returns:
            ssim_value: scalar or (B,) tensor
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """Compute SSIM"""
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class GFSNetLoss(nn.Module):
    """
    Complete loss function for GFSNet training

    Components:
    1. Multi-scale photometric + SSIM loss
    2. Temporal smoothness consistency
    3. Acceleration prior loss
    """

    def __init__(self, lambda_p=1.0, lambda_s=0.15, lambda_t=0.2, lambda_a=0.05,
                 alpha_photo=0.85, epsilon=1e-3):
        """
        Args:
            lambda_p: weight for photometric loss
            lambda_s: weight for SSIM loss
            lambda_t: weight for temporal smoothness
            lambda_a: weight for acceleration prior
            alpha_photo: weight for photometric in combined photo+SSIM loss
            epsilon: Charbonnier epsilon
        """
        super(GFSNetLoss, self).__init__()

        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.lambda_t = lambda_t
        self.lambda_a = lambda_a
        self.alpha_photo = alpha_photo

        self.charbonnier = CharbonnierLoss(epsilon)
        self.ssim = SSIM()

    def warp_frame(self, frame, flow):
        """
        Warp frame using flow field

        Args:
            frame: (B, 3, H, W) - RGB frame
            flow: (B, 2, H, W) - flow field (pixel domain)

        Returns:
            warped_frame: (B, 3, H, W)
        """
        B, C, H, W = frame.size()

        # Create base grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=frame.device, dtype=torch.float32),
            torch.arange(W, device=frame.device, dtype=torch.float32),
            indexing='ij'
        )

        # Add flow to grid
        x_new = x_grid.unsqueeze(0).repeat(B, 1, 1) + flow[:, 0, :, :]
        y_new = y_grid.unsqueeze(0).repeat(B, 1, 1) + flow[:, 1, :, :]

        # Normalize to [-1, 1]
        x_new = 2.0 * x_new / (W - 1) - 1.0
        y_new = 2.0 * y_new / (H - 1) - 1.0

        # Stack to form grid (B, H, W, 2)
        grid = torch.stack([x_new, y_new], dim=-1)

        # Warp using bilinear interpolation
        warped_frame = F.grid_sample(
            frame, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped_frame

    def compute_dynamic_mask(self, residual, gradient, percentile=0.8):
        """
        Compute dynamic mask to filter out low-texture and high-residual regions

        Args:
            residual: (B, 3, H, W) - photometric residual |I_warped - I_target|
            gradient: (B, 1, H, W) - gradient magnitude
            percentile: threshold percentile for masking

        Returns:
            mask: (B, 1, H, W) - binary mask (1=valid, 0=invalid)
        """
        # Compute residual magnitude
        residual_mag = torch.sqrt(torch.sum(residual ** 2, dim=1, keepdim=True) + 1e-8)

        # Threshold based on percentile
        threshold = torch.quantile(residual_mag.flatten(1), percentile, dim=1, keepdim=True)
        threshold = threshold.view(-1, 1, 1, 1)

        # Create mask: valid if residual < threshold AND gradient > min_gradient
        mask = (residual_mag < threshold).float() * (gradient > 0.01).float()

        return mask

    def multi_scale_photo_ssim_loss(self, frame_i, frame_j, Si, gradient):
        """
        Multi-scale photometric + SSIM loss (Equation 9-10)

        Args:
            frame_i: (B, 3, H, W) - center frame
            frame_j: (B, 3, H, W) - adjacent frame
            Si: (B, 2, H, W) - predicted stabilization flow
            gradient: (B, 1, H, W) - gradient magnitude for masking

        Returns:
            loss: scalar
        """
        total_loss = 0.0
        scales = [1, 2, 4]  # Three scales: 1/1, 1/2, 1/4

        for scale in scales:
            if scale > 1:
                # Downsample frames and flow
                frame_i_scaled = F.avg_pool2d(frame_i, scale)
                frame_j_scaled = F.avg_pool2d(frame_j, scale)
                Si_scaled = F.avg_pool2d(Si, scale) / scale  # Scale flow accordingly
                gradient_scaled = F.avg_pool2d(gradient, scale)
            else:
                frame_i_scaled = frame_i
                frame_j_scaled = frame_j
                Si_scaled = Si
                gradient_scaled = gradient

            # Warp frame_j to frame_i using Si
            frame_j_warped = self.warp_frame(frame_j_scaled, Si_scaled)

            # Compute residual
            residual = torch.abs(frame_j_warped - frame_i_scaled)

            # Compute dynamic mask (Equation 10)
            mask = self.compute_dynamic_mask(residual, gradient_scaled)

            # Photometric loss (Charbonnier)
            photo_loss = self.charbonnier(
                frame_j_warped * mask,
                frame_i_scaled * mask
            )

            # SSIM loss
            ssim_loss = 1.0 - self.ssim(frame_j_warped * mask, frame_i_scaled * mask)

            # Combined loss (Equation 9)
            scale_loss = self.alpha_photo * photo_loss + (1 - self.alpha_photo) * ssim_loss
            total_loss += scale_loss / len(scales)

        return total_loss

    def temporal_smoothness_loss(self, Si_current, Si_next):

        flow_diff = Si_next - Si_current

        # 计算 L1 损失并归一化
        loss = torch.mean(torch.abs(flow_diff))


        H, W = Si_current.shape[2], Si_current.shape[3]
        loss = loss / torch.sqrt(torch.tensor(H * W, dtype=torch.float32, device=loss.device))

        return loss

    def acceleration_prior_loss(self, Si_prev, Si_current, Si_next):

        velocity_1 = Si_current - Si_prev
        velocity_2 = Si_next - Si_current

        # Second-order difference (acceleration)
        acceleration = velocity_2 - velocity_1


        loss = torch.mean(torch.abs(acceleration))


        H, W = Si_current.shape[2], Si_current.shape[3]
        loss = loss / torch.sqrt(torch.tensor(H * W, dtype=torch.float32, device=loss.device))

        return loss

    def forward(self, outputs, batch_data, video_names):

        Si = outputs['Si']  # (B, 2, H, W)
        B = Si.shape[0]

        # Get center frame and adjacent frames (need to load from disk)
        # For simplicity, assume batch_data contains loaded frames
        frame_i = batch_data['frame_center']  # (B, 3, H, W)
        frame_j = batch_data['frame_next']    # (B, 3, H, W)
        gradient = outputs['E'][:, 1:2, :, :]  # (B, 1, H, W) - gradient magnitude from E

        # 1. Multi-scale photometric + SSIM loss
        loss_photo_ssim = self.multi_scale_photo_ssim_loss(
            frame_i, frame_j, Si, gradient
        )

        # 2. Temporal smoothness loss (if next frame prediction available)
        if 'Si_next' in outputs:
            loss_temporal = self.temporal_smoothness_loss(Si, outputs['Si_next'])
        else:
            loss_temporal = torch.tensor(0.0, device=Si.device)

        # 3. Acceleration prior loss (if prev and next predictions available)
        if 'Si_prev' in outputs and 'Si_next' in outputs:
            loss_acceleration = self.acceleration_prior_loss(
                outputs['Si_prev'], Si, outputs['Si_next']
            )
        else:
            loss_acceleration = torch.tensor(0.0, device=Si.device)



        # Total loss (Equation 13)
        total_loss = (
            self.lambda_p * loss_photo_ssim +
            self.lambda_t * loss_temporal +
            self.lambda_a * loss_acceleration
        )

        # Return loss dictionary
        loss_dict = {
            'total': total_loss,
            'photo_ssim': loss_photo_ssim,
            'temporal': loss_temporal,
            'acceleration': loss_acceleration
        }

        return loss_dict