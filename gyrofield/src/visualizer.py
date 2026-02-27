import numpy as np
import matplotlib.pyplot as plt
import cv2


class MotionFieldVisualizer:


    @staticmethod
    def visualize_motion_field(motion_field, save_path=None, scale=5, show=False):

        H, W = motion_field.shape[:2]
        dx = motion_field[..., 0]
        dy = motion_field[..., 1]

        # 计算幅度和方向
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx)

        # 创建HSV图像
        hsv = np.zeros((H, W, 3), dtype=np.uint8)
        hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(magnitude * scale * 10, 0, 255).astype(np.uint8)

        # 转换为RGB
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # 创建图像
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. 光流颜色图
        axes[0].imshow(flow_rgb)
        axes[0].set_title('Motion Field (HSV)')
        axes[0].axis('off')

        # 2. 幅度图
        im1 = axes[1].imshow(magnitude, cmap='jet')
        axes[1].set_title(f'Magnitude\nmean={magnitude.mean():.3f}, max={magnitude.max():.3f}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # 3. 箭头图 (稀疏采样)
        step = max(H // 30, W // 30, 1)
        y_coords, x_coords = np.mgrid[0:H:step, 0:W:step]
        u = dx[::step, ::step]
        v = dy[::step, ::step]

        axes[2].imshow(np.zeros((H, W)), cmap='gray', vmin=0, vmax=1)
        axes[2].quiver(x_coords, y_coords, u, v,
                       magnitude[::step, ::step],
                       cmap='jet', scale=50, width=0.002)
        axes[2].set_title('Motion Vectors')
        axes[2].axis('off')
        axes[2].set_xlim(0, W)
        axes[2].set_ylim(H, 0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def visualize_comparison(frame_t, frame_t1, frame_t_est, motion_field,
                             save_path=None, show=False):

        frame_t_rgb = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
        frame_t1_rgb = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2RGB)
        frame_t_est_rgb = cv2.cvtColor(frame_t_est, cv2.COLOR_BGR2RGB)

        # 计算误差
        diff = cv2.absdiff(frame_t, frame_t_est)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        mse = np.mean((frame_t.astype(float) - frame_t_est.astype(float)) ** 2)
        psnr = cv2.PSNR(frame_t, frame_t_est)

        # 运动场幅度
        magnitude = np.sqrt(motion_field[..., 0] ** 2 + motion_field[..., 1] ** 2)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(frame_t_rgb)
        axes[0, 0].set_title('Frame t (Ground Truth)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(frame_t1_rgb)
        axes[0, 1].set_title('Frame t+1 (Source)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(frame_t_est_rgb)
        axes[0, 2].set_title('Frame t (Reconstructed)')
        axes[0, 2].axis('off')

        im1 = axes[1, 0].imshow(diff_gray, cmap='magma', vmin=0, vmax=50)
        axes[1, 0].set_title(f'Error Map\nMSE={mse:.2f}, PSNR={psnr:.2f}dB')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

        im2 = axes[1, 1].imshow(magnitude, cmap='jet')
        axes[1, 1].set_title(f'Motion Magnitude\nmean={magnitude.mean():.3f}')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

        # 叠加对比
        blend = cv2.addWeighted(frame_t, 0.5, frame_t_est, 0.5, 0)
        blend_rgb = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
        axes[1, 2].imshow(blend_rgb)
        axes[1, 2].set_title('Overlay (GT + Reconstructed)')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return {'mse': mse, 'psnr': psnr}