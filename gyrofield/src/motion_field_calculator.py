import numpy as np
from src.gyro_processor import GyroProcessor


class MotionFieldCalculator:


    def __init__(self, K, width, height, num_blocks=16):

        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.width = width
        self.height = height
        self.num_blocks = num_blocks

        # 预计算像素网格（齐次坐标）
        self._init_pixel_grid()

    def _init_pixel_grid(self):
        """初始化像素坐标网格"""
        u = np.arange(self.width)
        v = np.arange(self.height)
        uu, vv = np.meshgrid(u, v)

        # 齐次坐标 (H, W, 3)
        self.pixel_grid = np.stack([
            uu,
            vv,
            np.ones_like(uu)
        ], axis=-1).astype(np.float64)

    def compute_homography(self, R):

        return self.K @ R @ self.K_inv

    def compute_motion_field_with_timestamps(
            self,
            gyro_ns,
            gyro_w,
            ts_start,
            ts_end,
            readout_time=None
    ):

        frame_dt = (ts_end - ts_start) * 1e-9  # 秒

        if readout_time is None:
            row_dt = frame_dt / self.height
        else:
            row_dt = readout_time / self.height

        # 计算每行的时间戳
        row_timestamps_ns = np.array([
            ts_start + int(row * row_dt * 1e9)
            for row in range(self.height)
        ])

        # 计算每行的旋转矩阵
        rotations = GyroProcessor.integrate_gyro_to_rotations(
            gyro_ns, gyro_w, row_timestamps_ns
        )

        # 计算运动场
        motion_field = np.zeros((self.height, self.width, 2), dtype=np.float32)

        for row in range(self.height):
            R_row = rotations[row]

            # 计算单应性矩阵
            H = self.compute_homography(R_row)

            # 获取该行的像素坐标
            pixels = self.pixel_grid[row]  # (W, 3)

            # 应用单应性变换: p' = H * p
            transformed = (H @ pixels.T).T  # (W, 3)

            # 归一化齐次坐标
            transformed[:, 0] /= transformed[:, 2]
            transformed[:, 1] /= transformed[:, 2]

            # 计算位移: Δ = p' - p
            motion_field[row, :, 0] = transformed[:, 0] - pixels[:, 0]  # dx
            motion_field[row, :, 1] = transformed[:, 1] - pixels[:, 1]  # dy

        return motion_field



