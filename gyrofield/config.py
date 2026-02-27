import numpy as np

# 相机内参矩阵
K = np.array([
    [0, 0,0 ],   # fx, 0, cx
    [0, 0, 0],   # 0, fy, cy
    [0, 0, 1.0]
])

# IMU到相机坐标系的旋转矩阵
R_CAM_IMU = np.array([
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1]
])

# 图像尺寸
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
# IMU 时间相对相机时间的偏移（单位：秒）
# 正值表示 IMU 时间戳比相机时间戳晚
IMU_TIME_OFFSET = 0

# Rolling Shutter 行读出时间（单位：秒）
READOUT_TIME = None

# 陀螺仪采样率 (Hz)，用于验证数据质量
GYRO_SAMPLE_RATE = 0
# 行块数量
NUM_BLOCKS = 16

# 数据根目录
DATA_ROOT = "new_data"

#
SCENES = {

}
# 输出目录
OUTPUT_DIR = "output"
