import os
import numpy as np
from tqdm import tqdm
import config
from src.data_loader import GyroDataLoader, FrameTimestampLoader
from src.motion_field_calculator import MotionFieldCalculator
from src.visualizer import MotionFieldVisualizer

def process_scene(scene_name):
    """处理单个场景"""
    print(f"\n{'=' * 60}")
    print(f"处理场景: {scene_name}")
    print(f"{'=' * 60}")

    # 构建路径
    scene_path = os.path.join(config.DATA_ROOT, scene_name)

    # 查找文件
    gyro_file = None
    frames_file = None

    for file in os.listdir(scene_path):
        if "gyro" in file and file.endswith(".csv"):
            gyro_file = os.path.join(scene_path, file)
        elif "frames" in file and file.endswith(".csv"):
            frames_file = os.path.join(scene_path, file)

    if not gyro_file or not frames_file:
        print(f"✗ 缺少必要文件: {scene_name}")
        return

    # 创建输出目录
    output_motion_dir = os.path.join(config.OUTPUT_DIR, "motion_fields", scene_name)
    output_vis_dir = os.path.join(config.OUTPUT_DIR, "visualizations", scene_name)
    os.makedirs(output_motion_dir, exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)

    # 加载数据
    print("加载数据...")
    gyro_loader = GyroDataLoader(gyro_file, config.R_CAM_IMU)
    frame_loader = FrameTimestampLoader(frames_file)

    # 初始化运动场计算器
    calculator = MotionFieldCalculator(
        config.K,
        config.IMAGE_WIDTH,
        config.IMAGE_HEIGHT,
        config.NUM_BLOCKS
    )

    # ============ 关键改动：计算时间偏移 ============
    # IMU 时间 = 相机时间 + 偏移
    offset_ns = int(config.IMU_TIME_OFFSET * 1e9)
    print(f"  IMU-Camera 时间偏移: {config.IMU_TIME_OFFSET * 1000:.1f} ms")

    # 处理每一对相邻帧
    num_frames = frame_loader.get_num_frames()
    print(f"总帧数: {num_frames}")

    processed_count = 0
    skipped_count = 0

    for i in tqdm(range(num_frames - 1), desc="计算运动场"):
        # 获取相机时间戳
        ts_start_cam = frame_loader.get_timestamp(i)
        ts_end_cam = frame_loader.get_timestamp(i + 1)

        # ============ 关键改动：转换到 IMU 时间轴 ============
        ts_start = ts_start_cam + offset_ns
        ts_end = ts_end_cam + offset_ns

        # 检查时间戳有效性
        if not gyro_loader.is_valid_ns(ts_start) or not gyro_loader.is_valid_ns(ts_end):
            skipped_count += 1
            continue


        gyro_ns, gyro_w = gyro_loader.get_gyro_segment(ts_start, ts_end)


        motion_field = calculator.compute_motion_field_with_timestamps(
            gyro_ns,
            gyro_w,
            ts_start,
            ts_end,
            readout_time=config.READOUT_TIME
        )

        # 保存运动场
        motion_file = os.path.join(
            output_motion_dir,
            f"motion_field_{i:04d}_to_{i + 1:04d}.npy"
        )
        np.save(motion_file, motion_field)

        # 可视化 (每10帧保存一次)
        if i % 10 == 0:
            vis_file = os.path.join(
                output_vis_dir,
                f"motion_field_{i:04d}_to_{i + 1:04d}.png"
            )
            MotionFieldVisualizer.visualize_motion_field(
                motion_field, vis_file, scale=5
            )

        processed_count += 1



def main():
    """主函数"""
    print("=" * 60)
    print("陀螺仪运动场计算)")

    # 处理所有场景
    for scene in config.SCENES:
        try:
            process_scene(scene)
        except Exception as e:
            print(f"✗ 场景 {scene} 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("所有场景处理完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()