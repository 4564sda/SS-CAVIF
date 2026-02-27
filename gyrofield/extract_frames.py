# extract_frames.py
import os
import cv2
import config

def extract_frames_from_video(video_path, output_dir):
    """把一个 mp4 按原始帧率逐帧保存为图片"""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  视频: {os.path.basename(video_path)}, fps={fps:.2f}, 总帧数={frame_count}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按帧号保存，frame_0000.png 这种格式
        filename = f"frame_{idx:04d}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, frame)

        idx += 1

    cap.release()
    print(f"  已保存帧数: {idx}, 输出目录: {output_dir}")

def main():
    root_dir = config.DATA_ROOT
    print(f"遍历目录: {root_dir}")

    if not os.path.isdir(root_dir):
        print(f"目录不存在: {root_dir}")
        return


    for scene_name in sorted(os.listdir(root_dir)):
        scene_dir = os.path.join(root_dir, scene_name)
        if not os.path.isdir(scene_dir):
            continue

        print(f"\n=== 处理场景文件夹: {scene_dir} ===")

        # 找该文件夹下所有 mp4
        for fname in sorted(os.listdir(scene_dir)):
            if not fname.lower().endswith(".mp4"):
                continue

            video_path = os.path.join(scene_dir, fname)

            base = os.path.splitext(fname)[0]
            output_dir = os.path.join(scene_dir, base + "_frames")

            extract_frames_from_video(video_path, output_dir)

if __name__ == "__main__":
    main()