import numpy as np
import csv
from scipy.interpolate import interp1d


class GyroDataLoader:


    def __init__(self, gyro_file, R_cam_imu):

        self.R_cam_imu = R_cam_imu
        self._load_gyro_data(gyro_file)

    def _load_gyro_data(self, gyro_file):

        gyro_reader = csv.reader(open(gyro_file), delimiter=",")
        gyro_data = np.array(list(gyro_reader))


        self.gyro_ns = np.array(gyro_data[:, 3], dtype=np.int64)  # 时间戳 (纳秒)
        gyro_w = np.array(gyro_data[:, :3], dtype=np.float64)  # [wx, wy, wz]

        self.gyro_cam = np.zeros_like(gyro_w)
        for i in range(len(self.gyro_ns)):
            self.gyro_cam[i] = np.matmul(self.R_cam_imu, gyro_w[i])

        self.t_min = self.gyro_ns[0]
        self.t_max = self.gyro_ns[-1]

        self.gyro_interp = interp1d(
            self.gyro_ns,
            self.gyro_cam,
            axis=0,
            fill_value="extrapolate",
            kind="linear"
        )

        print(f"  陀螺仪数据: {len(self.gyro_ns)} 个采样点")
        print(f"  时间范围: {self.t_min / 1e9:.3f}s - {self.t_max / 1e9:.3f}s")

    def is_valid_ns(self, timestamp_ns):

        return self.t_min <= timestamp_ns <= self.t_max

    def get_gyro_at(self, timestamp_ns):

        return self.gyro_interp(timestamp_ns)

    def get_gyro_segment(self, start_ns, end_ns):

        mask = (self.gyro_ns >= start_ns) & (self.gyro_ns <= end_ns)

        timestamps = self.gyro_ns[mask].copy()
        gyro_data = self.gyro_cam[mask].copy()


        if len(timestamps) == 0:
            timestamps = np.array([start_ns, end_ns])
            gyro_data = np.array([
                self.gyro_interp(start_ns),
                self.gyro_interp(end_ns)
            ])
        else:

            if timestamps[0] > start_ns:
                timestamps = np.insert(timestamps, 0, start_ns)
                start_gyro = self.gyro_interp(start_ns).reshape(1, 3)
                gyro_data = np.vstack([start_gyro, gyro_data])

            if timestamps[-1] < end_ns:
                timestamps = np.append(timestamps, end_ns)
                end_gyro = self.gyro_interp(end_ns).reshape(1, 3)
                gyro_data = np.vstack([gyro_data, end_gyro])

        return timestamps, gyro_data

    def get_gyro_between_frames(self, start_ns, end_ns, num_samples):

        timestamps = np.linspace(start_ns, end_ns, num_samples)
        gyro_data = np.array([self.get_gyro_at(ts) for ts in timestamps])
        return gyro_data


class FrameTimestampLoader:

    def __init__(self, frames_file):

        frames_reader = csv.reader(open(frames_file), delimiter=",")
        frames_data = list(frames_reader)

        self.timestamps = np.array(
            [int(item) for sublist in frames_data for item in sublist],
            dtype=np.int64
        )

        print(f"  帧时间戳: {len(self.timestamps)} 帧")

    def get_timestamp(self, frame_idx):

        return self.timestamps[frame_idx]

    def get_num_frames(self):

        return len(self.timestamps)

    def get_frame_interval(self, frame_idx):

        if frame_idx >= len(self.timestamps) - 1:
            return None
        dt_ns = self.timestamps[frame_idx + 1] - self.timestamps[frame_idx]
        return dt_ns * 1e-9