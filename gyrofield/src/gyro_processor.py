import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class GyroProcessor:


    @staticmethod
    def integrate_gyro(gyro_ns, gyro_w, start_ns, end_ns):

        R = np.eye(3)

        for i in range(len(gyro_ns) - 1):
            t0 = gyro_ns[i]
            t1 = gyro_ns[i + 1]

            if t1 <= start_ns or t0 >= end_ns:
                continue


            t_start = max(t0, start_ns)
            t_end = min(t1, end_ns)
            dt = (t_end - t_start) * 1e-9  # 转换为秒

            if dt <= 0:
                continue


            w = gyro_w[i]

            angle = np.linalg.norm(w) * dt

            if angle < 1e-10:
                continue

            axis = w / np.linalg.norm(w)


            dR = Rotation.from_rotvec(axis * angle).as_matrix()


            R = dR @ R

        return R

    @staticmethod
    def integrate_gyro_to_rotations(gyro_ns, gyro_w, row_timestamps_ns):

        num_rows = len(row_timestamps_ns)
        rotations = []

        # 以第一行为参考，计算每行相对于第一行的旋转
        t_ref = row_timestamps_ns[0]

        for row in range(num_rows):
            t_row = row_timestamps_ns[row]

            if t_row == t_ref:
                R = np.eye(3)
            else:
                R = GyroProcessor.integrate_gyro(
                    gyro_ns, gyro_w, t_ref, t_row
                )

            rotations.append(R)

        return rotations

    @staticmethod
    def smooth_rotations(rotations, num_output):

        if len(rotations) < 2:
            return [rotations[0]] * num_output

        # 转换为 Rotation 对象
        rot_objects = Rotation.from_matrix(np.array(rotations))

        # 创建插值器
        times_in = np.linspace(0, 1, len(rotations))
        times_out = np.linspace(0, 1, num_output)

        slerp = Slerp(times_in, rot_objects)
        smooth_rot = slerp(times_out)

        return [r.as_matrix() for r in smooth_rot]

    @staticmethod
    def compute_cumulative_rotation(gyro_samples):

        import warnings
        warnings.warn(
            "compute_cumulative_rotation() 已废弃",
            DeprecationWarning
        )

        num_samples = len(gyro_samples)
        rotations = [np.eye(3)]

        for i in range(1, num_samples):
            w = gyro_samples[i]
            angle = np.linalg.norm(w)

            if angle < 1e-10:
                rotations.append(rotations[-1].copy())
                continue

            axis = w / angle
            dR = Rotation.from_rotvec(axis * angle).as_matrix()
            R_new = dR @ rotations[-1]
            rotations.append(R_new)

        return rotations

    @staticmethod
    def slerp_rotations(rotations, num_output):

        return GyroProcessor.smooth_rotations(rotations, num_output)