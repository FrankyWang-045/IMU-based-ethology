from data import Data
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np


class preprocess(Data):
    '''IMU数据预处理类，继承自Data类，包括预处理方法：Madgwick滤波、通道分割、设备安装反转识别、降采样'''
    def __init__(self, file_path, beta=0.1):
        super().__init__(file_path)
        self.beta = beta

        #预处理结果占位
        self.quaternions = None   # (N, 4)
        self.features = None      # (N, 9)        
    
    #预处理总过程
    def preprocess(self):

        self._estimate_orientation()
        self._compute_euler_angles()
        self._angle_to_trigonometric()
        self._combine_features()        
        #self._downsampling(factor=2)  # 降采样到50Hz
        return self

    #madgwick算法估计姿态四元数
    @staticmethod
    def _madgwick_step(q, g, a, dt, beta):
        """单步 Madgwick 更新。
        q: 当前四元数 [w, x, y, z]
        g: 角速度 (rad/s), shape (3,)
        a: 加速度, shape (3,)
        """
        # 加速度归一化（算法只关心方向，不关心大小）
        norm_a = np.linalg.norm(a)
        if norm_a == 0.0:
            return q  # 全零数据无法修正，直接返回
        ax_, ay_, az_ = a / norm_a
        qw, qx, qy, qz = q

        # 1) 陀螺仪积分：四元数变化率 = 0.5 * q ⊗ [0, g]
        q_dot_gyro = 0.5 * np.array([
            -qx * g[0] - qy * g[1] - qz * g[2],
            qw * g[0] + qy * g[2] - qz * g[1],
            qw * g[1] - qx * g[2] + qz * g[0],
            qw * g[2] + qx * g[1] - qy * g[0],
        ])

        # 2) 梯度下降修正：f 是"估计重力方向 vs 实测方向"的误差，
        #    J 是 f 对 q 的雅可比矩阵，J.T @ f 指向误差下降方向
        f = np.array([
            2 * (qx * qz - qw * qy) - ax_,
            2 * (qw * qx + qy * qz) - ay_,
            2 * (0.5 - qx * qx - qy * qy) - az_,
        ])
        J = np.array([
            [-2 * qy,  2 * qz, -2 * qw, 2 * qx],
            [ 2 * qx,  2 * qw,  2 * qz, 2 * qy],
            [ 0.0,    -4 * qx, -4 * qy, 0.0   ],
        ])
        step = J.T @ f
        step /= np.linalg.norm(step)

        # 3) 合成、积分、归一化
        q_new = q + (q_dot_gyro - beta * step) * dt
        return q_new / np.linalg.norm(q_new)

    #逐点估计主循环
    def _estimate_orientation(self):
        gyro = self.gyro
        gyro = np.deg2rad(gyro) #角度转化为弧度

        n = len(self.time)
        quats = np.empty((n, 4))
        quats[0] = np.array([1.0, 0.0, 0.0, 0.0])  # 初始姿态与世界系对齐(算法会很快收敛到真实姿态)

        dt = np.diff(self.time)   # 每步的时间间隔，长度是 n-1

        for i in range(1, n):
            quats[i] = self._madgwick_step(quats[i-1], gyro[i], self.acc[i], dt[i-1], self.beta)  #调用算法逐点估计姿态四元数

        self.quaternions = quats
        return self
    
    #将四元数转化为欧拉角
    def _compute_euler_angles(self):

        if self.quaternions is None:
            raise RuntimeError("请先运行 estimate_orientation()")

        qw = self.quaternions[:, 0]
        qx = self.quaternions[:, 1]
        qy = self.quaternions[:, 2]
        qz = self.quaternions[:, 3]

        # roll (x 轴)
        roll = np.arctan2(2 * (qw * qx + qy * qz),
                        1 - 2 * (qx * qx + qy * qy))
        # pitch (y 轴)，注意 arcsin 的输入要限制在 [-1, 1] 内
        sinp = 2 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # yaw (z 轴)
        yaw = np.arctan2(2 * (qw * qz + qx * qy),
                        1 - 2 * (qy * qy + qz * qz))

        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)
        self.euler_angles = np.stack([roll, pitch, yaw], axis=1)
        return self

    #将欧拉角转化为三角函数
    def _angle_to_trigonometric(self):
        roll = self.euler_angles[:, 0]
        pitch = self.euler_angles[:, 1]
        yaw = self.euler_angles[:, 2]

        roll_sin = np.sin(np.deg2rad(roll))
        roll_cos = np.cos(np.deg2rad(roll))
        pitch_sin = np.sin(np.deg2rad(pitch))
        pitch_cos = np.cos(np.deg2rad(pitch))
        yaw_sin = np.sin(np.deg2rad(yaw))
        yaw_cos = np.cos(np.deg2rad(yaw))
        self.trig = np.stack([roll_sin, roll_cos, pitch_sin, pitch_cos, yaw_sin, yaw_cos], axis=1)
        return self

    
    #拼接为9轴数据
    def _combine_features(self):
        if self.trig is None:
            raise RuntimeError("请先运行 _angle_to_trigonometric()")
        self.features = np.hstack([self.acc, self.gyro, self.euler_angles, self.trig])
        return self
    
    #降采样
    def _downsampling(self, factor):
        self.features = self.features[::factor]
        self.time = self.time[::factor]

    #预处理完成保存为.npz格式文件
    def save(self, out_dir):
        if self.features is None:
            raise RuntimeError("请先调用 preprocess() 再保存")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)   # 文件夹不存在就创建

        np.savez_compressed(
            out_dir / f"{self.name}.npz",
            features=self.features,
            quaternions=self.quaternions,
            time=self.time,
            interp_mask=self.interp_mask,
        )




proc = preprocess("Test_File/A5_C5_C5-c8.csv")
proc.preprocess()
proc.save("output")
# 检查 output/A5_C5_C5-c8.npz 是否生成