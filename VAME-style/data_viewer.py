import numpy as np
import matplotlib.pyplot as plt

# ==================== 用户设置区 ====================
file_path = r'f:\IMU_computational_ethology\Data\VAME-Style-Data\容量测试\n=4\A4_C4_C4-c8.npz'

# 手动设置信号显示范围，单位：秒
start_time_sec = 380 # 起始时间（秒）
end_time_sec   = 395    # 结束时间（秒）

# 通道名称与颜色（每张子图 3 个通道）
channel_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
# ===================================================

# 读取 .npz 文件
data = np.load(file_path)
features = data['features']   # shape: (N, 6)
time_sec = data['time']       # shape: (N,)

# 根据时间范围筛选索引
mask = (time_sec >= start_time_sec) & (time_sec <= end_time_sec)
t = time_sec[mask]
sig = features[mask, :]       # shape: (M, 6)

# 创建上下两个子图
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 第一张图：通道 0,1,2
for i in range(3):
    axes[0].plot(t, sig[:, i], color=colors[i], label=channel_names[i])
axes[0].set_ylabel('Amplitude')
axes[0].set_title(f'Signals ({start_time_sec}s ~ {end_time_sec}s)')
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.5)

# 第二张图：通道 3,4,5
for i in range(3, 6):
    axes[1].plot(t, sig[:, i], color=colors[i], label=channel_names[i])
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()