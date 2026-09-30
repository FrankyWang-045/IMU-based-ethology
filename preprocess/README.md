# IMU 数据预处理流水线

IMU 原始数据（CSV）→ 视频-IMU 时间配准 → 完整预处理（时间对齐、晶振频偏校正、20Hz 低通）→ 导出 **~99Hz、8 列**标准 CSV。

> 本目录（`preprocess/`）为**后续预处理版本**：不整体上采样、不做 EKF 重力分离，
> 直接保留 `_align_imu` 估计出的 ~99Hz 均匀信号并做 20Hz 低通滤波。
> 早期「粗分类版本」（构建 VQ-VAE 之前）的配套脚本不在此目录，请勿与本流水线混淆。

## 目录结构

| 文件 | 说明 |
|------|------|
| `imu_processor.py` | 核心库：`data_import()`（批量导入 CSV，取前 9 列）、`data_preprocess()`（时间对齐 + 20Hz 低通，输出 ~99Hz 8 列） |
| `aligner.py` | 交互式 IMU-视频时间配准 GUI（`IMUVideoAligner`），支持传入预加载的 `raw` 数组 |
| `imu_utils.py` | 辅助工具：`save_processed_csv()` |
| `preprocess_pipeline.ipynb` | 主入口 notebook，3 个 cell（全局配置 / 配准 / 处理与导出） |

## 环境依赖

- Python ≥ 3.9
- numpy、pandas、scipy、matplotlib、opencv-python

```
pip install numpy pandas scipy matplotlib opencv-python
```

## 快速开始

在 `preprocess_pipeline.ipynb` 中按顺序运行三个 cell：

0. **Cell 0 — 全局参数配置**：集中定义所有路径（`VIDEO_PATH` / `IMU_DIR` / `OUTPUT_PATH`）与参数（各 cell 专用参数）。修改配置只需编辑本 cell。
1. **Cell 1 — 视频配准**：弹出 GUI 窗口完成对齐（操作见下节），关闭窗口后自动记录 `raw` 和 `t_diff`。
2. **Cell 2 — 数据处理与导出**：默认复用 Cell 1 的 `raw` 与 `t_diff`，运行后完成预处理并保存到 `OUTPUT_PATH` 指定的 CSV（~99Hz、8 列）。

**两种使用模式**：
- 先配准再处理（推荐）：顺序运行 Cell 0 → Cell 1 → Cell 2，数据只从磁盘读取一次。
- 跳过配准直接处理：已知 `t_diff` 时，在 Cell 0 中把 `T_DIFF_MANUAL` 设为具体数值（如 `8.138736`），直接运行 Cell 0 → Cell 2（此时 raw 会在 Cell 2 内自动加载）。

## 配准 GUI 操作说明

以小鼠坠入行为箱的瞬间作为同步信号（IMU 曲线上出现冲击尖峰）：

| 操作 | 作用 |
|------|------|
| 滑条 / 步进按钮 / ← → 键 | 视频逐帧浏览；空格键播放/暂停 |
| 定位到坠落帧后，点击 **"1. Mark video event"** | 标记视频事件时间 |
| 点击 **"2. Mark IMU event"**，再点击曲线上的坠落尖峰 | 标记 IMU 事件时间，自动计算 `t_diff` |
| 右键拖动 / Shift+左键拖动 | 平移 IMU 曲线（微调对齐） |
| 左键拖动 / 滚轮 | 平移视图 / 缩放 |
| Shift+← / Shift+→ | 按一个视频帧的时长微调 `t_diff` |
| **"Save t_diff"** | 将结果写入视频旁的 `t_diff.txt` |

**符号约定**：`t_diff = IMU 事件时间 − 视频事件时间`，即 IMU 时间减去 `t_diff` 后与视频时间对齐。

首次运行时，视频前 `VIDEO_DURATION` 秒（默认 120）会被一次性抽取为 360p JPEG 帧缓存（存放在视频旁的 `*_360p_frames/` 目录），之后复用缓存，无需删除。

## 数据格式

**输入**：目录下的原始 CSV（**带表头**，约 23 列）。`data_import()` 自动跳过表头、仅取前 9 列
（系统时间、设备名、片上时间、加速度 X/Y/Z、角速度 X/Y/Z），角度/磁场/四元数等占位列（常含空格 `' '`）被跳过。

**输出**：`save_processed_csv()` 保存的 CSV 含 **8 列**：

| 列 | 含义 |
|----|------|
| `time` | 对齐后的时间（秒，**非整数秒**、~99Hz 均匀；从 0 附近起，t_diff 已扣除） |
| `ax, ay, az` | 20Hz 低通后的加速度（g） |
| `wx, wy, wz` | 20Hz 低通后的角速度（deg/s） |
| `interp_mask` | 插值掩码：1 = 丢帧间隙内的插值点，0 = 正常采样点 |

> 输出采样率 ≈ `_align_imu` 估计的片上采样率（~99Hz），**不做上采样**，因此时间戳为
> 非整数秒且网格密度为原始采样密度。

## 预处理流水线（`data_preprocess` 内部）

1. **时间对齐**（保持原逻辑不变）：片上时间与系统时间对齐，线性拟合校正晶振频偏（~99 Hz），PCHIP 插值补掉帧，得到 `~99Hz` 均匀信号与丢帧间隙 `gaps`。
2. **时间偏移**：按 `t_diff` 做秒级全局偏移 `ta = t_new_0 − t_diff`，丢弃 `ta < 0` 的样本（与配准 GUI 结果一致）。
3. **插值掩码**：在 `~99Hz` 时间轴上生成，落在丢帧间隙 `[gap_start, gap_end]` 内的**所有**采样点（含补帧点）均标记 `interp_mask = 1`。
4. **20Hz 低通滤波**（4 阶 Butterworth）：作用于 6 轴信号。滤波器的 Nyquist 与阶跃响应均使用 `_align_imu` 估计的真实 `fs`（~99Hz），因此在 ~99Hz 信号上以 20Hz 截止可正常工作（不依赖旧版 `fs>100` 守卫）。**不做 EKF 姿态解算 / 重力分离**。

## 下游切分（滑动窗口等）约定

后续对预处理输出做切分时，**以采样点数（sample count）为依据，不以秒为依据**；
切分时需**同步切分对应的 `time` 列**，使每段数据都携带其时间戳。

- 若需「约 N 秒」的窗口，请先用 `fs ≈ info['fs']`（~99Hz）换算为采样点数，例如 2s ≈ 198 点。
- `data_preprocess(..., return_drift=True)` 会返回 `info['fs']`，供下游换算窗长/步长。

## 常见问题

- **GUI 窗口不弹出**：确认 Cell 1 首行的 `%matplotlib tk` 生效；JupyterLab 需安装 `ipympl` 并改用 `%matplotlib widget`。
- **视频帧抽取很慢**：仅首次运行抽取，之后读取缓存；如需重新抽取，删除视频旁的 `*_360p_frames/` 目录。
- **更换数据集**：修改 Cell 0 参数区的 `VIDEO_PATH` / `IMU_DIR` / `OUTPUT_PATH` 后，重启 kernel 并按顺序重新运行全部 cell（`raw` 与 `t_diff` 是跨 cell 共享的内存变量）。
- **想改低通截止频率**：修改 Cell 0 的 `LPF_FC`（默认 20.0 Hz）；想关掉低通就把 `APPLY_LPF` 设为 `False`。
- **输出只有 8 列（无 aG/anG）**：这是本版本的设计——只做 20Hz 低通，不做 EKF 重力分离。若需要重力/非重力分离，请使用早期粗分类版本（不在此目录）。
