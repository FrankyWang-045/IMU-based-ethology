# IMU 数据预处理流水线 (preprocess_v2)

IMU 原始数据（CSV）→ 视频-IMU 时间配准 → 完整预处理（时间对齐、**统一 100 Hz 重采样**、20Hz 低通）→ 导出 **100 Hz、8 列**标准 CSV。

> **与 `preprocess/`（v1）的唯一实质差异**：输出采样率由「设备原生率（~99.3 Hz，随设备浮动）」
> 改为**固定 100 Hz**。这使不同设备、不同批次的数据落在同一条时间栅格上，可以逐行并列比较。
> 路径参数、目录结构、cell 划分、函数签名、输出格式与 v1 完全一致，下游代码无需修改。

## 目录结构

| 文件 | 说明 |
|------|------|
| `imu_processor.py` | 核心库：`data_import()`（批量导入 CSV，取前 9 列）、`data_preprocess()`（时间对齐 + 100 Hz 重采样 + 20Hz 低通，输出 8 列）。含 `__main__` CLI 入口 |
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

### 方式一：notebook（与 v1 完全相同）

在 `preprocess_pipeline.ipynb` 中按顺序运行三个 cell：

0. **Cell 0 — 全局参数配置**：集中定义所有路径（`VIDEO_PATH` / `IMU_DIR` / `OUTPUT_PATH`）与参数（各 cell 专用参数）。修改配置只需编辑本 cell。
1. **Cell 1 — 视频配准**：弹出 GUI 窗口完成对齐（操作见下节），关闭窗口后自动记录 `raw` 和 `t_diff`。
2. **Cell 2 — 数据处理与导出**：默认复用 Cell 1 的 `raw` 与 `t_diff`，运行后完成预处理并保存到 `OUTPUT_PATH` 指定的 CSV（100 Hz、8 列）。

**两种使用模式**：
- 先配准再处理（推荐）：顺序运行 Cell 0 → Cell 1 → Cell 2，数据只从磁盘读取一次。
- 跳过配准直接处理：已知 `t_diff` 时，在 Cell 0 中把 `T_DIFF_MANUAL` 设为具体数值（如 `8.138736`），直接运行 Cell 0 → Cell 2（此时 raw 会在 Cell 2 内自动加载）。

### 方式二：命令行（v2 新增）

```bash
python imu_processor.py <IMU数据目录> <输出CSV路径> [选项]

# 示例
python imu_processor.py "F:/Data/Data1_83" "F:/out/mouse_A.csv" --t-diff 6.56
python imu_processor.py "F:/Data/Data1_83" "F:/out/raw.csv" --no-lpf      # 不低通
python imu_processor.py "F:/Data/Data1_83" "F:/out/lin.csv" --interp linear
```

| 选项 | 默认 | 说明 |
|------|------|------|
| `--t-diff` | `0.0` | 时间偏移（秒），对齐视频用 |
| `--fs-out` | `100.0` | 统一输出采样率 (Hz) |
| `--interp` | `pchip` | 重采样插值方式：`pchip` / `linear` |
| `--lpf-fc` | `20.0` | 低通截止频率 (Hz) |
| `--no-lpf` | 关 | 关闭低通滤波 |
| `--gap-threshold` | `0.02` | 丢帧间隙判定阈值（秒） |

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

**输出**：`save_processed_csv()` 保存的 CSV 含 **8 列**（与 v1 完全相同）：

| 列 | 含义 |
|----|------|
| `time` | 对齐后的时间（秒，**100 Hz 等间隔**；从 0 附近起，t_diff 已扣除） |
| `ax, ay, az` | 20Hz 低通后的加速度（g） |
| `wx, wy, wz` | 20Hz 低通后的角速度（deg/s） |
| `interp_mask` | 插值掩码：1 = 丢帧间隙内的插值点，0 = 正常采样点 |

> 输出采样率恒为 `imu_processor.FS_OUT = 100.0` Hz，与设备原生率无关。
> 因此 `time` 列是 0.01 s 的整数倍（减去 `t_diff` 后），不同设备的数据可以直接按行号并列。

## 预处理流水线（`data_preprocess` 内部）

1. **时间对齐**（逻辑与 v1 一致）：剔除 BLE 损坏时间戳 → 线性拟合校正晶振频偏，把片上采样计数器映射回主机墙钟（真实秒）→ 得到不规则样本时间轴 `t_est` 与丢帧间隙 `gaps`。
2. **100 Hz 重采样（唯一一次插值）**：用 PCHIP（或线性）把不规则原始样本**直接**插值到 `fs_out`（默认 100 Hz）等间隔栅格上。这一步同时完成 v1 中的「补丢帧」与「重采样」，比 v1 少一次插值。
3. **时间偏移**：按 `t_diff` 做秒级全局偏移 `ta = t_new − t_new[0] − t_diff`，丢弃 `ta < 0` 的样本。
4. **插值掩码**：在 100 Hz 时间轴上生成，落在丢帧间隙 `[gap_start, gap_end]` 内的**所有**采样点（含补帧点）均标记 `interp_mask = 1`。
5. **20Hz 低通滤波**（4 阶 Butterworth，零相位）：作用于 6 轴信号，Nyquist 用 `fs_out`（100 Hz），20 Hz 截止对应 2.5 倍余量。**不做 EKF 姿态解算 / 重力分离**。

### 关于插值方式

`data_preprocess(..., interp_method=...)` 可选：

| 方式 | 说明 |
|------|------|
| `pchip`（默认） | 分段三次 Hermite 保形插值，无过冲；与 v1 行为一致，数值误差更小 |
| `linear` | 分段线性插值，更直观、更快，但插值误差略大 |

实测（Data1_83，78 min 全量 6 通道）。参考基准 =「在设备原生率栅格上先做 20 Hz 低通，
再投到 1000 Hz 公共栅格」，下表为相对 RMS 误差：

| 通道 | `pchip` | `linear` | 差值 |
|------|---------|----------|------|
| ax | 0.875 % | 1.275 % | +0.40 |
| ay | 0.797 % | 1.106 % | +0.31 |
| az | 0.790 % | 1.136 % | +0.35 |
| wx | 2.273 % | 3.641 % | +1.37 |
| wy | 2.251 % | 3.681 % | +1.43 |
| wz | 2.095 % | 3.582 % | +1.49 |

结论：**默认用 `pchip`**。`linear` 的误差约为 `pchip` 的 1.4~1.7 倍（角速度通道差距最明显），
虽然绝对量仍在 3.7% 以内、对窗口级行为分类影响有限，但没有理由主动选更差的那个。
`linear` 仅在需要「插值结果严格落在相邻样本连线上」这类可解释性要求时才有意义。

## v1 → v2 变更清单

**删除的变量 / 参数 / 导入**

| 位置 | 删除项 | 原因 |
|------|--------|------|
| `_frameloss_test()` | 整个函数（含 `fl_count` / `lostframes` 两个变量） | 与 `gaps` 重复计算丢帧；且是一次 O(n) Python 循环。丢帧统计改由 `gaps` 统一给出 |
| `_align_imu()` 返回值 | `sig`（清洗后的原始信号） | 调用方 `data_preprocess` 从未使用 |
| `_align_imu()` 返回值 | `t_est`（反解时间轴） | 调用方从未使用，`gaps` 已在函数内部算好 |
| `_align_imu()` 内部 | 逐通道 `for` 循环的 6 个 `PchipInterpolator` | `PchipInterpolator` 原生支持 (n, 6) 按 axis=0 插值，循环属重复构造 |
| `data_import()` | `df_list` 临时列表、`data` → `df` 的中间命名 | 用 `pd.concat(生成器)` 一次完成 |
| `data_import()` | 两次 `reshape(-1, 1)` 临时数组 | 改用 `np.column_stack` |
| `data_preprocess()` | `t_new_0` | 与 `ta` 合并为一行 |
| `data_preprocess()` | `imu_p` | 直接把 `out[keep]` 传入滤波器 |
| `data_preprocess()` | `mask_full` / `mask_p` | 合并为一个 `mask`（直接在保留区间上生成） |
| `_lowpass_6axis()` | 冗余的 `fs <= 2*lpf_fc` 提前返回分支 | 与前面的守卫合并为一个条件 |
| `aligner.py` | `_step_btns` / `_op_btns` 两个列表 | 合并为 `_btns`（用途仅是保持控件引用不被 GC） |
| 全局 | `fs = 100.0 * k` 之外的频率硬编码 | 统一为常量 `FS_OUT` |

**新增**

| 项 | 说明 |
|----|------|
| `FS_OUT = 100.0` | 全局统一输出采样率，全流程唯一的频率常量 |
| `INTERP_METHOD = "pchip"` | 显式的重采样插值方式常量 |
| `LPF_ORDER = 4` | 滤波器阶数常量 |
| `_make_grid()` | 等间隔栅格构造函数；与 v1 的 `arange(t0, t1+dt/2, dt)` 不同，严格不越过 `t1`，避免末点外推 |
| `_build_interpolator()` | 隔离 `pchip` / `linear` 两种插值方式的构造 |
| `data_preprocess(fs_out, interp_method)` | 两个**带默认值**的可选参数，不传即为 100 Hz + pchip，旧调用方式不受影响 |
| `info['fs_native']` | 返回设备原生率估计（用于诊断）。`info['fs']` 现在返回输出率 100.0 |
| `_main()` / `__main__` | 命令行入口 |

**行为差异（除采样率外的两点）**

1. `data_import()` 改为 `sorted(os.listdir(...))`：v1 依赖 `os.listdir` 的返回顺序（NTFS 恰好字母序），分文件录制时顺序错乱会静默破坏时间序列。
2. `_make_grid()` 不再在末尾多生成一个点：v1 的 `arange(t0, t1+dt/2, dt)` 在小数部分 ≥ 0.5 时会多出 1 个越过 `t1` 的点，构成外推；v2 严格限制在 `[t0, t1]`。

## 下游切分（滑动窗口等）约定

后续对预处理输出做切分时，**以采样点数（sample count）为依据，不以秒为依据**；
切分时需**同步切分对应的 `time` 列**，使每段数据都携带其时间戳。

- 输出固定 100 Hz，换算直接：`400 点 = 4.000 s`、`200 点 = 2.000 s`。
- `data_preprocess(..., return_drift=True)` 会返回 `info['fs']`（恒为 100.0）与
  `info['fs_native']`（设备原生率，用于诊断时钟是否正常）。

## 常见问题

- **GUI 窗口不弹出**：确认 Cell 1 首行的 `%matplotlib tk` 生效；JupyterLab 需安装 `ipympl` 并改用 `%matplotlib widget`。
- **视频帧抽取很慢**：仅首次运行抽取，之后读取缓存；如需重新抽取，删除视频旁的 `*_360p_frames/` 目录。
- **更换数据集**：修改 Cell 0 参数区的 `VIDEO_PATH` / `IMU_DIR` / `OUTPUT_PATH` 后，重启 kernel 并按顺序重新运行全部 cell（`raw` 与 `t_diff` 是跨 cell 共享的内存变量）。
- **想改低通截止频率**：修改 Cell 0 的 `LPF_FC`（默认 20.0 Hz）；想关掉低通就把 `APPLY_LPF` 设为 `False`。
- **想改输出采样率**：`data_preprocess(..., fs_out=200.0)`，或 CLI `--fs-out 200`。整条流水线没有其他频率硬编码。
- **输出只有 8 列（无 aG/anG）**：这是本版本的设计——只做 20Hz 低通，不做 EKF 重力分离。若需要重力/非重力分离，请使用早期粗分类版本（不在此目录）。
