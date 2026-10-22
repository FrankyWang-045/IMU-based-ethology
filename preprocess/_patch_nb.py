# -*- coding: utf-8 -*-
"""用 nbformat 给 preprocess_v2/preprocess_pipeline.ipynb 打补丁：
  1) Cell 0: 新增 DO_REGISTRATION / T_DIFF_FILE / FORCE_RELOAD_RAW + autoreload
  2) Cell 1: 受 DO_REGISTRATION 守卫; raw 走 get_raw(按目录缓存)
  3) t_diff 记录 cell: 受守卫
  4) Cell 2: DO_REGISTRATION=False 时直接读 t_diff 文件 + 输出指纹防重复
  5) 概览 markdown: 补充说明
"""
import nbformat

NB = r"F:/WorkBuddy/IMU数据处理/preprocess_v2/preprocess_pipeline.ipynb"
nb = nbformat.read(NB, as_version=4)

# ---------- Cell 0 (code, idx 2) ----------
cell0 = r'''# ============================================================
# Cell 0: 全局参数配置 (Global Paths & Parameters)
# 功能: 集中定义所有路径与参数, 供 Cell 1 / Cell 2 共用
# ============================================================

# 修改本目录下 imu_processor.py / aligner.py 后, 重跑 cell 自动生效, 无需重启内核
%load_ext autoreload
%autoreload 2

# --------------------- 路径配置 ---------------------
VIDEO_PATH  = r"\\hulab.lumiani.ai\Data\WSB\Computational_Ethology\Data\8.25_social_all_paired\A5_C5_C5-c8\data2.mp4"     # 视频文件路径 (Cell 1 配准用)
IMU_DIR     = r"\\hulab.lumiani.ai\Data\WSB\Computational_Ethology\Data\8.25_social_all_paired\A5_C5_C5-c8\c8"            # IMU 原始 CSV 目录
OUTPUT_PATH = r"\\hulab.lumiani.ai\Data\WSB\Computational_Ethology\processed_Data\Train\6channel_100Hz\A5_C5_C5.csv"      # 输出 CSV 路径 (Cell 2 保存用)

# --------------------- 视频配准开关 ---------------------
DO_REGISTRATION = False   # True : 运行 Cell 1 的配准 GUI, 得到 t_diff
                          # False: 跳过 GUI, Cell 2 直接读取 t_diff 文件里的数值
T_DIFF_FILE = None        # t_diff 文件路径; None = 自动取 视频同目录/t_diff.txt

# --------------------- Cell 1 (配准) 专用参数 ---------------------
IMU_DURATION = 120.0  # GUI 中仅显示 IMU 前 N 秒 (加速显示)
VIEW_WINDOW  = 30.0   # IMU 曲线初始视窗宽度 (s)

# --------------------- Cell 2 (处理导出) 专用参数 ---------------------
# 输出采样率固定 100 Hz (见 imu_processor.FS_OUT, 与设备原生率无关)
T_DIFF_MANUAL = None  # 手动指定 t_diff (秒); None = 用 t_diff 文件 / Cell 1 结果
APPLY_LPF     = True  # 是否低通滤波
LPF_FC        = 20.0  # 低通滤波截止频率 (Hz)

# --------------------- 重复运行 / 缓存控制 ---------------------
# notebook 会反复运行: 切换 IMU_DIR 后必须读到新目录的数据, 否则会把上一只鼠的
# 数据重复写到新输出 (这正是「一对小鼠文件完全相同」的根源)。
FORCE_RELOAD_RAW = False  # True: 每次 Cell 1/2 都强制重新读取 IMU 目录
# ----------------------------------------------------

print("配置完成:")
print(f"  VIDEO_PATH       = {VIDEO_PATH}")
print(f"  IMU_DIR          = {IMU_DIR}")
print(f"  OUTPUT_PATH      = {OUTPUT_PATH}")
print(f"  DO_REGISTRATION  = {DO_REGISTRATION}   (False -> 读取 t_diff 文件)")
print(f"  T_DIFF_MANUAL    = {T_DIFF_MANUAL}")
print(f"  FORCE_RELOAD_RAW = {FORCE_RELOAD_RAW}")
'''

# ---------- Cell 1 (code, idx 4) ----------
cell1 = r'''# ============================================================
# Cell 1: 视频配准 (Video-IMU Time Alignment)
# 仅当 Cell 0 的 DO_REGISTRATION=True 时运行 (弹出 GUI 与视频对齐)
# DO_REGISTRATION=False 时本 cell 只提示跳过, Cell 2 会直接读取 t_diff 文件
# ============================================================
%matplotlib tk
from imu_utils import get_raw, t_diff_path
from aligner import IMUVideoAligner

if not DO_REGISTRATION:
    print("DO_REGISTRATION=False -> 跳过视频配准 GUI。")
    print("Cell 2 将直接读取 t_diff 文件:", t_diff_path(VIDEO_PATH, T_DIFF_FILE))
else:
    # 按 IMU_DIR 缓存读取: 目录变化会自动重新加载 (不会复用上一只鼠的旧数据)
    raw = get_raw(IMU_DIR, force=FORCE_RELOAD_RAW, verbose=True)
    print(f"raw 来源目录: {IMU_DIR}  shape={raw.shape}  首样本 ax={raw[0, 1]:.4f}")
    aligner = IMUVideoAligner(
        VIDEO_PATH,
        raw_data=raw,               # 传入预加载数据, 不重复读取磁盘
        lpf_fc=LPF_FC,
        imu_duration=IMU_DURATION,
        view_window=VIEW_WINDOW,
    )
    aligner.show()   # 弹出 GUI 窗口; 关闭窗口后代码继续往下执行
'''

# ---------- t_diff 记录 cell (code, idx 5) ----------
cell5 = r'''# 记录配准结果 (GUI 中 "Save t_diff" 同时会写入视频旁的 t_diff.txt)
if DO_REGISTRATION:
    t_diff = aligner.offset
    print(f"\n配准结果: t_diff = {t_diff:.6f} s  (Cell 2 将默认使用该值)")
else:
    print("未运行配准 (DO_REGISTRATION=False); t_diff 将在 Cell 2 由 t_diff 文件决定。")
'''

# ---------- Cell 2 (code, idx 7) ----------
cell7 = r'''# ============================================================
# Cell 2: 数据处理与导出 (Data Processing & Export)
# 功能: 取得 raw 与 t_diff -> 完整预处理 -> 保存 CSV
# 输出: 100 Hz, 8 列 CSV (time, ax, ay, az, wx, wy, wz, interp_mask)
# ============================================================

from imu_processor import data_preprocess
from imu_utils import get_raw, load_t_diff, save_processed_csv, t_diff_path

# ---- 1) 解析 t_diff ----
if T_DIFF_MANUAL is not None:
    t_diff = float(T_DIFF_MANUAL)
    _src = "Cell 0 的 T_DIFF_MANUAL"
elif not DO_REGISTRATION:
    t_diff = load_t_diff(VIDEO_PATH, T_DIFF_FILE)      # 直接读 t_diff 文件
    _src = f"t_diff 文件 {t_diff_path(VIDEO_PATH, T_DIFF_FILE)}"
elif "t_diff" in globals():
    _src = "Cell 1 配准结果"
else:
    raise RuntimeError("未检测到 t_diff: 请运行 Cell 1 完成配准, 或将 "
                       "DO_REGISTRATION 设为 False 以读取 t_diff 文件, "
                       "或在 Cell 0 设置 T_DIFF_MANUAL")
print(f"t_diff = {t_diff:.6f} s  (来源: {_src})")

# ---- 2) 读取 raw (按 IMU_DIR 缓存; 目录变化自动重载) ----
raw = get_raw(IMU_DIR, force=FORCE_RELOAD_RAW, verbose=True)

# ---- 3) 完整预处理 (复用 raw, 不重复读取) ----
# 输出 100 Hz, 8 列: time, ax, ay, az, wx, wy, wz, interp_mask
proc, info = data_preprocess(
    raw,
    t_diff=t_diff,
    apply_lpf=APPLY_LPF,
    lpf_fc=LPF_FC,
    return_drift=True,
)
print(f"proc shape: {proc.shape}")
print(f"fs_out = {info['fs']:.2f} Hz (device native {info['fs_native']:.2f} Hz, "
      f"drift = {info['drift']:.3f}%)")
print(f"dropped = {info['dropped_samples']}, "
      f"interpolated = {info['interpolated_samples']}")

# ---- 4) 保存 CSV (8 通道: time, ax, ay, az, wx, wy, wz, interp_mask) ----
save_processed_csv(proc, OUTPUT_PATH)

# ---- 5) 输出指纹: 立即核对本次输出确实来自当前 IMU_DIR (防止重复文件) ----
print(f"[指纹] IMU_DIR        = {IMU_DIR}")
print(f"[指纹] rows           = {proc.shape[0]}")
print(f"[指纹] mean(ax,ay,az) = ({proc[:, 1].mean():+.4f}, "
      f"{proc[:, 2].mean():+.4f}, {proc[:, 3].mean():+.4f})")
print(f"[指纹] first(ax,ay,az)= ({proc[0, 1]:+.4f}, "
      f"{proc[0, 2]:+.4f}, {proc[0, 3]:+.4f})")
'''

nb.cells[2]["source"] = cell0
nb.cells[4]["source"] = cell1
nb.cells[5]["source"] = cell5
nb.cells[7]["source"] = cell7

# ---------- 概览 markdown (idx 0): 追加开关说明 ----------
md = nb.cells[0]["source"]
extra = ("\n\n**v2.1 更新（2026-09-16）**\n"
         "- 新增开关 `DO_REGISTRATION`：\n"
         "  - `True` → 运行 Cell 1 交互式配准 GUI，得到 `t_diff`；\n"
         "  - `False` → **跳过 GUI，Cell 2 直接读取视频同目录的 `t_diff.txt`**"
         "（可用 `T_DIFF_FILE` 指定其它路径）。\n"
         "- **修复重复文件隐患**：旧版 Cell 2 用 `if \"raw\" not in globals()` 判断是否读盘，"
         "换 `IMU_DIR` 后 raw 仍是上一只鼠的数据 → 会把同一份数据写进两个输出文件。"
         "现改为按 `IMU_DIR` 缓存（`imu_utils.get_raw`）：**目录变了就自动重载**；"
         "并新增 `FORCE_RELOAD_RAW` 强制刷新、Cell 2 打印**输出指纹**便于核对。\n"
         "- 新增 `%autoreload 2`：修改 `imu_processor.py` / `aligner.py` 后重跑 cell 即生效。")
if "v2.1 更新" not in md:
    nb.cells[0]["source"] = md + extra

nbformat.write(nb, NB)
print("[OK] notebook 已写回:", NB)
print("cells:", len(nb.cells))
