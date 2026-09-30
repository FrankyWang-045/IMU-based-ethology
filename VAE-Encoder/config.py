"""
IMU-Former 配置模块
====================
所有超参数集中定义，基于 EEGFormer 框架迁移到 9 轴 IMU 行为数据。

设计决策（用户确认，2026-08-20）：
  1. 输入域：时域 + 轻量频域辅助头（λ=0 起步，纯 VQ）
  2. 跨轴：分模态 patch（g/a/ω 各 3 轴），共享编码器，耦合交给注意力
  3. 增广：暂不启用（最小验证阶段）
  4. Euler 角：不入模型（避免冗余）
  5. 窗长：2s @ 256Hz = 512 点
  6. 码本：K=256
  7. 采样率：256Hz（用户确认；注意现有 configs.py 中 SAMPLE_RATE=300 是旧管线值）

参数推导关系：
  window_len = sample_rate * window_sec = 256 * 2 = 512
  n_patches_per_modality = floor((window_len - patch_len) / patch_stride) + 2
                         = floor((512 - 64) / 32) + 2 = 16
  total_tokens = 16 * 3 模态 = 48
  patch_flat_dim = patch_len * 3 轴 = 192
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import math


@dataclass
class IMUFormerConfig:
    """IMU-Former 全局配置。所有派生量在 __post_init__ 中自动计算。"""

    # ═══════════════════════════════════════════════════════════════
    # 数据参数
    # ═══════════════════════════════════════════════════════════════
    sample_rate: int = 256          # 采样率 Hz（用户确认）
    window_sec: float = 2.0         # 窗口时长（秒）
    hop_sec: float = 1.0            # 窗口步进（秒），50% 重叠

    # CSV 列名（与用户数据完全一致）
    # time, aGx,aGy,aGz, anGx,anGy,anGz, wx,wy,wz, interp_mask
    channel_names: Tuple[str, ...] = (
        'aGx', 'aGy', 'aGz',       # 重力加速度 (3 轴)
        'anGx', 'anGy', 'anGz',    # 非重力加速度 (3 轴)
        'wx', 'wy', 'wz',          # 角速度 (3 轴)
    )
    mask_col: str = 'interp_mask'   # 插值掩码列：1=插值, 0=原始

    # 模态分组（channel_names 中的索引）
    modality_groups: Tuple[Tuple[int, ...], ...] = (
        (0, 1, 2),    # gravity: aGx, aGy, aGz
        (3, 4, 5),    # accel:   anGx, anGy, anGz
        (6, 7, 8),    # gyro:    wx, wy, wz
    )
    modality_names: Tuple[str, ...] = ('gravity', 'accel', 'gyro')

    # 固定单位缩放（非统计归一化！全程对所有样本施加相同常数）
    # 设为 None 则不缩放；设为具体值则按通道乘以缩放因子
    # 推荐预设：将 gyro 从 deg/s 转为 rad/s（÷57.2958），使三模态数值量级接近
    #   → channel_scales = (1, 1, 1, 1, 1, 1, math.pi/180, math.pi/180, math.pi/180)
    # 用户倾向不归一化（绝对幅值有行为意义），默认 None，最小验证先不缩放
    channel_scales: Optional[Tuple[float, ...]] = None

    # ═══════════════════════════════════════════════════════════════
    # Patching 参数
    # ═══════════════════════════════════════════════════════════════
    patch_len: int = 64             # P = 64 采样点 = 0.25s @ 256Hz
    patch_stride: int = 32          # S = 32 采样点 = 50% 重叠

    # ═══════════════════════════════════════════════════════════════
    # 模型参数（后续步骤使用，此处统一声明）
    # ═══════════════════════════════════════════════════════════════
    d_model: int = 128              # 嵌入维度 D
    codebook_size: int = 256        # 码本大小 K
    n_encoder_layers: int = 4       # 编码器层数
    n_decoder_layers: int = 2       # 解码器层数（故意浅）
    n_heads: int = 4                # 多头注意力头数

    # ═══════════════════════════════════════════════════════════════
    # 损失参数
    # ═══════════════════════════════════════════════════════════════
    commit_beta: float = 0.25       # 承诺损失权重 β
    lambda_freq: float = 0.0        # 频域辅助损失权重 λ（=0 → 纯 VQ）

    # ═══════════════════════════════════════════════════════════════
    # 派生量（自动计算，勿手动设置）
    # ═══════════════════════════════════════════════════════════════
    window_len: int = field(init=False, default=0)
    hop_len: int = field(init=False, default=0)
    n_patches_per_modality: int = field(init=False, default=0)
    n_modalities: int = field(init=False, default=0)
    total_tokens: int = field(init=False, default=0)
    patch_flat_dim: int = field(init=False, default=0)

    def __post_init__(self):
        """从基本参数推导所有派生量。"""
        # 窗口与步进（采样点数）
        self.window_len = int(self.sample_rate * self.window_sec)
        self.hop_len = int(self.sample_rate * self.hop_sec)

        # 每 modality 的 patch 数量（PatchTST 公式，含右侧 reflect padding）
        # N = floor((L - P) / S) + 2
        # 实现时对信号右侧 reflect-pad (P - S) 个点，使边界 patch 完整
        self.n_patches_per_modality = (
            (self.window_len - self.patch_len) // self.patch_stride + 2
        )
        self.n_modalities = len(self.modality_groups)
        self.total_tokens = self.n_patches_per_modality * self.n_modalities
        self.patch_flat_dim = self.patch_len * 3  # 每 modality 3 轴

        # 一致性检查
        assert self.window_len % self.patch_stride == 0, (
            f"window_len ({self.window_len}) 应为 patch_stride ({self.patch_stride}) 的整数倍，"
            f"否则 patch 数量公式可能不精确"
        )
        assert len(self.channel_names) == 9, "应有 9 个通道"
        assert sum(len(g) for g in self.modality_groups) == 9, "模态分组应覆盖全部 9 通道"

    def summary(self) -> str:
        """返回可读的参数摘要。"""
        lines = [
            "════════ IMU-Former 配置摘要 ════════",
            f"  采样率:          {self.sample_rate} Hz",
            f"  窗口:            {self.window_sec}s = {self.window_len} 点",
            f"  步进:            {self.hop_sec}s = {self.hop_len} 点 (50% 重叠)",
            f"  Patch:           P={self.patch_len} ({self.patch_len/self.sample_rate:.3f}s), "
            f"S={self.patch_stride} ({self.patch_stride/self.sample_rate:.3f}s)",
            f"  Patch/模态:      {self.n_patches_per_modality}",
            f"  模态数:          {self.n_modalities} ({', '.join(self.modality_names)})",
            f"  总 token 数:     {self.total_tokens}",
            f"  Patch 展平维度:  {self.patch_flat_dim} (= {self.patch_len} × 3)",
            f"  模型维度 D:      {self.d_model}",
            f"  码本大小 K:      {self.codebook_size}",
            f"  编码器层数:      {self.n_encoder_layers}",
            f"  解码器层数:      {self.n_decoder_layers}",
            f"  承诺损失 β:      {self.commit_beta}",
            f"  频域辅助 λ:      {self.lambda_freq} ({'纯VQ' if self.lambda_freq == 0 else '启用'})",
            f"  单位缩放:        {'无' if self.channel_scales is None else '已设'}",
            "══════════════════════════════════════",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 独立验证模块
# ═══════════════════════════════════════════════════════════════
def _verify():
    """验证配置推导的正确性。"""
    cfg = IMUFormerConfig()

    print(cfg.summary())

    # 检查关键数值
    assert cfg.window_len == 512, f"window_len 应为 512, 得到 {cfg.window_len}"
    assert cfg.n_patches_per_modality == 16, (
        f"n_patches_per_modality 应为 16, 得到 {cfg.n_patches_per_modality}"
    )
    assert cfg.total_tokens == 48, f"total_tokens 应为 48, 得到 {cfg.total_tokens}"
    assert cfg.patch_flat_dim == 192, f"patch_flat_dim 应为 192, 得到 {cfg.patch_flat_dim}"

    # 验证 PatchTST 公式: N = floor((L-P)/S) + 2
    formula_val = (cfg.window_len - cfg.patch_len) // cfg.patch_stride + 2
    assert formula_val == cfg.n_patches_per_modality

    # 验证实际 unfold 后的 patch 数（右侧 pad P-S 后）
    pad_right = cfg.patch_len - cfg.patch_stride  # 32
    effective_len = cfg.window_len + pad_right     # 544
    actual_patches = (effective_len - cfg.patch_len) // cfg.patch_stride + 1
    assert actual_patches == cfg.n_patches_per_modality, (
        f"unfold 实际 patch 数 ({actual_patches}) != 公式值 ({cfg.n_patches_per_modality})"
    )

    print("\n[OK] 配置验证全部通过")
    print(f"  window_len={cfg.window_len}, pad_right={pad_right}, "
          f"effective_len={effective_len}, patches={actual_patches}")

    # 测试自定义参数
    cfg_custom = IMUFormerConfig(window_sec=1.0, patch_len=32, patch_stride=16)
    assert cfg_custom.window_len == 256
    assert cfg_custom.n_patches_per_modality == (256 - 32) // 16 + 2  # = 15
    print(f"\n[OK] 自定义参数验证: 1s 窗口 → {cfg_custom.window_len} 点, "
          f"{cfg_custom.n_patches_per_modality} patches/模态")


if __name__ == '__main__':
    _verify()
