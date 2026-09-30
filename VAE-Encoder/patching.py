"""
IMU-Former 分块模块 (Patching)
================================
将 9 轴 IMU 窗口按模态分组、分块、展平为 token 序列。

设计原理（参考 EEGFormer / PatchTST）：
  - 分模态 patch：g(3轴), a(3轴), ω(3轴) 各自独立切块
    理由：每个模态的 3 轴是同一物理量在不同坐标系的投影，是最小不可分解单元；
    跨模态耦合留给编码器注意力层学习。
  - Reflect padding：右侧 pad (P - S) 个点，使边界 patch 完整
    （PatchTST 公式 N = floor((L-P)/S) + 2）
  - 无可学习参数：纯确定性变换

数据流：
  输入:  (B, 512, 9)  ← batch × 时间 × 9通道
         ↓ 按模态分组
  分块:  g → (B, 16, 64, 3)    ← 16 个 patch, 每个 64 点 × 3 轴
         a → (B, 16, 64, 3)
         ω → (B, 16, 64, 3)
         ↓ 展平 P×3
  token: g → (B, 16, 192)
         a → (B, 16, 192)
         ω → (B, 16, 192)
         ↓ 拼接
  输出:  (B, 48, 192)  ← 48 个 token, 每个 192 维
         + modality_ids: (B, 48)  0=g, 1=a, 2=ω
         + positions:    (B, 48)  patch 在模态内的序号 (0..15)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple

from .config import IMUFormerConfig

logger = logging.getLogger(__name__)


class ModalityPatcher(nn.Module):
    """
    分模态 Patching 模块（无可学习参数）。

    Parameters
    ----------
    config : IMUFormerConfig
        提供 patch_len, patch_stride, modality_groups 等参数。
    """

    def __init__(self, config: IMUFormerConfig):
        super().__init__()
        self.config = config
        self.patch_len = config.patch_len          # P = 64
        self.stride = config.patch_stride          # S = 32
        self.modality_groups = config.modality_groups
        self.n_modalities = config.n_modalities    # 3
        self.n_patches_per_mod = config.n_patches_per_modality  # 16
        self.total_tokens = config.total_tokens    # 48
        self.patch_flat_dim = config.patch_flat_dim  # 192

        # 右侧 reflect padding 量（使边界 patch 完整）
        self.pad_right = self.patch_len - self.stride  # = 32

        # 预计算 modality_ids 和 positions（对所有样本相同）
        # token 排列：[g_0..g_15, a_0..a_15, ω_0..ω_15]
        mod_ids = []
        positions = []
        for m in range(self.n_modalities):
            mod_ids.extend([m] * self.n_patches_per_mod)
            positions.extend(range(self.n_patches_per_mod))

        self.register_buffer('modality_ids', torch.tensor(mod_ids, dtype=torch.long))
        self.register_buffer('positions', torch.tensor(positions, dtype=torch.long))

        logger.info(
            "ModalityPatcher: P=%d S=%d pad_right=%d → %d patches/模态, "
            "%d 总 token, flat_dim=%d",
            self.patch_len, self.stride, self.pad_right,
            self.n_patches_per_mod, self.total_tokens, self.patch_flat_dim,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        分模态切块 + 展平。

        Args:
            x: (B, L, 9) — batch 个 IMU 窗口

        Returns:
            patches:   (B, 48, 192) — 展平后的 patch token 序列
            mod_ids:   (B, 48)      — 每个 token 的模态 ID (0=g, 1=a, 2=ω)
            positions: (B, 48)      — 每个 token 在模态内的位置 (0..15)
        """
        B, L, C = x.shape
        assert C == 9, f"期望 9 通道, 得到 {C}"
        assert L == self.config.window_len, (
            f"期望 window_len={self.config.window_len}, 得到 {L}"
        )

        all_patches = []

        for m_idx, ch_indices in enumerate(self.modality_groups):
            # ── 1. 提取模态通道 ──
            # x[:, :, (i,j,k)] → (B, L, 3)
            x_mod = x[:, :, ch_indices]

            # ── 2. 转置为 (B, 3, L) 以便在时间维做 padding ──
            x_mod_t = x_mod.permute(0, 2, 1)  # (B, 3, L)

            # ── 3. 右侧 reflect-pad: (B, 3, L) → (B, 3, L + pad_right) ──
            x_padded = F.pad(x_mod_t, (0, self.pad_right), mode='reflect')
            # L=512 → 544

            # ── 4. Unfold: 沿时间维滑窗切块 ──
            # tensor.unfold(dim, size, step)
            # (B, 3, 544) → (B, 3, N_patches, P)
            patches = x_padded.unfold(2, self.patch_len, self.stride)
            # 验证: (544 - 64) // 32 + 1 = 16 ✓

            # ── 5. 重排维度: (B, 3, N, P) → (B, N, P, 3) ──
            patches = patches.permute(0, 2, 3, 1).contiguous()  # (B, 16, 64, 3)

            # ── 6. 展平 P×3: (B, 16, 64, 3) → (B, 16, 192) ──
            patches = patches.reshape(B, self.n_patches_per_mod, -1)

            all_patches.append(patches)

        # ── 7. 拼接三模态: (B, 48, 192) ──
        # 顺序: [g_0..g_15, a_0..a_15, ω_0..ω_15]
        patches_cat = torch.cat(all_patches, dim=1)

        # ── 8. 展开 modality_ids 和 positions 到 batch ──
        mod_ids = self.modality_ids.unsqueeze(0).expand(B, -1)    # (B, 48)
        positions = self.positions.unsqueeze(0).expand(B, -1)      # (B, 48)

        return patches_cat, mod_ids, positions

    def patches_to_signal(self, patches: torch.Tensor) -> torch.Tensor:
        """
        将 patch token 序列还原为信号（overlap-add + 均值）。

        用途：
          - Decoder 输出头：把重建的 patch → 重建的时域信号
          - 验证：检查 patch 内容是否正确捕获原始信号

        Args:
            patches: (B, 48, 192) — 与 forward 输出格式相同

        Returns:
            signal: (B, L, 9) — 重建信号（重叠区域取均值）
        """
        B = patches.shape[0]
        L = self.config.window_len

        signal = torch.zeros(B, L, 9, device=patches.device, dtype=patches.dtype)

        # 先计算 overlap count（三模态 patching 结构相同，count 一致）
        count_1d = torch.zeros(L, device=patches.device, dtype=patches.dtype)
        for i in range(self.n_patches_per_mod):
            start = i * self.stride
            end = min(start + self.patch_len, L)
            count_1d[start:end] += 1.0

        offset = 0
        for m_idx, ch_indices in enumerate(self.modality_groups):
            n = self.n_patches_per_mod
            # 取出当前模态的 patches: (B, n, 192)
            mod_patches = patches[:, offset:offset + n, :].contiguous()
            offset += n

            # Reshape: (B, n, 192) → (B, n, P, 3)
            mod_patches = mod_patches.reshape(B, n, self.patch_len, 3)

            # Overlap-add
            for i in range(n):
                start = i * self.stride
                end = min(start + self.patch_len, L)
                length = end - start

                # 逐通道累加（避免高级索引 += 不写回的问题）
                for j, ch in enumerate(ch_indices):
                    signal[:, start:end, ch] += mod_patches[:, i, :length, j]

        # 重叠区域取均值（count 对所有通道一致）
        signal = signal / count_1d[None, :, None].clamp(min=1.0)
        return signal


# ═══════════════════════════════════════════════════════════════
# 独立验证模块
# ═══════════════════════════════════════════════════════════════

def _verify():
    """验证 Patching 模块的正确性。"""
    print("══════ ModalityPatcher 验证 ══════\n")
    cfg = IMUFormerConfig()
    patcher = ModalityPatcher(cfg)

    B = 4  # batch size
    L = cfg.window_len  # 512

    # --- 测试 1: 形状检查 ---
    print("[1] 形状检查 ...")
    x = torch.randn(B, L, 9)
    patches, mod_ids, positions = patcher(x)

    assert patches.shape == (B, 48, 192), (
        f"patches.shape 应为 ({B}, 48, 192), 得到 {patches.shape}"
    )
    assert mod_ids.shape == (B, 48), f"mod_ids.shape 应为 ({B}, 48)"
    assert positions.shape == (B, 48), f"positions.shape 应为 ({B}, 48)"
    print(f"    输入:  x = {x.shape}")
    print(f"    输出:  patches = {patches.shape}  ✓")
    print(f"           mod_ids = {mod_ids.shape}  ✓")
    print(f"           positions = {positions.shape}  ✓")

    # --- 测试 2: 模态 ID 正确性 ---
    print("\n[2] 模态 ID 检查 ...")
    # 第一个样本的 mod_ids 应为 [0]*16 + [1]*16 + [2]*16
    expected_ids = (
        [0] * cfg.n_patches_per_modality +
        [1] * cfg.n_patches_per_modality +
        [2] * cfg.n_patches_per_modality
    )
    assert mod_ids[0].tolist() == expected_ids, (
        f"mod_ids 不匹配: {mod_ids[0].tolist()[:5]}... vs {expected_ids[:5]}..."
    )
    print(f"    mod_ids[0] 前 5: {mod_ids[0][:5].tolist()} (应全 0 = gravity)  ✓")
    print(f"    mod_ids[0] [16:21]: {mod_ids[0][16:21].tolist()} (应全 1 = accel)  ✓")
    print(f"    mod_ids[0] [32:37]: {mod_ids[0][32:37].tolist()} (应全 2 = gyro)  ✓")

    # --- 测试 3: 位置 ID 正确性 ---
    print("\n[3] 位置 ID 检查 ...")
    # 每个模态内 positions 应为 [0, 1, ..., 15]
    for m in range(3):
        start = m * cfg.n_patches_per_modality
        end = start + cfg.n_patches_per_modality
        expected_pos = list(range(cfg.n_patches_per_modality))
        assert positions[0, start:end].tolist() == expected_pos, (
            f"模态 {m} 的 positions 不匹配"
        )
    print(f"    每模态 positions = [0..{cfg.n_patches_per_modality-1}]  ✓")

    # --- 测试 4: Patch 内容正确性（核心测试） ---
    # 构造已知信号，检查 patch 是否包含正确的信号切片
    print("\n[4] Patch 内容正确性 ...")
    B2 = 1
    x_known = torch.zeros(B2, L, 9)
    # 在每个通道填入不同的常数，便于检查
    for ch in range(9):
        x_known[:, :, ch] = float(ch + 1)  # ch0=1.0, ch1=2.0, ..., ch8=9.0

    patches_k, _, _ = patcher(x_known)

    # 检查 gravity 模态 (channels 0,1,2 → 值 1,2,3)
    # patch 0 应覆盖原始信号 [0:64]（无 padding 在左侧）
    # patch 0 的内容应为 [1.0, 2.0, 3.0] 重复 64 次
    g_patch0 = patches_k[0, 0, :]  # (192,) = 64×3
    g_patch0_reshaped = g_patch0.reshape(64, 3)  # (P, 3_axes)
    # gravity channels: aGx=1.0, aGy=2.0, aGz=3.0
    assert torch.allclose(g_patch0_reshaped[:, 0], torch.ones(64)), (
        f"gravity patch 0 axis 0 应全为 1.0"
    )
    assert torch.allclose(g_patch0_reshaped[:, 1], torch.full((64,), 2.0)), (
        f"gravity patch 0 axis 1 应全为 2.0"
    )
    assert torch.allclose(g_patch0_reshaped[:, 2], torch.full((64,), 3.0)), (
        f"gravity patch 0 axis 2 应全为 3.0"
    )
    print(f"    gravity patch 0: axis 值 = {g_patch0_reshaped[0].tolist()} (应 [1,2,3])  ✓")

    # 检查 accel 模态 (channels 3,4,5 → 值 4,5,6)
    a_patch0 = patches_k[0, 16, :].reshape(64, 3)  # patch 16 = accel patch 0
    assert torch.allclose(a_patch0[:, 0], torch.full((64,), 4.0))
    assert torch.allclose(a_patch0[:, 1], torch.full((64,), 5.0))
    assert torch.allclose(a_patch0[:, 2], torch.full((64,), 6.0))
    print(f"    accel patch 0:   axis 值 = {a_patch0[0].tolist()} (应 [4,5,6])  ✓")

    # 检查 gyro 模态 (channels 6,7,8 → 值 7,8,9)
    w_patch0 = patches_k[0, 32, :].reshape(64, 3)  # patch 32 = gyro patch 0
    assert torch.allclose(w_patch0[:, 0], torch.full((64,), 7.0))
    assert torch.allclose(w_patch0[:, 1], torch.full((64,), 8.0))
    assert torch.allclose(w_patch0[:, 2], torch.full((64,), 9.0))
    print(f"    gyro patch 0:    axis 值 = {w_patch0[0].tolist()} (应 [7,8,9])  ✓")

    # --- 测试 5: Patch 内容与时序对应 ---
    # 用线性递增信号验证 patch 的时间偏移
    print("\n[5] 时间偏移正确性 ...")
    x_ramp = torch.zeros(1, L, 9)
    for ch in range(9):
        x_ramp[0, :, ch] = torch.arange(L, dtype=torch.float32)  # 0,1,2,...,511

    patches_r, _, _ = patcher(x_ramp)

    # gravity patch 0: 应覆盖 [0:64], 值 0..63
    g_p0 = patches_r[0, 0, :].reshape(64, 3)
    expected_start = 0
    expected_vals = torch.arange(expected_start, expected_start + 64, dtype=torch.float32)
    assert torch.allclose(g_p0[:, 0], expected_vals), (
        f"gravity patch 0 时间偏移错误: 期望起始 {expected_start}"
    )
    print(f"    gravity patch 0: 起始值={g_p0[0,0]:.0f}, 结束值={g_p0[-1,0]:.0f} "
          f"(应 0..63)  ✓")

    # gravity patch 1: 应覆盖 [32:96], 值 32..95
    g_p1 = patches_r[0, 1, :].reshape(64, 3)
    expected_start = 32
    expected_vals = torch.arange(expected_start, expected_start + 64, dtype=torch.float32)
    assert torch.allclose(g_p1[:, 0], expected_vals)
    print(f"    gravity patch 1: 起始值={g_p1[0,0]:.0f}, 结束值={g_p1[-1,0]:.0f} "
          f"(应 32..95)  ✓")

    # gravity patch 15 (最后一个): 起始于 15*32=480, 覆盖 [480:544]
    # 但原始信号只有 512 点, [512:544] 是 reflect padding
    # reflect: 位置 512→510, 513→509, ..., 543→481
    g_p15 = patches_r[0, 15, :].reshape(64, 3)
    # 前 32 点 [480:512]: 原始值 480..511
    assert torch.allclose(g_p15[:32, 0], torch.arange(480, 512, dtype=torch.float32))
    # 后 32 点 [512:544]: reflect of [510:478:-1] = 510,509,...,479
    expected_reflect = torch.arange(510, 478, -1, dtype=torch.float32)
    assert torch.allclose(g_p15[32:, 0], expected_reflect), (
        f"reflect padding 错误: 期望 {expected_reflect[:3].tolist()}, "
        f"得到 {g_p15[32:35, 0].tolist()}"
    )
    print(f"    gravity patch 15: 前32点 [480..511], 后32点 reflect [510..479]  ✓")

    # --- 测试 6: 往返一致性 (forward → patches_to_signal) ---
    print("\n[6] 往返一致性 (patch → signal → patch) ...")
    x_test = torch.randn(2, L, 9)
    patches_t, _, _ = patcher(x_test)
    x_recon = patcher.patches_to_signal(patches_t)

    # 非重叠区域应精确还原；重叠区域取均值，也应力接近
    # 检查整体误差
    max_err = (x_test - x_recon).abs().max().item()
    mean_err = (x_test - x_recon).abs().mean().item()
    print(f"    往返最大误差: {max_err:.6f}")
    print(f"    往返平均误差: {mean_err:.6f}")

    # 前 P-S=32 个点（无重叠）应精确还原
    err_non_overlap = (x_test[:, :32, :] - x_recon[:, :32, :]).abs().max().item()
    assert err_non_overlap < 1e-5, f"非重叠区误差过大: {err_non_overlap}"
    print(f"    非重叠区 [0:32] 误差: {err_non_overlap:.2e}  ✓")

    # --- 测试 7: batch 一致性 ---
    print("\n[7] Batch 一致性 ...")
    x_batch = torch.randn(8, L, 9)
    patches_b, mod_ids_b, pos_b = patcher(x_batch)
    # 所有样本的 mod_ids 和 positions 应相同
    for b in range(1, 8):
        assert torch.equal(mod_ids_b[0], mod_ids_b[b]), "mod_ids 应 batch 间一致"
        assert torch.equal(pos_b[0], pos_b[b]), "positions 应 batch 间一致"
    print(f"    8 个样本的 mod_ids/positions 全部一致  ✓")

    # --- 测试 8: 梯度可传播 ---
    print("\n[8] 梯度传播 ...")
    x_grad = torch.randn(1, L, 9, requires_grad=True)
    patches_g, _, _ = patcher(x_grad)
    loss = patches_g.sum()
    loss.backward()
    assert x_grad.grad is not None, "梯度未传播到输入"
    assert not torch.isnan(x_grad.grad).any(), "梯度含 NaN"
    print(f"    梯度正常传播到输入, grad norm = {x_grad.grad.norm():.4f}  ✓")

    print("\n[OK] ModalityPatcher 验证全部通过")


if __name__ == '__main__':
    _verify()
