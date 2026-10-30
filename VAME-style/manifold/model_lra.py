import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """因果 1D 卷积：只在时间轴左侧 pad。"""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, pad_mode="replicate"):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, (self.pad, 0), mode=self.pad_mode)
        return self.conv(x)


class LRAEncoder(nn.Module):
    """
    LRA 编码器：从输入序列提取初始条件 z_0。
    先用 CausalConv 做局部特征提取，再全局池化到 z_0。
    输入 [B, T, 6] -> 输出 [B, z_dim]
    """

    def __init__(
        self,
        in_features=6,
        cnn_channels=(32, 64, 64),
        kernel_sizes=(7, 5, 5),
        z_dim=6,
        pad_mode="replicate",
    ):
        super().__init__()
        assert len(cnn_channels) == len(kernel_sizes)

        dims = [in_features] + list(cnn_channels)
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            self.convs.append(CausalConv1d(dims[i], dims[i + 1], k, pad_mode=pad_mode))

        self.to_z0 = nn.Linear(cnn_channels[-1], z_dim)
        self.receptive_field = 1 + sum(k - 1 for k in kernel_sizes)

    def forward(self, x):
        h = x.transpose(1, 2)  # [B, C, T]
        for conv in self.convs:
            h = F.relu(conv(h))
        h = h.transpose(1, 2)  # [B, T, C]
        # 全局均值池化：将整个序列压缩为单一初始条件
        h_mean = h.mean(dim=1)  # [B, C]
        z_0 = self.to_z0(h_mean)  # [B, z_dim]
        return z_0


class LinearRecurrence(nn.Module):
    """
    全局共享线性递推：z_{t+1} = z_t @ A^T + b
    A 用正交初始化以促进振荡模态（如周期性行走）。
    """

    def __init__(self, z_dim=6):
        super().__init__()
        self.z_dim = z_dim
        
        # 正交初始化：随机矩阵做 QR 分解，再轻微缩放到边缘稳定
        q, _ = torch.linalg.qr(torch.randn(z_dim, z_dim))
        self.A = nn.Parameter(q * 0.95)
        self.b = nn.Parameter(torch.zeros(z_dim))

    def forward(self, z_0, T):
        """
        z_0: [B, z_dim]
        返回 z: [B, T, z_dim]
        """
        z_list = [z_0]
        z_t = z_0
        for _ in range(T - 1):
            z_t = torch.matmul(z_t, self.A.t()) + self.b
            z_list.append(z_t)
        return torch.stack(z_list, dim=1)  # [B, T, z_dim]


class Decoder(nn.Module):
    """
    逐帧 MLP 解码器：z [B, T, z_dim] -> out [B, T, 6]
    与原始 Decoder 保持一致。
    """

    def __init__(self, z_dim=6, out_features=6, hidden=(64, 128)):
        super().__init__()
        layers = []
        prev = z_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        return self.mlp(z)


class PoseLRA(nn.Module):
    """Linearly Recurrent Autoencoder for IMU behavior."""

    def __init__(
        self,
        in_features=6,
        z_dim=6,
        cnn_channels=(32, 64, 64),
        kernel_sizes=(7, 5, 5),
        decoder_hidden=(64, 128),
        pad_mode="replicate",
        **kwargs,  # 兼容旧 config 中的 gru_hidden, gru_layers, dynamics_hidden 等
    ):
        super().__init__()

        self.encoder = LRAEncoder(
            in_features=in_features,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            z_dim=z_dim,
            pad_mode=pad_mode,
        )

        self.recurrence = LinearRecurrence(z_dim=z_dim)

        self.decoder = Decoder(
            z_dim=z_dim,
            out_features=in_features,
            hidden=decoder_hidden,
        )

        self.receptive_field = self.encoder.receptive_field
        self.z_dim = z_dim

    def encode(self, x):
        z_0 = self.encoder(x)
        T = x.size(1)
        z = self.recurrence(z_0, T)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)  # [B, T, z_dim]
        recon = self.decode(z)  # [B, T, 6]
        
        # z_pred: 与 ae_loss 兼容的动力学预测
        # z_pred[:, t] 表示对 z_{t+1} 的预测，整体错一位
        z_pred = torch.zeros_like(z)
        if z.size(1) > 1:
            z_pred[:, :-1] = torch.matmul(z[:, :-1], self.recurrence.A.t()) + self.recurrence.b
            z_pred[:, -1] = z[:, -1]  # 最后一帧无预测，复制自身
        return recon, z, z_pred


def lra_loss(recon, target, z, z_pred, smooth_weight=0.01, dyn_weight=0.1,
             A_reg=0.001, A=None):
    """
    LRA 损失 = 重构 + 时序平滑 + 动态一致性 + 正交正则
    
    注：在纯 LRA 中，dyn_loss 理论上趋近于 0（因为 z 本身由 A 递推生成），
        可作为模型自检指标；若显著大于 0，说明递推与编码器输出不一致。
        
    A_reg: 正交正则权重，约束 A^T A ≈ I，使特征值模长接近 1（边缘稳定）。
    """
    rec_loss = F.mse_loss(recon, target)
    smooth_loss = F.mse_loss(z[:, 1:], z[:, :-1])
    dyn_loss = F.mse_loss(z_pred[:, :-1], z[:, 1:])

    # A 矩阵正交正则：鼓励特征值在单位圆上（支持振荡，防止爆炸/坍缩）
    a_reg_loss = torch.tensor(0.0, device=z.device)
    if A is not None and A_reg > 0:
        I = torch.eye(A.size(0), device=A.device)
        a_reg_loss = torch.norm(A.t() @ A - I, p='fro') ** 2

    total = rec_loss + smooth_weight * smooth_loss + dyn_weight * dyn_loss + A_reg * a_reg_loss

    return total, {
        "rec": rec_loss.item(),
        "smooth": smooth_loss.item(),
        "dyn": dyn_loss.item(),
        "a_reg": a_reg_loss.item(),
    }