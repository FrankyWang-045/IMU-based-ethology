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


class Encoder(nn.Module):
    """
    1D-CNN + 双向 GRU 编码器。
    输入 [B, T, 6] -> 输出 [B, T, z_dim]
    """

    def __init__(
        self,
        in_features=6,
        cnn_channels=(32, 64, 64),
        kernel_sizes=(7, 5, 5),
        gru_hidden=64,
        gru_layers=1,
        z_dim=9,
        pad_mode="replicate",
    ):
        super().__init__()
        assert len(cnn_channels) == len(kernel_sizes)

        dims = [in_features] + list(cnn_channels)
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            self.convs.append(CausalConv1d(dims[i], dims[i + 1], k, pad_mode=pad_mode))

        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.to_z = nn.Linear(gru_hidden * 2, z_dim)
        self.receptive_field = 1 + sum(k - 1 for k in kernel_sizes)

    def forward(self, x):
        h = x.transpose(1, 2)  # [B, C, T]
        for conv in self.convs:
            h = F.relu(conv(h))
        h = h.transpose(1, 2)  # [B, T, C]
        h, _ = self.gru(h)
        return self.to_z(h)


class Decoder(nn.Module):
    """
    逐帧 MLP 解码器：z [B, T, z_dim] -> out [B, T, 6]
    """

    def __init__(self, z_dim=9, out_features=6, hidden=(64, 128)):
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


class DynamicsMLP(nn.Module):
    """学习 z_t -> z_{t+1} 的动力学模型。"""

    def __init__(self, z_dim=9, hidden=(64, 64)):
        super().__init__()
        layers = []
        prev = z_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, z_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        return self.mlp(z)


class PoseAE(nn.Module):
    """单分支 6D -> z_dim AutoEncoder。"""

    def __init__(
        self,
        in_features=6,
        z_dim=9,
        cnn_channels=(32, 64, 64),
        kernel_sizes=(7, 5, 5),
        gru_hidden=64,
        gru_layers=1,
        decoder_hidden=(64, 128),
        dynamics_hidden=(64, 64),
        pad_mode="replicate",
    ):
        super().__init__()

        self.encoder = Encoder(
            in_features=in_features,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            z_dim=z_dim,
            pad_mode=pad_mode,
        )

        self.decoder = Decoder(
            z_dim=z_dim,
            out_features=in_features,
            hidden=decoder_hidden,
        )

        self.dynamics = DynamicsMLP(
            z_dim=z_dim,
            hidden=dynamics_hidden,
        )

        self.receptive_field = self.encoder.receptive_field

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        z_next_pred = self.dynamics(z)
        return recon, z, z_next_pred


def ae_loss(recon, target, z, z_pred, smooth_weight=0.01, dyn_weight=0.1, curve_weight=0.1):
    """
    总损失 = 重构 + 时序平滑 + 动态可预测 + 曲率惩罚。
    """
    rec_loss = F.mse_loss(recon, target)
    smooth_loss = F.mse_loss(z[:, 1:], z[:, :-1])
    dyn_loss = F.mse_loss(z_pred[:, :-1], z[:, 1:])

    # 曲率损失：惩罚 latent 轨迹的二阶差分（加速度）
    if z.shape[1] >= 3:
        second_diff = z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]
        curve_loss = (second_diff ** 2).mean()
    else:
        curve_loss = torch.tensor(0.0, device=z.device)

    total = rec_loss + smooth_weight * smooth_loss + dyn_weight * dyn_loss + curve_weight * curve_loss
    return total, {
        "rec": rec_loss.item(),
        "smooth": smooth_loss.item(),
        "dyn": dyn_loss.item(),
        "curve": curve_loss.item(),
    }