import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """因果 1D 卷积：只在时间轴左侧补 pad，不看未来帧。"""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, pad_mode="replicate"):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, (self.pad, 0), mode=self.pad_mode)
        return self.conv(x)


class CausalBranchEncoder(nn.Module):
    """一个分支的因果编码器，输出逐帧 mu/logvar。"""

    def __init__(self, in_features, channels, kernel_sizes, z_dim, pad_mode="replicate"):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        dims = [in_features] + list(channels)
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            self.convs.append(
                CausalConv1d(dims[i], dims[i + 1], k, pad_mode=pad_mode)
            )
        self.receptive_field = 1 + sum(k - 1 for k in kernel_sizes)
        self.to_mu = nn.Linear(channels[-1], z_dim)
        self.to_logvar = nn.Linear(channels[-1], z_dim)

    def forward(self, x):
        h = x.transpose(1, 2)
        for conv in self.convs:
            h = F.relu(conv(h))
        h = h.transpose(1, 2)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar


class PerFrameDecoder(nn.Module):
    """逐帧 MLP 解码器：z (B,T,Z) -> out (B,T,C)。"""

    def __init__(self, z_dim, out_features, hidden=(64, 128)):
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


class PoseVAE(nn.Module):
    """
    2 分支频率分解因果 VAE。
    latent 顺序固定为 branch_configs 里的顺序，例如 [slow, fast]。
    """

    def __init__(
        self,
        in_features=6,
        z_dim=9,
        branch_configs=None,
        decoder_hidden=(64, 128),
        pad_mode="replicate",
    ):
        super().__init__()
        if branch_configs is None:
            branch_configs = [
                {
                    "name": "slow",
                    "z_dim": 3,
                    "channels": [32, 64, 64, 64],
                    "kernel_sizes": [33, 33, 33, 32],
                },
                {
                    "name": "fast",
                    "z_dim": 6,
                    "channels": [32, 64, 64],
                    "kernel_sizes": [5, 5, 6],
                },
            ]

        self.branch_order = [b["name"] for b in branch_configs]
        self.branch_configs = branch_configs
        self.pad_mode = pad_mode

        # 编码器
        self.encoders = nn.ModuleDict()
        total_z = 0
        for cfg in branch_configs:
            self.encoders[cfg["name"]] = CausalBranchEncoder(
                in_features=in_features,
                channels=cfg["channels"],
                kernel_sizes=cfg["kernel_sizes"],
                z_dim=cfg["z_dim"],
                pad_mode=pad_mode,
            )
            total_z += cfg["z_dim"]

        assert total_z == z_dim, f"分支 z_dim 之和 {total_z} != 总 z_dim {z_dim}"

        # 解码器：每个分支一个
        self.decoders = nn.ModuleDict()
        for cfg in branch_configs:
            self.decoders[cfg["name"]] = PerFrameDecoder(
                z_dim=cfg["z_dim"],
                out_features=in_features,
                hidden=decoder_hidden,
            )

        self.receptive_field = max(
            self.encoders[name].receptive_field for name in self.branch_order
        )

    def encode(self, x, sample=True):
        mus, logvars, zs = [], [], []
        for name in self.branch_order:
            mu, logvar = self.encoders[name](x)
            mus.append(mu)
            logvars.append(logvar)
            if sample:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                zs.append(mu + eps * std)
            else:
                zs.append(mu)
        mu = torch.cat(mus, dim=-1)
        logvar = torch.cat(logvars, dim=-1)
        z = torch.cat(zs, dim=-1)
        return z, mu, logvar

    def decode(self, z):
        """
        z: (B, T, z_dim)
        返回 dict:
            slow / fast : (B, T, 6)  各分支重建
            recon       : (B, T, 6)  相加后的总重建
        """
        recon = {}
        start = 0
        for cfg in self.branch_configs:
            name = cfg["name"]
            z_dim = cfg["z_dim"]
            z_b = z[..., start : start + z_dim]
            recon[name] = self.decoders[name](z_b)
            start += z_dim

        # 总重建 = 各分支相加
        recon["recon"] = sum(recon[name] for name in self.branch_order)
        return recon

    def forward(self, x, sample=True):
        z, mu, logvar = self.encode(x, sample=sample)
        out = self.decode(z)
        return out["recon"], mu, logvar

    def forward_branches(self, x, sample=True):
        z, mu, logvar = self.encode(x, sample=sample)
        recon_dict = self.decode(z)
        return recon_dict, mu, logvar


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))


def reconstruction_loss(pred, target):
    return F.mse_loss(pred, target)