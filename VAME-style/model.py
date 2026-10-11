import torch
from torch import nn
import torch.nn.functional as F

'''
输入 x: (B, T, 6)   # [ax, ay, az, wx, wy, wz]
  ↓ Encoder: 因果 Conv1d 堆叠，感受野 ~25 帧（0.25s）
h: (B, T, hidden)   # 逐帧，第 t 帧只看过 x[:, t-24:t+1]
  ↓ Lambda: 逐帧 Linear → mu (B,T,z), logvar (B,T,z)
  ↓ 重参数化: z = mu + σ·ε
  ↓ Decoder: 逐帧 MLP
x̂: (B, T, 5)        # 用 z_t 重构当前帧
损失 = MSE(x, x̂) + β·KL(mu, logvar)   # 都是逐帧求平均

'''

#1d因果卷积算法
class CausalConv1d(nn.Module): 
    def __init__(self, in_ch, out_ch, kernel_size): 
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size) #定义感受野为每帧侧

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))  # 在时间维度前面填充 pad 个 0
        return self.conv(x)

#因果卷积编码器
class Encoder(nn.Module):

    def __init__(self, in_features=5, channels=(32, 64, 64), kernel_size=9):
        
        #通道数5、卷积核大小9；三层网络（32，64，64）
        super().__init__()
        dims = [in_features, *channels]
        self.convs = nn.ModuleList()  #在Encoder类中创建一个ModuleList，用于存储卷积层，在forward中反复调用。
        self.receptive_field = 1 + len(channels) * (kernel_size - 1)    #显式保存感受野大小：感受野 = 通道数 ×（卷积核大小-1） + 1
        for i in range(len(dims) - 1):
            self.convs.append(CausalConv1d(dims[i], dims[i + 1], kernel_size))

    def forward(self, x):
        # x: (B, T, 5)  ← 注意 Conv1d 要 (B, C, T)，需要 transpose

        x = x.transpose(1, 2)  # (B, 5, T)
        for conv in self.convs:
            x = F.relu(conv(x))  #卷积层外套ReLU激活函数

        return x.transpose(1, 2) #重新转置回来



class Lambda(nn.Module):
    """逐帧计算潜变量分布参数并采样。"""

    def __init__(self, hidden_size=64, z_dim=6):
        super().__init__()

        self.hidden_to_mean = nn.Linear(hidden_size, z_dim)
        self.hidden_to_logvar = nn.Linear(hidden_size, z_dim)

    
    def forward(self, h):
        # h: (B, T, hidden_size)
        mu = self.hidden_to_mean(h)        
        logvar = self.hidden_to_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class Decoder(nn.Module):

    def __init__(self, z_dim=6, out_features=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim,64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )

    def forward(self, z):
        # z: (B, T, z_dim)
        return self.mlp(z)  # (B, T, out_features)
    
# 模型训练全程封装
class PoseVAE(nn.Module):
    def __init__(self, in_features=6, z_dim=6):
        super().__init__()

        self.encoder = Encoder(in_features=in_features)
        self.lmbda = Lambda(hidden_size=64, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, out_features=in_features)
        self.receptive_field = self.encoder.receptive_field
        
    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.lmbda(h)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar



#损失函数

def reconstruction_loss(x_recon, x):
    recon_loss = F.mse_loss(x_recon, x)
    return recon_loss

def kl_loss(mu, logvar):
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))
    return kl
