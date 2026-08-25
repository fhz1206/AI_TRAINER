"""
models.diffusion — 扩散模型：图像生成 + 图像编辑适配

- DiffusionModel     : DDPM 风格文生图式无条件生成（噪声预测 ε-MSE 训练，逐步去噪采样）
- DiffusionEditModel : 扩散编辑适配架构——在扩散骨干上增加条件输入通道（原图拼接），
                       以「清晰图 vs 退化图」自监督配对训练，实现修复/编辑类任务；
                       推理用 SDEdit 思路：从加噪的源图出发条件化去噪。

两个类共用轻量 UNet 骨干（时间步嵌入 + 双层下采样上采样 + 瓶颈注意力）。
默认参数面向教学级 CPU 可训；image_size 建议 32 或 64（需能被 4 整除）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 时间步嵌入 ====================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(
            half, device=t.device).float() / max(half - 1, 1))
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


# ==================== 基础块 ====================
class ResBlock(nn.Module):
    """Conv-残差块 + 时间步注入（stride 由 conv1/skip 的卷积自身完成）"""

    def __init__(self, in_ch, out_ch, time_dim, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride)

    def forward(self, x, t_emb):
        h = F.silu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        identity = x if self.skip is None else self.skip(x)
        return F.silu(h + identity)


class BottleneckAttention(nn.Module):
    """瓶颈处自注意力：把特征图展平成 token 序列做标准多头注意力"""

    def __init__(self, channels, n_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x):
        b, c, hh, ww = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)          # (B, HW, C)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, hh, ww)


class DiffusionUNet(nn.Module):
    """
    轻量 UNet 噪声预测网络。
    in_channels=3 用于无条件生成；编辑适配版传 6（带噪目标图 + 条件图拼接）。
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=32,
                 image_size=32, attn_heads=4):
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError(f"image_size({image_size}) 需能被 4 整除")
        c = base_channels
        time_dim = c * 4
        self.time_mlp = TimeEmbedding(time_dim)

        self.stem = nn.Conv2d(in_channels, c, 3, padding=1)
        self.down1 = ResBlock(c, c * 2, time_dim, stride=2)       # /2
        self.down2 = ResBlock(c * 2, c * 4, time_dim, stride=2)   # /4
        self.mid1 = ResBlock(c * 4, c * 4, time_dim)
        self.mid_attn = BottleneckAttention(c * 4, attn_heads)
        self.mid2 = ResBlock(c * 4, c * 4, time_dim)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.up_block2 = ResBlock(c * 4, c * 2, time_dim)          # 拼接 skip 后通道翻倍
        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.up_block1 = ResBlock(c * 2, c, time_dim)

        self.head = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, out_channels, 3, padding=1),  # 恒定输出目标通道数的噪声预测
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h0 = self.stem(x)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(h1, t_emb)
        m = self.mid2(self.mid_attn(self.mid1(h2, t_emb)), t_emb)
        u2 = self.up_block2(torch.cat([self.up2(m), h1], dim=1), t_emb)
        u1 = self.up_block1(torch.cat([self.up1(u2), h0], dim=1), t_emb)
        return self.head(u1)


def _linear_schedule(timesteps):
    betas = torch.linspace(1e-4, 0.02, timesteps)
    alphas = 1.0 - betas
    ac = torch.cumprod(alphas, dim=0)
    return betas, ac


# ==================== 无条件生成 ====================
class DiffusionModel(nn.Module):
    """DDPM 图像生成：训练即「猜噪声」，采样即「逐步去噪」"""

    def __init__(self, image_size=32, base_channels=32, num_timesteps=300,
                 attn_heads=4):
        super().__init__()
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        self.net = DiffusionUNet(in_channels=3, base_channels=base_channels,
                                 image_size=image_size, attn_heads=attn_heads)
        betas, ac = _linear_schedule(num_timesteps)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', ac)

    def q_sample(self, x0, t, noise):
        """前向加噪：x_t = √ᾱ·x0 + √(1-ᾱ)·ε"""
        sa = self.alphas_cumprod[t].sqrt()[:, None, None, None]
        som = (1 - self.alphas_cumprod[t]).sqrt()[:, None, None, None]
        return sa * x0 + som * noise

    def forward(self, x0):
        """训练一步：返回噪声预测 MSE 损失"""
        b = x0.size(0)
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.net(x_t, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, n=1, device='cpu'):
        """从纯噪声出发迭代去噪，返回 (n,3,H,W)，值域 [-1,1]"""
        self.eval()
        size = self.image_size
        x = torch.randn(n, 3, size, size, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.net(x, t)
            alpha = 1 - self.betas[i]
            ac = self.alphas_cumprod[i]
            mean = (x - (1 - alpha) / (1 - ac).sqrt() * eps) / alpha.sqrt()
            if i > 0:
                ac_prev = self.alphas_cumprod[i - 1]
                var = (1 - ac_prev) / (1 - ac) * self.betas[i]
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)


# ==================== 编辑适配 ====================
class DiffusionEditModel(nn.Module):
    """
    扩散编辑适配架构：UNet 输入通道扩为「带噪目标图 + 条件图」拼接。
    训练数据自动构造退化对（模糊+降采样），学会由劣化输入恢复/改写目标图。
    """

    def __init__(self, image_size=32, base_channels=32, num_timesteps=300,
                 attn_heads=4):
        super().__init__()
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        self.net = DiffusionUNet(in_channels=6, base_channels=base_channels,
                                 image_size=image_size, attn_heads=attn_heads)
        betas, ac = _linear_schedule(num_timesteps)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', ac)

    @staticmethod
    @torch.no_grad()
    def make_condition(x0):
        """构造编辑条件图：模糊 + 降采样再放大（模拟低质输入）"""
        blurred = F.avg_pool2d(x0, 3, stride=1, padding=1)
        small = F.interpolate(blurred, scale_factor=0.25, mode='nearest')
        return F.interpolate(small, size=x0.shape[-2:], mode='nearest')

    def q_sample(self, x0, t, noise):
        sa = self.alphas_cumprod[t].sqrt()[:, None, None, None]
        som = (1 - self.alphas_cumprod[t]).sqrt()[:, None, None, None]
        return sa * x0 + som * noise

    def forward(self, cond, x0):
        """训练一步：cond 为条件图（[-1,1]），x0 为目标清晰图"""
        b = x0.size(0)
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.net(torch.cat([x_t, cond], dim=1), t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def edit(self, source, strength=0.5):
        """
        SDEdit 式编辑：把源图加噪到 strength 对应强度，再条件化去噪回干净图。
        source: (n,3,H,W) 值域 [-1,1]；strength∈(0,1) 越大改动越大。
        """
        self.eval()
        n = source.size(0)
        device = source.device
        t_max = max(int(self.num_timesteps * float(strength)), 1)
        t0 = torch.full((n,), t_max - 1, device=device, dtype=torch.long)
        x = self.q_sample(source, t0, torch.randn_like(source))

        for i in reversed(range(t_max)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.net(torch.cat([x, source], dim=1), t)
            alpha = 1 - self.betas[i]
            ac = self.alphas_cumprod[i]
            mean = (x - (1 - alpha) / (1 - ac).sqrt() * eps) / alpha.sqrt()
            if i > 0:
                ac_prev = self.alphas_cumprod[i - 1]
                var = (1 - ac_prev) / (1 - ac) * self.betas[i]
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)
