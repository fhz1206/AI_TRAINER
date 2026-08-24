"""
architectures.moe — Mixture of Experts 层（修复版）

原 model.py 的实现存在广播形状 bug（mask 二维索引导致 output[mask] 形状不匹配），
本模块重写为展平索引的安全实现，语义与原设计一致：Top-K 路由 + 负载均衡辅助损失。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """Mixture of Experts：门控路由选择 Top-K 专家，仅激活被选中的网络以节省计算"""

    def __init__(self, d_model, d_ff, num_experts=4, top_k=2,
                 dropout=0.1, aux_loss_weight=0.02):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: (B, S, D)
        返回: (output (B, S, D), aux_loss 标量张量)
        """
        b, s, d = x.shape
        x_flat = x.reshape(-1, d)                      # (N, D)
        gate_logits = self.gate(x_flat)                # (N, E)
        gate_weights = F.softmax(gate_logits, dim=-1)

        topk_w, topk_i = torch.topk(gate_weights, self.top_k, dim=-1)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-8)

        output = torch.zeros_like(x_flat)
        flat_out = output.view(-1) if False else None  # noqa: 保持形状提示

        for e_idx, expert in enumerate(self.experts):
            # 展平索引：token 维 × topk 维，避免二维布尔掩码广播问题
            mask_e = (topk_i == e_idx)                 # (N, K)
            token_idx, slot_idx = mask_e.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_in = x_flat[token_idx]              # (M, D)
            expert_out = expert(expert_in)             # (M, D)
            w = topk_w[token_idx, slot_idx].unsqueeze(-1)
            output.index_add_(0, token_idx, expert_out * w)

        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            usage = (topk_i.unsqueeze(-1) == torch.arange(
                self.num_experts, device=x.device)).any(dim=1).float().mean(dim=0)
            target = torch.ones_like(usage) / self.num_experts
            aux_loss = F.mse_loss(usage, target) * self.aux_loss_weight

        return output.view(b, s, d), aux_loss
