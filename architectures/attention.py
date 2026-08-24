"""
architectures.attention — 可插拔注意力机制集合

像搭积木一样按名字装配注意力：
    from architectures import build_attention
    attn = build_attention('flash', d_model=256, n_heads=8, dropout=0.1)

内置类型：
- full  : 显式 QK^T -> mask -> softmax -> ·V，可输出注意力权重（教学可视化）
- flash : PyTorch scaled_dot_product_attention 内核融合实现（默认，省显存提速）
- linear: 线性注意力（核函数近似），O(S) 复杂度，超长序列友好

扩展新注意力：继承 BaseSelfAttention 并用 @register_attention('名字') 注册即可，
训练器/前端无需任何改动。
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_REGISTRY = {}


def register_attention(name):
    """注册装饰器：把注意力类挂入全局注册表，供积木式装配"""
    def deco(cls):
        ATTENTION_REGISTRY[name.lower()] = cls
        return cls
    return deco


def available_attentions():
    return sorted(ATTENTION_REGISTRY)


def build_attention(name, d_model, n_heads, dropout=0.1):
    cls = ATTENTION_REGISTRY.get((name or 'flash').lower())
    if cls is None:
        raise ValueError(f"未知注意力类型: {name}，可选: {available_attentions()}")
    return cls(d_model=d_model, n_heads=n_heads, dropout=dropout)


def _split_heads(x, n_heads):
    b, s, d = x.shape
    return x.view(b, s, n_heads, d // n_heads).transpose(1, 2)


def _combine_heads(x):
    b, h, s, dh = x.shape
    return x.transpose(1, 2).contiguous().view(b, s, h * dh)


class BaseSelfAttention(nn.Module):
    """
    注意力基类：统一 QKV 投影与多头拆分，子类只实现核心计算。
    forward 返回 (输出, 注意力权重或None)；不支持权重的实现返回 None。
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def _project(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        return (_split_heads(q, self.n_heads),
                _split_heads(k, self.n_heads),
                _split_heads(v, self.n_heads))

    def _out(self, x):
        return self.proj(x)

    def forward(self, x, causal=True, key_padding_mask=None, need_weights=False):
        raise NotImplementedError


@register_attention('full')
class FullAttention(BaseSelfAttention):
    """标准缩放点积注意力：矩阵运算全程显式，便于展示注意力热力图"""

    def forward(self, x, causal=True, key_padding_mask=None, need_weights=False):
        q, k, v = self._project(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal:
            s = scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(s, s, dtype=torch.bool, device=scores.device), diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :]  # (B,1,1,S)
            scores = scores.masked_fill(pad, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)
        out = torch.matmul(weights, v)
        out = self._out(_combine_heads(out))
        return (out, weights) if need_weights else (out, None)


@register_attention('flash')
class FlashAttention(BaseSelfAttention):
    """
    Flash Attention 风格实现：调用 PyTorch 内核融合的 scaled_dot_product_attention，
    不实例化完整注意力矩阵，显存占用低、速度快（本平台默认选项）。
    """

    def forward(self, x, causal=True, key_padding_mask=None, need_weights=False):
        if need_weights:
            # 需要可视化权重时退回 full 实现
            return FullAttention.forward(self, x, causal, key_padding_mask, True)
        q, k, v = self._project(x)
        attn_mask = None
        is_causal = bool(causal)
        if key_padding_mask is not None:
            # 有填充时构造加性掩码（因果 + 屏蔽pad），交由统一路径处理
            s = x.size(1)
            base = torch.zeros(s, s, device=x.device)
            if causal:
                base = base.masked_fill(
                    torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), 1),
                    float('-inf'))
            pad = key_padding_mask[:, None, None, :].to(x.dtype)  # (B,1,1,S)
            add_pad = torch.zeros_like(pad).masked_fill(pad > 0.5, float('-inf'))
            attn_mask = base[None] + add_pad
            is_causal = False
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=0.0)
        return self._out(_combine_heads(out)), None


@register_attention('linear')
class LinearAttention(BaseSelfAttention):
    """
    线性注意力：用核函数 φ(x)=elu(x)+1 近似 softmax，
    复杂度从 O(S²) 降到 O(S)，适合超长序列；代价是表达能力略降、无显式权重。
    """

    def forward(self, x, causal=True, key_padding_mask=None, need_weights=False):
        q, k, v = self._project(x)
        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0
        eps = 1e-6

        if key_padding_mask is not None:
            keep = (~key_padding_mask).to(phi_k.dtype)[:, None, :, None]
            phi_k = phi_k * keep

        if causal:
            # 前缀和技巧：num_t = φq_t·Σ_{i<=t} φk_iᵀv_i ; den_t = φq_t·Σ_{i<=t} φk_i
            kv_cum = torch.cumsum(phi_k.unsqueeze(-1) * v.unsqueeze(-2), dim=2)
            k_cum = torch.cumsum(phi_k, dim=2)
            num = torch.einsum('bhsd,bhsde->bhse', phi_q, kv_cum)
            den = torch.einsum('bhsd,bhsd->bhs', phi_q, k_cum).unsqueeze(-1)
            out = num / (den + eps)
        else:
            kv = torch.einsum('bhsd,bhse->bhde', phi_k, v)
            num = torch.einsum('bhsd,bhde->bhse', phi_q, kv)
            den = torch.einsum('bhsd,bhd->bhs', phi_q,
                               phi_k.sum(dim=2)).unsqueeze(-1)
            out = num / (den + eps)

        return self._out(_combine_heads(out)), None
