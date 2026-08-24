"""
architectures.blocks — 积木式网络块

所有块都通过 build_attention() 装配注意力组件：
- TransformerBlock  : 因果语言模型块（自注意力 + 可选 MoE/普通 FFN），文本与多模态共用
- TransformerEncoderBlock : 双向注意力块（无因果掩码），ViT 使用
- ViTBlock          = TransformerEncoderBlock 的别名（Patch embedding 后的图像块）
"""
import torch
import torch.nn as nn

from .attention import build_attention
from .moe import MoELayer


class TransformerBlock(nn.Module):
    """
    Decoder-only 语言模型积木块（因果注意力）。
    attn_name 决定注意力实现（full/flash/linear...）；use_moe=True 时 FFN 换成 MoE。

    forward 返回 (hidden, aux_loss)；need_weights=True 时额外返回权重便于可视化，
    此时内部临时切换到支持权重的路径（flash 会委托 full 计算）。
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 attn_name='flash', use_moe=False,
                 moe_experts=4, moe_top_k=2, aux_loss_weight=0.02):
        super().__init__()
        self.attn_name = (attn_name or 'flash').lower()
        self.attn = build_attention(self.attn_name, d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.use_moe = use_moe
        if use_moe:
            self.ffn = MoELayer(d_model, d_ff, num_experts=moe_experts,
                                top_k=moe_top_k, dropout=dropout,
                                aux_loss_weight=aux_loss_weight)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

    def forward(self, x, key_padding_mask=None, need_weights=False):
        h = self.norm1(x)
        attn_out, weights = self.attn(h, causal=True,
                                      key_padding_mask=key_padding_mask,
                                      need_weights=need_weights)
        x = x + self.drop(attn_out)

        h = self.norm2(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(h)
            x = x + self.drop(ffn_out)
        else:
            x = x + self.drop(self.ffn(h))
            aux_loss = torch.tensor(0.0, device=x.device)

        if need_weights:
            return x, weights
        return x, aux_loss


class TransformerEncoderBlock(nn.Module):
    """
    双向（非因果）编码器积木块，用于 ViT 等图像理解任务。
    图像 patch 序列通常等长无填充，暂不引入 padding 掩码。
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, attn_name='flash'):
        super().__init__()
        self.attn_name = (attn_name or 'flash').lower()
        self.attn = build_attention(self.attn_name, d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, causal=False,
                                key_padding_mask=key_padding_mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ViT 块与通用编码器块同构，提供别名增强语义可读性
ViTBlock = TransformerEncoderBlock
