"""
architectures — 模型架构积木库

提供可插拔的注意力机制、MoE、标准积木块；上层模型（models 包）从这里取件组装。

快速上手：
    from architectures import build_attention, TransformerBlock
    blk = TransformerBlock(d_model=256, n_heads=8, d_ff=512, attn_name='flash')
"""
from .attention import (
    ATTENTION_REGISTRY,
    register_attention,
    available_attentions,
    build_attention,
    BaseSelfAttention,
    FullAttention,
    FlashAttention,
    LinearAttention,
)
from .moe import MoELayer
from .blocks import TransformerBlock, TransformerEncoderBlock, ViTBlock

__all__ = [
    'ATTENTION_REGISTRY', 'register_attention', 'available_attentions',
    'build_attention', 'BaseSelfAttention',
    'FullAttention', 'FlashAttention', 'LinearAttention',
    'MoELayer', 'TransformerBlock', 'TransformerEncoderBlock', 'ViTBlock',
]
