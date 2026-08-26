"""
model_zoo.text_cls — 语言分类模型（字符级 Transformer 编码器）

与生成式模型同源的积木式架构，差异点：
- 双向注意力编码（非因果），CLS 位置输出接分类头
- 支持 attention_plan 混合注意力逐层装配 / attention_type 统一类型
"""
import torch
import torch.nn as nn

from architectures.blocks import TransformerEncoderBlock
from architectures.attention import resolve_attention_plan


class TextClassifier(nn.Module):
    """文本分类：嵌入 → N 层双向注意力编码块 → CLS 池化 → 分类头"""

    def __init__(self, vocab_size=5000, num_classes=2, d_model=128,
                 n_layers=4, n_heads=4, d_ff=256, max_seq_len=64,
                 dropout=0.1, pad_token_id=0, attention_type='flash',
                 attention_plan=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.attention_type = (attention_type or 'flash').lower()

        # 逐层混合注意力计划（缺失时回退统一类型）
        self.attention_plan = resolve_attention_plan(
            attention_plan, n_layers, default=self.attention_type)
        if len(set(self.attention_plan)) > 1:
            print(f"[TextClassifier] 混合注意力计划({n_layers}层): "
                  f"{self.attention_plan}")
            self.attention_summary = '混合[' + '→'.join(self.attention_plan) + ']'
        else:
            self.attention_summary = self.attention_plan[0]

        self.token_emb = nn.Embedding(vocab_size, d_model,
                                      padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_len + 1, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, attn_name=self.attention_plan[i])
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx] = 0
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        """
        input_ids: (B, S)
        返回类别 logits (B, num_classes)
        """
        b, s = input_ids.size()
        if s > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            s = self.max_seq_len

        x = self.token_emb(input_ids)
        pos = torch.arange(s, device=input_ids.device)
        x = x + self.pos_emb(pos)[None]
        x = self.drop(x)

        pad_mask = (input_ids == self.pad_token_id)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        x = self.norm(x)

        # CLS 式均值池化（屏蔽 padding 位）
        keep = (~pad_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        return self.head(pooled)
