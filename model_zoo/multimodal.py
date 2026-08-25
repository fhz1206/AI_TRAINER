"""
models.multimodal — 多模态单流模型（Decoder-only）

单流（single-stream）设计：图像 patch 与文本 token 映射到同一 d_model 空间，
拼成一条序列交给同一个因果 Transformer 建模——与 GPT-4V / LLaVA 类模型同构的简化版。

输入输出：
- forward(pixel_values, text_ids): 图文拼接序列，返回每步 logits（文本部分用于下一 token 预测）
- generate(image, prompt_ids, max_new_tokens): 给图 + 文本前缀，自回归续写

注意力可插拔（attention_type: full/flash/linear，默认 flash），Web 端积木式选择。
"""
import torch
import torch.nn as nn

from architectures.blocks import TransformerBlock


class MultiModalSingleStream(nn.Module):
    """图文单流 Decoder-only 模型"""

    def __init__(self, vocab_size=1000, image_size=32, patch_size=8, in_channels=3,
                 d_model=256, n_layers=6, n_heads=8, d_ff=512,
                 max_seq_len=128, dropout=0.1, pad_token_id=0,
                 use_moe=False, moe_experts=4, moe_top_k=2,
                 attention_type='flash'):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size({image_size}) 必须能被 patch_size({patch_size}) 整除")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.num_patches = (image_size // patch_size) ** 2
        self.attention_type = (attention_type or 'flash').lower()

        # 图像侧：patch 切分投影 + 可学习模态位置嵌入
        self.patch_embed = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size, stride=patch_size)
        self.img_pos = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        # 文本侧
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.txt_pos = nn.Embedding(max_seq_len - self.num_patches, d_model)
        self.drop = nn.Dropout(dropout)

        # 单流主干：一条因果注意力链同时处理图文 token
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
                attn_name=self.attention_type,
                use_moe=use_moe, moe_experts=moe_experts, moe_top_k=moe_top_k,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.trunc_normal_(self.img_pos, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx] = 0
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pixel_values, text_ids):
        """
        pixel_values: (B, C, H, W)，值域 [-1,1]
        text_ids:     (B, S_txt)
        返回文本位置的 logits (B, S_txt, vocab)。
        图像 token 位于序列最前，文本 token 通过因果注意力读取全部图像上下文。
        """
        b = pixel_values.size(0)
        patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)  # (B,N,D)
        img_tokens = patches + self.img_pos

        s_txt = min(text_ids.size(1), self.txt_pos.num_embeddings)
        txt_ids = text_ids[:, :s_txt]
        pos_ids = torch.arange(s_txt, device=text_ids.device)
        txt_tokens = self.token_emb(txt_ids) + self.txt_pos(pos_ids)[None]

        x = torch.cat([img_tokens, txt_tokens], dim=1)   # 单流拼接
        x = self.drop(x)
        for blk in self.blocks:
            x, _ = blk(x)                                 # 因果掩码内建
        x = self.final_norm(x)
        return self.lm_head(x[:, self.num_patches:])      # 只返回文本位 logits

    @torch.no_grad()
    def generate(self, pixel_values, prompt_ids, max_new_tokens=20,
                 temperature=1.0, top_k=50):
        """给图 + 文本前缀，自回归生成后续 token id"""
        self.eval()
        temperature = max(temperature, 1e-8)
        generated = prompt_ids.clone()
        budget = self.txt_pos.num_embeddings
        for _ in range(max_new_tokens):
            if generated.size(1) >= budget:
                break
            logits = self.forward(pixel_values, generated)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                vals, idx = torch.topk(next_logits, k, dim=-1)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, idx, vals)
            probs = torch.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tok], dim=1)
        return generated
