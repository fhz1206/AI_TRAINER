"""
models.text — 生成式文本 Transformer（Decoder-only，积木式注意力）

由原 model.py 的 TextTransformerModel 重构而来：
- 注意力从写死的 nn.TransformerDecoder 换成可插拔积木（full/flash/linear，
  默认 flash），Web 端可像搭积木一样切换
- FFN 可选切换为 MoE（使用修复版 architectures.moe.MoELayer）
- 对外接口保持不变：forward(input_ids[, return_aux_loss]) / generate(...)
"""
import torch
import torch.nn as nn

from architectures.blocks import TransformerBlock


class TextTransformerModel(nn.Module):
    """
    Decoder-only 生成式语言模型。

    注意力配置两种方式：
    - attention_type：统一类型（full/flash/linear，默认 flash）
    - attention_plan：逐层"积木序列"，支持混合注意力。结构：
          {
            'sequence': ['flash', 'linear', ...],  # 用户拖拽搭建的层序列
            'head': 'full' 或 None,                # 首层特殊设置（可选）
            'tail': 'full' 或 None,                # 尾层特殊设置（可选）
          }
      展开规则：head + sequence循环重复填充 + tail；
      若展开后层数超过 n_layers 则自动截断到 n_layers。
    """

    def __init__(self, vocab_size=1000, d_model=512, n_layers=6, n_heads=8,
                 d_ff=2048, max_seq_len=128, dropout=0.1, pad_token_id=0,
                 use_moe=False, moe_experts=4, moe_top_k=2, aux_loss_weight=0.02,
                 moe_noise_epsilon=0.01, use_mla=False, mla_dim=256,
                 attention_type='flash', attention_plan=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_moe = use_moe
        self.use_mla = use_mla
        self.pad_token_id = pad_token_id
        self.aux_loss_weight = aux_loss_weight
        self.attention_type = (attention_type or 'flash').lower()

        # ---- 解析逐层注意力计划（混合注意力积木）----
        self.attention_plan = self._resolve_attention_plan(
            attention_plan, n_layers)
        if self.attention_plan is None:
            # 未搭建积木：回退统一 attention_type
            self.attention_plan = [self.attention_type] * n_layers
        if len(set(self.attention_plan)) > 1:
            print(f"[TextTransformer] 混合注意力计划({n_layers}层): "
                  f"{self.attention_plan}")
        else:
            self.attention_type = self.attention_plan[0]
        self.attention_summary = (
            '混合[' + '→'.join(self.attention_plan) + ']'
            if len(set(self.attention_plan)) > 1 else self.attention_plan[0])

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
                attn_name=self.attention_plan[i],
                use_moe=use_moe, moe_experts=moe_experts,
                moe_top_k=moe_top_k, aux_loss_weight=aux_loss_weight,
            )
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        if use_mla:
            self.mla_proj = nn.Linear(d_model, mla_dim)
            self.lm_head = nn.Linear(mla_dim, vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_emb.weight  # 权重共享

        self._init_weights()

    @staticmethod
    def _resolve_attention_plan(attention_plan, n_layers):
        """
        把用户搭建的注意力积木计划展开为长度恰为 n_layers 的类型列表。

        规则：
        - sequence 循环重复填充：层数不足时重复已搭建的序列
        - head/tail 特殊层：非空时分别固定在首/尾
        - 展开后超过 n_layers：自动截断并打印提醒
        - 计划缺失/为空/非法：回退到统一 attention_type（由调用方语义处理，
          这里返回 [attention_type] 形式由上层兜底）
        """
        from architectures import available_attentions
        valid = set(available_attentions())

        if not attention_plan:
            return None
        seq = attention_plan.get('sequence') or []
        seq = [str(a).lower() for a in seq if str(a).lower() in valid]
        head = str(attention_plan.get('head') or '').lower() or None
        tail = str(attention_plan.get('tail') or '').lower() or None
        head = head if head in valid else None
        tail = tail if tail in valid else None

        if not seq:
            return None

        # 超出模型层数：先提醒再截断（只保留前 n_layers 块）
        if len(seq) > n_layers:
            print(f"[TextTransformer] ⚠️ 搭建的注意力积木({len(seq)}块)超过模型层数"
                  f"({n_layers})，已自动截断多余的积木")
            seq = seq[:n_layers]

        # 循环填充主体到 n_layers 层
        plan = [seq[i % len(seq)] for i in range(n_layers)]
        # 首尾特殊设置：覆盖循环体的第一格/最后一格
        if head:
            plan[0] = head
        if tail:
            plan[-1] = tail
        return plan

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
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, input_ids, return_aux_loss=False):
        """
        input_ids: (B, S)；返回 (B, S, vocab) logits；
        return_aux_loss=True 时额外返回 MoE 负载均衡辅助损失之和。
        """
        batch_size, seq_len = input_ids.size()
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            seq_len = self.max_seq_len

        pos = torch.arange(seq_len, device=input_ids.device)
        pos = pos.unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        key_padding_mask = (input_ids == self.pad_token_id)

        total_aux = x.new_zeros(())
        for blk in self.blocks:
            x, aux = blk(x, key_padding_mask=key_padding_mask)
            total_aux = total_aux + aux

        x = self.final_norm(x)
        if self.use_mla:
            x = self.mla_proj(x)
        logits = self.lm_head(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        if return_aux_loss:
            return logits, total_aux
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0,
                 top_k=50, pad_token_id=None, eos_token_id=None):
        """自回归生成；语义与旧版一致"""
        self.eval()
        pad_token_id = self.pad_token_id if pad_token_id is None else pad_token_id
        batch_size = prompt_ids.size(0)

        if prompt_ids.max() >= self.vocab_size or prompt_ids.min() < 0:
            raise ValueError(
                f"Prompt token超出词表范围！词表大小: {self.vocab_size}, "
                f"Prompt最大token: {prompt_ids.max().item()}, 最小token: {prompt_ids.min().item()}"
            )
        temperature = max(temperature, 1e-8)
        top_k = min(top_k, self.vocab_size) if top_k > 0 else 0

        generated = prompt_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

        for _ in range(max_new_tokens):
            input_seq = generated[:, -self.max_seq_len:]
            logits = self.forward(input_seq)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise RuntimeError("模型输出logits包含nan/inf，参数损坏，请重新训练")
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                valid_k = min(top_k, next_logits.size(-1))
                topk_vals, topk_idx = torch.topk(next_logits, valid_k, dim=-1)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, topk_idx, topk_vals)

            probs = torch.softmax(next_logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                probs = torch.ones_like(probs) / self.vocab_size
            else:
                s = probs.sum(dim=-1, keepdim=True)
                probs = torch.where(s > 0, probs / s,
                                    torch.ones_like(probs) / self.vocab_size)

            next_tok = torch.multinomial(probs, num_samples=1)
            next_tok[finished] = pad_token_id
            generated = torch.cat([generated, next_tok], dim=1)

            if eos_token_id is not None:
                finished |= (next_tok.squeeze(-1) == eos_token_id)
                if bool(finished.all()):
                    break

        return generated
