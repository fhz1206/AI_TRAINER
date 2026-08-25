"""
models.legacy — 旧版组件存档（仅供历史 .pth 检查点反序列化）

旧模型文件以 pickle 整对象保存，按模块路径 model.MoELayer 找类。
此处原样保留旧实现（含已知的广播形状 bug），保证旧检查点可加载、行为与当年一致；
新训练一律使用 architectures.moe.MoELayer（修复版），不要引用本模块。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """旧版 MoE 实现（存档，勿在新代码中使用）"""

    def __init__(self, d_model, d_ff, num_experts=4, top_k=2,
                 dropout=0.1, aux_loss_weight=0.02):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.d_model = d_model

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
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        gate_logits = self.gate(x_flat)
        gate_weights = F.softmax(gate_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            mask = (topk_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            expert_input = x_flat[mask]
            expert_output = self.experts[expert_idx](expert_input)
            weight = topk_weights[mask][:, topk_indices[mask].eq(expert_idx).float().argmax(dim=-1)]
            output[mask] += expert_output * weight.unsqueeze(-1)

        if self.training:
            expert_usage = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                expert_usage[i] = (topk_indices == i).any(dim=-1).float().mean()
            target_usage = torch.ones_like(expert_usage) / self.num_experts
            aux_loss = F.mse_loss(expert_usage, target_usage) * self.aux_loss_weight
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return output.view(batch_size, seq_len, d_model), aux_loss


class TextTransformerModel(nn.Module):
    """生成式Transformer大模型（Decoder-only架构），支持MLA和MoE"""
    def __init__(self, vocab_size=1000, d_model=512, n_layers=6, n_heads=8, 
                 d_ff=2048, max_seq_len=128, dropout=0.1, pad_token_id=0,
                 use_moe=False, moe_experts=4, moe_top_k=2, aux_loss_weight=0.02,
                 moe_noise_epsilon=0.01, use_mla=False, mla_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_moe = use_moe
        self.use_mla = use_mla
        self.pad_token_id = pad_token_id
        self.aux_loss_weight = aux_loss_weight

        # 词嵌入+位置嵌入（支持pad token）
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # 标准Decoder-only架构
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # MoE层（替换每层decoder的FFN，可选）
        if use_moe:
            self.moe_layers = nn.ModuleList([
                MoELayer(d_model, d_ff, num_experts=moe_experts, top_k=moe_top_k,
                         dropout=dropout, aux_loss_weight=aux_loss_weight)
                for _ in range(n_layers)
            ])
        else:
            self.moe_layers = None

        # LM头，支持MLA
        if use_mla:
            self.mla_proj = nn.Linear(d_model, mla_dim)
            self.lm_head = nn.Linear(mla_dim, vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_emb.weight  # 权重共享

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
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, input_ids, return_aux_loss=False):
        """
        生成式前向传播，支持MoE
        input_ids: shape (batch_size, seq_len)
        返回：(batch_size, seq_len, vocab_size) 预测logits
        """
        batch_size, seq_len = input_ids.size()
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            seq_len = self.max_seq_len

        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool)
        )
        tgt_key_padding_mask = (input_ids == self.pad_token_id)

        x = self.decoder(
            x, x,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # MoE 前向传播
        total_aux_loss = 0.0
        if self.use_moe and self.moe_layers is not None:
            for moe_layer in self.moe_layers:
                x, aux_loss = moe_layer(x)
                total_aux_loss += aux_loss

        if self.use_mla:
            x = self.mla_proj(x)
        logits = self.lm_head(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        if return_aux_loss:
            return logits, total_aux_loss
        return logits

    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=50, pad_token_id=0, eos_token_id=None):
        """生成式推理方法，和原有逻辑一致"""
        self.eval()
        batch_size = prompt_ids.size(0)
        
        if prompt_ids.max() >= self.vocab_size or prompt_ids.min() < 0:
            raise ValueError(
                f"Prompt token超出词表范围！词表大小: {self.vocab_size}, "
                f"Prompt最大token: {prompt_ids.max().item()}, 最小token: {prompt_ids.min().item()}"
            )
        temperature = max(temperature, 1e-8)
        top_k = min(top_k, self.vocab_size) if top_k > 0 else 0

        generated = prompt_ids.clone()

        for step in range(max_new_tokens):
            input_seq = generated[:, -self.max_seq_len:] if generated.size(1) > self.max_seq_len else generated
            with torch.no_grad():
                logits = self.forward(input_seq)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise RuntimeError("模型输出logits包含nan/inf，参数损坏，请重新训练")
                next_token_logits = logits[:, -1, :] / temperature
                
                if top_k > 0:
                    valid_top_k = min(top_k, next_token_logits.size(-1))
                    topk_logits, topk_indices = torch.topk(next_token_logits, valid_top_k, dim=-1)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, topk_indices, topk_logits)

                probs = torch.softmax(next_token_logits, dim=-1)
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    probs = torch.ones_like(probs) / self.vocab_size
                else:
                    prob_sum = probs.sum(dim=-1, keepdim=True)
                    probs = torch.where(prob_sum > 0, probs / prob_sum, torch.ones_like(probs) / self.vocab_size)

                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated
