"""
model_io.py — 模型存取统一入口（Safetensors 格式）

格式说明：
- 权重：单个 <name>.safetensors 文件（只含张量，不执行代码，加载安全）
- 元数据：<name>.safetensors.json 旁车文件（架构参数/训练参数/指标），用于加载时重建网络
- 词表：TextDataset 另存为 *_token2char.json（纯 JSON）
"""
import json
import os

from safetensors.torch import load_file, save_file

from model_zoo.vision import SimpleResNet, ViTModel
from model_zoo.text import TextTransformerModel
from model_zoo.text_cls import TextClassifier
from model_zoo.diffusion import DiffusionModel, DiffusionEditModel
from model_zoo.multimodal import MultiModalSingleStream

_META_SUFFIX = '.safetensors.json'


def _build_metadata(model):
    """从模型对象的 _metadata 提取可序列化元数据"""
    meta = getattr(model, '_metadata', {}) or {}
    return {
        '_format': 'aitp-v1',
        'model_type': meta.get('model_type', ''),
        'architecture': meta.get('architecture', ''),
        'model_params': meta.get('model_params', {}),
        'train_params': meta.get('train_params', {}),
        'metrics': {k: v for k, v in meta.items()
                    if k.startswith('final_') or k == 'total_time'},
        'description': meta.get('description', ''),
    }


def save_model(model, model_path):
    """保存为 safetensors + 旁车元数据（自动 clone 解除权重共享）"""
    if not model_path.endswith('.safetensors'):
        raise ValueError('仅支持 .safetensors 格式')
    # clone 解除权重共享（如 lm_head 与 token_emb 绑定），safetensors 不接受共享内存张量；
    # 加载侧重建结构时会重新执行绑定，两份相同数据写回同一存储，语义不变
    state = {k: v.detach().contiguous().clone()
             for k, v in model.state_dict().items()}
    save_file(state, model_path)
    meta_path = model_path + '.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(_build_metadata(model), f, ensure_ascii=False, indent=2)


def _build_text_kwargs(mp_):
    """文本生成模型的重建参数（新旧 safetensors 共用）"""
    kwargs = dict(
        vocab_size=mp_.get('vocab_size', 1000),
        d_model=mp_.get('d_model', 256),
        n_layers=mp_.get('n_layers', 4),
        n_heads=mp_.get('n_heads', 8),
        d_ff=mp_.get('d_ff', 1024),
        max_seq_len=mp_.get('max_seq_len', 128),
        dropout=mp_.get('dropout', 0.1),
        pad_token_id=mp_.get('pad_token_id', 0),
        use_moe=mp_.get('use_moe', False),
        use_mla=mp_.get('use_mla', False),
        moe_experts=mp_.get('moe_experts', 4),
        moe_top_k=mp_.get('moe_top_k', 2),
        mla_dim=mp_.get('mla_dim', 256),
        aux_loss_weight=min(mp_.get('aux_loss_weight', 0.02), 0.05),
        # 新字段：旧元数据缺失时默认 flash（与旧版行为最接近的融合实现）
        attention_type=mp_.get('attention_type', 'flash'),
    )
    # 混合注意力积木计划：存在且合法时逐层装配；键过滤防脏数据
    plan = mp_.get('attention_plan')
    if isinstance(plan, dict) and plan.get('sequence'):
        from architectures import available_attentions
        valid = set(available_attentions())
        seq = [str(a).lower() for a in (plan.get('sequence') or [])
               if str(a).lower() in valid]
        head = str(plan.get('head') or '').lower()
        tail = str(plan.get('tail') or '').lower()
        if seq:
            kwargs['attention_plan'] = {
                'sequence': seq,
                'head': head if head in valid else None,
                'tail': tail if tail in valid else None,
            }
    return kwargs


def _rebuild_architecture(meta):
    """根据元数据重建网络结构（覆盖全部注册的模型类型）"""
    mp_ = meta.get('model_params', {})
    mtype = meta.get('model_type', '')

    if mtype == 'image_cnn':
        return SimpleResNet(
            image_size=mp_.get('image_size', 224),
            num_classes=mp_.get('num_classes', 2),
            in_channels=mp_.get('in_channels', 3),
            base_channels=mp_.get('base_channels', 64),
            dropout=mp_.get('dropout', 0.1),
        )
    if mtype == 'image_vit':
        return ViTModel(
            image_size=mp_.get('image_size', 224),
            patch_size=mp_.get('patch_size', 32),
            in_channels=mp_.get('in_channels', 3),
            num_classes=mp_.get('num_classes', 10),
            d_model=mp_.get('d_model', 192),
            n_layers=mp_.get('n_layers', 6),
            n_heads=mp_.get('n_heads', 4),
            d_ff=mp_.get('d_ff', 384),
            dropout=mp_.get('dropout', 0.1),
            attention_type=mp_.get('attention_type', 'flash'),
            attention_plan=mp_.get('attention_plan'),
        )
    if mtype == 'text_generation':
        return TextTransformerModel(**_build_text_kwargs(mp_))
    if mtype == 'text_classifier':
        return TextClassifier(
            vocab_size=mp_.get('vocab_size', 5000),
            num_classes=mp_.get('num_classes', 2),
            d_model=mp_.get('d_model', 128),
            n_layers=mp_.get('n_layers', 4),
            n_heads=mp_.get('n_heads', 4),
            d_ff=mp_.get('d_ff', 256),
            max_seq_len=mp_.get('max_seq_len', 64),
            dropout=mp_.get('dropout', 0.1),
            pad_token_id=mp_.get('pad_token_id', 0),
            attention_type=mp_.get('attention_type', 'flash'),
            attention_plan=mp_.get('attention_plan'),
        )
    if mtype == 'image_diffusion':
        return DiffusionModel(
            image_size=mp_.get('image_size', 32),
            base_channels=mp_.get('base_channels', 32),
            num_timesteps=mp_.get('num_timesteps', 300),
            attn_heads=mp_.get('attn_heads', 4),
        )
    if mtype == 'image_edit_diffusion':
        return DiffusionEditModel(
            image_size=mp_.get('image_size', 32),
            base_channels=mp_.get('base_channels', 32),
            num_timesteps=mp_.get('num_timesteps', 300),
            attn_heads=mp_.get('attn_heads', 4),
        )
    if mtype == 'multimodal_stream':
        return MultiModalSingleStream(
            vocab_size=mp_.get('vocab_size', 1000),
            image_size=mp_.get('image_size', 32),
            patch_size=mp_.get('patch_size', 8),
            in_channels=mp_.get('in_channels', 3),
            d_model=mp_.get('d_model', 192),
            n_layers=mp_.get('n_layers', 4),
            n_heads=mp_.get('n_heads', 4),
            d_ff=mp_.get('d_ff', 384),
            max_seq_len=mp_.get('max_seq_len', 64),
            dropout=mp_.get('dropout', 0.1),
            pad_token_id=mp_.get('pad_token_id', 0),
            use_moe=mp_.get('use_moe', False),
            moe_experts=mp_.get('moe_experts', 4),
            moe_top_k=mp_.get('moe_top_k', 2),
            attention_type=mp_.get('attention_type', 'flash'),
        )
    raise ValueError(f'未知模型类型: {mtype or "(缺失)"}，无法重建网络')


def load_model(model_path, map_location='cpu'):
    """
    统一加载入口：读旁车元数据 -> 重建结构 -> 载入权重 -> 附回 _metadata。
    返回 nn.Module（可直接 .eval()）。
    """
    if not model_path.endswith('.safetensors'):
        raise ValueError('仅支持 .safetensors 格式')
    meta_path = model_path + '.json'
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'缺少元数据文件: {meta_path}')
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    model = _rebuild_architecture(meta)
    state = load_file(model_path, device=str(map_location))
    model.load_state_dict(state, strict=True)
    merged = dict(meta.get('metrics', {}))
    merged.update({
        'model_type': meta.get('model_type', ''),
        'architecture': meta.get('architecture', ''),
        'model_params': meta.get('model_params', {}),
        'train_params': meta.get('train_params', {}),
        'description': meta.get('description', ''),
    })
    object.__setattr__(model, '_metadata', merged)
    return model


def save_vocab_json(token2char, folder_path):
    """词表以 JSON 存储（键为 int 序号，JSON 中转为字符串键）"""
    vocab_path = f'{folder_path}_token2char.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in token2char.items()}, f, ensure_ascii=False)
    return vocab_path


def load_vocab(vocab_path_no_ext):
    """
    加载词表（JSON）。传不带扩展名的路径前缀。
    返回 token2char dict 或 None。
    """
    json_path = f'{vocab_path_no_ext}_token2char.json'
    if os.path.exists(json_path):
        with open(json_path, encoding='utf-8') as f:
            return {int(k): v for k, v in json.load(f).items()}
    return None
