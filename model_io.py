"""
model_io.py — 模型存取统一入口（SafeTensors 新格式 + .pth 旧格式兼容）

新格式说明：
- 权重：单个 <name>.safetensors 文件（只含张量，不执行代码，加载安全）
- 元数据：<name>.safetensors.json 旁车文件（架构参数/训练参数/指标），用于加载时重建网络
- 词表：TextDataset 另存为 *_token2char.json（纯 JSON，彻底去 pickle）
旧格式 .pth（整对象 pickle）仅保留读取能力，用于历史模型平滑过渡。
"""
import json
import os

import torch
from safetensors.torch import load_file, save_file

from model import SimpleResNet, TextTransformerModel

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
    """按扩展名保存：.safetensors（推荐，自动写旁车元数据）/ .pth（旧行为）"""
    if model_path.endswith('.safetensors'):
        state = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(state, model_path)
        meta_path = model_path + '.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(_build_metadata(model), f, ensure_ascii=False, indent=2)
    else:
        torch.save(model, model_path)


def _rebuild_architecture(meta):
    """根据元数据重建网络结构"""
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
    if mtype == 'text_generation':
        return TextTransformerModel(
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
        )
    raise ValueError(f'未知模型类型: {mtype or "(缺失)"}，无法重建网络')


def load_model(model_path, map_location='cpu'):
    """
    统一加载入口：
    - .safetensors：读旁车元数据 -> 重建结构 -> 载入权重 -> 附回 _metadata
    - .pth：旧格式整对象反序列化（历史模型兼容）
    返回 nn.Module（与旧 torch.load 行为一致，可直接 .eval()）
    """
    if model_path.endswith('.safetensors'):
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
    return torch.load(model_path, map_location=map_location, weights_only=False)


def save_vocab_json(token2char, folder_path):
    """词表改用 JSON 存储（键为 int 序号，JSON 中转为字符串键）"""
    vocab_path = f'{folder_path}_token2char.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in token2char.items()}, f, ensure_ascii=False)
    return vocab_path


def load_vocab(vocab_path_no_ext):
    """
    加载词表：优先 JSON（新），回退 .pth（旧）。传不带扩展名的路径前缀。
    返回 token2char dict 或 None。
    """
    json_path = f'{vocab_path_no_ext}_token2char.json'
    if os.path.exists(json_path):
        with open(json_path, encoding='utf-8') as f:
            return {int(k): v for k, v in json.load(f).items()}
    pth_path = f'{vocab_path_no_ext}_token2char.pth'
    if os.path.exists(pth_path):
        return torch.load(pth_path, map_location='cpu', weights_only=False)
    return None
