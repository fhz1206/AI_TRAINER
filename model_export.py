"""
model_export.py — 标准模型导出包组装器

所有模型下载统一输出为 HuggingFace 风格的标准结构：

    <model_name>.zip
    ├── model.safetensors          # 模型权重（Safetensors 格式）
    ├── config.json                # 架构配置（类型/超参/训练参数/指标）
    ├── vocab.json                 # 词表（文本类模型；token -> id）
    ├── merges.txt                 # BPE 合并规则（平台分词器）
    ├── vision_encoder.safetensors # 多模态专属：视觉编码器单独权重
    ├── preprocessor_config.json   # 多模态专属：图像预处理配置
    └── LICENSE                    # BSD-3-Clause 许可证

"""
import io
import json
import os
import zipfile

from safetensors.torch import load_file, save as st_save

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BPE_TOKENIZER_PATH = os.path.join(BASE_DIR, 'static', 'tokenizer_bpe.json')
REPO_LICENSE_PATH = os.path.join(BASE_DIR, 'LICENSE')

EXPORT_VERSION = 1

BSD3_TEXT = """BSD 3-Clause License

Copyright (c) 2026, AI_TRAINER contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

TEXT_MODEL_TYPES = {'text_generation'}
MULTIMODAL_MODEL_TYPES = {'multimodal_stream'}


def _license_text():
    try:
        with open(REPO_LICENSE_PATH, encoding='utf-8') as f:
            t = f.read().strip()
            if t:
                return t
    except OSError:
        pass
    return BSD3_TEXT


def _load_sidecar_meta(weight_path):
    """读取 safetensors 旁车元数据；不存在返回空 dict"""
    try:
        with open(weight_path + '.json', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _model_type(meta, weight_path):
    mt = (meta.get('model_type') or '').lower()
    if mt:
        return mt
    # 兜底：按文件名前缀推断旧产物
    name = os.path.basename(weight_path).lower()
    if name.startswith(('cnn_', 'vit_')):
        return 'image_cnn' if name.startswith('cnn_') else 'image_vit'
    if name.startswith('text_gen'):
        return 'text_generation'
    if name.startswith('dif_edit'):
        return 'image_edit_diffusion'
    if name.startswith('dif_gen'):
        return 'image_diffusion'
    if name.startswith('mm_'):
        return 'multimodal_stream'
    return 'unknown'


def _build_config(meta, model_type):
    """从旁车元数据组装标准 config.json"""
    metrics = {k: v for k, v in meta.items()
               if k.startswith('final_') or k == 'total_time'}
    return {
        'model_type': model_type,
        'architecture': meta.get('architecture', ''),
        'format': {'weights': 'safetensors', 'license': 'bsd-3-clause',
                   'export_version': EXPORT_VERSION},
        'model_params': meta.get('model_params', {}),
        'train_params': meta.get('train_params', {}),
        'metrics': metrics,
        'description': meta.get('description', ''),
    }


def _load_state_dict(weight_path):
    """读取 safetensors 权重为 state_dict"""
    return load_file(weight_path)


def _find_char_vocab(weight_path, meta):
    """定位训练期落盘的字符级词表（*_token2char.json），返回 token2char 或 None"""
    candidates = []
    data_path = (meta.get('train_params') or {}).get('data_path', '')
    if data_path:
        base = data_path.rstrip('/\\')
        candidates.append(base + '_token2char.json')
    # 模型同目录兜底搜索（按时间最近）
    model_dir = os.path.dirname(weight_path) or '.'
    try:
        entries = sorted(
            (os.path.getmtime(os.path.join(model_dir, f)), f)
            for f in os.listdir(model_dir) if '_token2char.' in f)
        candidates += [os.path.join(model_dir, f) for _, f in reversed(entries)]
    except OSError:
        pass

    for p in candidates:
        if p.endswith('.json') and os.path.exists(p):
            try:
                with open(p, encoding='utf-8') as f:
                    return {int(k): v for k, v in json.load(f).items()}
            except (OSError, ValueError):
                continue
    return None


def _tokenizer_files(weight_path, meta, model_type, config):
    """
    生成 vocab.json / merges.txt；非文本类模型返回空表。
    vocab.json 来源优先级：训练期字符词表 > 平台 BPE 词表（保证文件恒在）；
    实际来源写入 config['format']['vocab_source']。
    merges.txt 固定为平台 BPE 合并规则（GPT-2 风格）。
    """
    files = {}
    if model_type not in TEXT_MODEL_TYPES and model_type not in MULTIMODAL_MODEL_TYPES:
        return files

    # ---- vocab.json ----
    token2char = _find_char_vocab(weight_path, meta)
    if token2char:
        config['format']['vocab_source'] = 'training_char_vocab'
        files['vocab.json'] = json.dumps(
            {ch: int(i) for i, ch in sorted(token2char.items())},
            ensure_ascii=False, indent=2)
    else:
        # 回退：平台 BPE 词表（训练词表文件已被清理时的标准兜底）
        try:
            with open(BPE_TOKENIZER_PATH, encoding='utf-8') as f:
                bpe_vocab = json.load(f).get('vocab') or {}
            if bpe_vocab:
                config['format']['vocab_source'] = 'platform_bpe_fallback'
                files['vocab.json'] = json.dumps(
                    bpe_vocab, ensure_ascii=False)
        except OSError:
            pass

    # ---- merges.txt ----
    try:
        with open(BPE_TOKENIZER_PATH, encoding='utf-8') as f:
            bpe = json.load(f)
        merges = bpe.get('merges') or []
        lines = ['#version: 0.2'] + [
            ' '.join(m) if isinstance(m, (list, tuple)) else str(m)
            for m in merges]
        files['merges.txt'] = '\n'.join(lines) + '\n'
    except OSError:
        pass
    return files


def _vision_encoder_files(sd, model_params):
    """
    多模态专属：从完整权重中拆出视觉编码器单独权重与预处理配置。
    当前单流模型的视觉编码器为 patch_embed（Conv 切分投影）；
    未来替换为独立 ViT 编码器时只需扩展此处的抽取逻辑。
    直接按键名前缀过滤 state_dict，无需重建模型、不受元数据完整性影响。
    """
    files = {}
    enc_sd = {k: v.detach().contiguous().clone() for k, v in sd.items()
              if k.startswith('patch_embed.')}
    if not enc_sd:
        return files

    # st_save 返回字节串（本环境 save_file 不支持内存缓冲）
    files['vision_encoder.safetensors'] = st_save(enc_sd)
    files['preprocessor_config.json'] = json.dumps({
        'encoder': 'patch_embed',
        'encoder_num_params': sum(v.numel() for v in enc_sd.values()),
        'image_size': model_params.get('image_size', 32),
        'patch_size': model_params.get('patch_size', 8),
        'do_resize': True,
        'do_normalize': True,
        'image_mean': [0.5, 0.5, 0.5],
        'image_std': [0.5, 0.5, 0.5],
        'value_range': [-1, 1],
        'note': '像素值缩放到[-1,1]后输入；与训练端 ImageTextPairDataset 一致',
    }, ensure_ascii=False, indent=2)
    return files


def build_standard_zip(weight_path):
    """
    把单个模型文件组装成标准导出包。
    返回 (zip字节缓冲, 包名)；无法解析权重时抛 ValueError。
    """
    filename = os.path.basename(weight_path)
    stem = filename.rsplit('.', 1)[0]
    meta = _load_sidecar_meta(weight_path)

    model_type = _model_type(meta, weight_path)
    config = _build_config(meta, model_type)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # ---- 权重（统一 model.safetensors）----
        sd = _load_state_dict(weight_path)
        config['format']['weight_source'] = 'safetensors'
        # st_save 直接返回字节串（本环境 save_file 不支持内存缓冲）
        zf.writestr('model.safetensors', st_save(sd))

        # ---- 架构配置 ----
        zf.writestr('config.json', json.dumps(config, ensure_ascii=False, indent=2))

        # ---- 分词器文件 ----
        for name, content in _tokenizer_files(
                weight_path, meta, model_type, config).items():
            zf.writestr(name, content)

        # ---- 多模态专属：视觉编码器单独权重 + 预处理配置 ----
        if model_type in MULTIMODAL_MODEL_TYPES and sd is not None:
            for name, content in _vision_encoder_files(
                    sd, config.get('model_params') or {}).items():
                zf.writestr(name, content)

        # ---- 许可证 ----
        zf.writestr('LICENSE', _license_text())

    zip_name = stem + '_export.zip'
    buf.seek(0)
    return buf, zip_name
