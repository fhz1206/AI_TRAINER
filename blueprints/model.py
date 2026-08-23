from flask import Blueprint, request, jsonify, session, send_file
from os import path as os_path, getcwd, listdir
import io
import json
import os
import zipfile
from urllib.parse import unquote
from database import get_user_models, delete_model_record, get_model_count
from blueprints.utils import login_required

model_bp = Blueprint('model', __name__, url_prefix='/api')

@model_bp.route('/list_models')
@login_required
def api_list_models():
    """列出用户模型（和原有逻辑完全一致）"""
    user_id = session['user_id']
    models = get_user_models(user_id)
    
    if models:
        result = []
        for m in models:
            if m['file_path'] and os_path.exists(m['file_path']):
                result.append({
                    'name': m['model_name'],
                    'size': f"{m['file_size'] / 1024 / 1024:.2f} MB" if m['file_size'] else '未知',
                    'type': '🖼️ CNN' if m['model_type'] == 'cnn' else '📝 Transformer',
                    'accuracy': m['accuracy'],
                    'loss': m['loss'],
                    'created_at': m['created_at']
                })
        return jsonify({'models': result})
    
    # 兼容旧数据：从磁盘扫描
    user_dir = f'models/{user_id}'
    if not os_path.exists(user_dir):
        return jsonify({'models': []})
    
    result = []
    for f in sorted(listdir(user_dir), reverse=True):
        if f.endswith(('.pth', '.safetensors')):
            filepath = os_path.join(user_dir, f)
            file_size = os_path.getsize(filepath)
            model_type = '🖼️ CNN' if 'cnn' in f else '📝 Transformer' if 'text' in f else '📦 模型'
            result.append({
                'name': f,
                'size': f'{file_size / 1024 / 1024:.2f} MB',
                'type': model_type,
                'accuracy': None,
                'loss': None,
                'created_at': None
            })
    
    return jsonify({'models': result})

@model_bp.route('/download_model/<filename>')
@login_required
def api_download_model(filename):
    """下载模型：自动打包为 .zip（模型权重 + 元数据旁车 + 训练词表）"""
    user_id = session['user_id']
    filename = os_path.basename(unquote(filename))
    user_dir = f'models/{user_id}'
    filepath = os_path.join(user_dir, filename)

    if not os_path.exists(filepath):
        return jsonify({'status': 'error', 'message': '文件不存在'}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(filepath, arcname=filename)
        # 元数据旁车（safetensors 新格式）
        sidecar = filepath + '.json'
        meta = None
        if os_path.exists(sidecar):
            zf.write(sidecar, arcname=os_path.basename(sidecar))
            try:
                with open(sidecar, encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                meta = None
        # 词表：从元数据的训练数据路径定位（新格式），一并打入包中
        if meta:
            data_path = (meta.get('train_params') or {}).get('data_path', '')
            if data_path:
                base = data_path.rstrip('/\\')
                for ext in ('_token2char.json', '_token2char.pth'):
                    vp = base + ext
                    if os_path.exists(vp):
                        zf.write(vp, arcname=os_path.basename(vp))
                        break

    zip_name = os_path.splitext(filename)[0] + '.zip'
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=zip_name,
                     mimetype='application/zip')

@model_bp.route('/delete_model/<filename>', methods=['DELETE'])
@login_required
def api_delete_model(filename):
    """删除模型（和原有逻辑完全一致）"""
    user_id = session['user_id']
    filename = os_path.basename(unquote(filename))
    filepath = f'models/{user_id}/{filename}'
    
    if os_path.exists(filepath):
        os.remove(filepath)
        sidecar = filepath + '.json'
        if os_path.exists(sidecar):
            os.remove(sidecar)
    
    delete_model_record(filename, user_id)
    
    return jsonify({'status': 'success', 'message': '模型已删除'})