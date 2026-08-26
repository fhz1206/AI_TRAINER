from flask import Blueprint, request, jsonify, session
import os
import json
import shutil
import re
import zipfile
from time import time
from random import randint
from werkzeug.utils import secure_filename
from config import Config
from state import training_tasks
from blueprints.utils import login_required
from database import add_activity_log
# -------------------------- 依赖部分（原有依赖完全保留，新增MediaPipe相关） --------------------------
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import io
import base64
import cv2
import numpy as np
import mediapipe as mp
# -----------------------------------------------------------------------------------
test_bp = Blueprint('test', __name__)
# -------------------------- 全局变量（原有线程池保留，新增手部检测缓存） --------------------------
# 手部检测模型线程锁，保证线程安全
_hand_model_lock = threading.Lock()
_global_hand_models = None
# 原有线程池，最大并发数可根据服务器配置调整
_executor = ThreadPoolExecutor(max_workers=4)
# -----------------------------------------------------------------------------------
# -------------------------- MediaPipe手部模型初始化（官方Tasks API标准写法，无任何自定义包装） --------------------------
def _init_hand_models():
    """官方Tasks API标准初始化，直接返回原生检测器，无冗余兼容层"""
    global _global_hand_models
    if _global_hand_models is not None:
        return _global_hand_models
    with _hand_model_lock:
        if _global_hand_models is None:
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                # 模型路径：项目根目录下的mediapipe_models/hand_landmarker.task
                model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mediapipe_models', 'hand_landmarker.task')
                if not os.path.exists(model_path):
                    raise RuntimeError(f"手部检测模型文件不存在，请下载hand_landmarker.task放到{model_path}路径下")
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=1,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                # 直接创建官方原生检测器，无任何包装
                _global_hand_models = vision.HandLandmarker.create_from_options(options)
                print("✅ MediaPipe手部模型（官方Tasks API）初始化成功")
            except ImportError as e:
                error_msg = f"MediaPipe导入失败：{str(e)}，请确认已安装最新版mediapipe（pip install --upgrade mediapipe）"
                print(error_msg)
                raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"MediaPipe模型初始化失败：{str(e)}，请检查模型文件路径和环境配置"
                print(error_msg)
                raise RuntimeError(error_msg)
    return _global_hand_models
# -----------------------------------------------------------------------------------
# -------------------------- 原有上传接口，完全保留逻辑，仅新增head_mode适配 --------------------------
# -------------------------- 本地测试代码模板接口 --------------------------
_TEST_TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_templates')
_TEST_TEMPLATE_WHITELIST = ('image', 'image_vit', 'text', 'text_cls',
                            'diffusion', 'diffusion_edit', 'multimodal',
                            'head_mode')


@test_bp.route('/api/test_template/<name>')
@login_required
def api_test_template(name):
    """返回本地测试示例代码（白名单校验，防目录穿越）"""
    key = (name or '').lower()
    if key not in _TEST_TEMPLATE_WHITELIST:
        return jsonify({'status': 'error',
                        'message': f'未知模板，可选: {list(_TEST_TEMPLATE_WHITELIST)}'}), 404
    path = os.path.join(_TEST_TEMPLATE_DIR, f'{key}.py.txt')
    try:
        with open(path, encoding='utf-8') as f:
            code = f.read()
    except OSError:
        return jsonify({'status': 'error', 'message': '模板文件缺失'}), 404
    return jsonify({'status': 'success', 'name': key, 'code': code})


@test_bp.route('/api/upload_test_data', methods=['POST'])
@login_required
def upload_test_data():
    """上传测试数据集（原逻辑100%保留，新增head_mode单图上传支持）"""
    user_id = session['user_id']
    framework = request.form.get('framework')
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': '请选择文件'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '请选择文件'})
    # 原框架格式校验保留（扩展：ViT 同为 zip 数据集；编辑/多模态为单图）
    if framework in ('image', 'image_vit') and not file.filename.lower().endswith('.zip'):
        return jsonify({'status': 'error', 'message': '图片分类测试仅支持.zip格式'})
    if framework == 'text_cls' and not file.filename.lower().endswith('.zip'):
        return jsonify({'status': 'error', 'message': '语言分类测试仅支持.zip格式（顶层类别文件夹）'})
    if framework == 'text' and not (file.filename.lower().endswith('.txt') or file.filename.lower().endswith('.zip')):
        return jsonify({'status': 'error', 'message': '文本测试仅支持.txt或.zip格式'})
    # 单图类测试格式校验（手部/扩散编辑/多模态）
    if framework in ('head_mode', 'diffusion_edit', 'multimodal') and not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        return jsonify({'status': 'error', 'message': '该测试仅支持jpg/png等图片格式'})
    # 原保存路径逻辑完全保留
    test_dir = Config.TEST_DATA_FOLDER
    os.makedirs(test_dir, exist_ok=True)
    user_test_dir = os.path.join(test_dir, str(user_id))
    os.makedirs(user_test_dir, exist_ok=True)
    secure_name = secure_filename(file.filename)
    file_ext = os.path.splitext(secure_name)[1].lower()
    if framework in ['image', 'text'] and file.filename.lower().endswith('.zip') and not secure_name.lower().endswith('.zip'):
        secure_name = secure_name + '.zip'
        file_ext = '.zip'
    save_path = os.path.join(user_test_dir, f"test_{int(time())}_{secure_name}")
    file.save(save_path)
    extract_path = None
    file_count = 1
    # 新增head_mode无需解压，直接返回路径
    if framework == 'head_mode':
        return jsonify({
            'status': 'success',
            'message': '上传成功',
            'file_count': 1,
            'path': save_path,
            'filename': file.filename
        })
    # 原zip解压逻辑完全保留
    if file_ext == '.zip':
        extract_base = os.path.splitext(save_path)[0]
        extract_path = extract_base
        if os.path.exists(extract_path):
            try:
                if os.path.isdir(extract_path):
                    shutil.rmtree(extract_path)
                else:
                    os.remove(extract_path)
            except Exception:
                extract_path = f"{extract_path}_{int(time())}_{randint(100, 999)}"
        os.makedirs(extract_path, exist_ok=True)
        try:
            with zipfile.ZipFile(save_path, 'r') as zf:
                zf.extractall(extract_path)
            os.remove(save_path)
        except Exception as e:
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path, ignore_errors=True)
            if os.path.exists(save_path):
                os.remove(save_path)
            return jsonify({'status': 'error', 'message': f'ZIP解压失败: {str(e)}，请检查压缩包是否损坏'}), 400
        file_count = 0
        for root, dirs, files in os.walk(extract_path):
            file_count += len(files)
    return jsonify({
        'status': 'success',
        'message': '上传成功',
        'file_count': file_count,
        'path': extract_path if extract_path else save_path,
        'filename': file.filename
    })
# -----------------------------------------------------------------------------------
# -------------------------- 原有线程执行函数，完全保留逻辑 --------------------------
def _execute_code_in_thread(code: str, global_vars: dict, timeout: int = 300):
    """线程安全的代码执行函数，原逻辑完全保留"""
    stdout_io = io.StringIO()
    stderr_io = io.StringIO()
    try:
        with redirect_stdout(stdout_io), redirect_stderr(stderr_io):
            exec(code, global_vars)
        custom_result = global_vars.get('_hand_detection_result', {})
        return {
            'success': True,
            'stdout': stdout_io.getvalue(),
            'stderr': stderr_io.getvalue(),
            'custom_result': custom_result
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': stdout_io.getvalue(),
            'stderr': stderr_io.getvalue(),
            'error': str(e)
        }
# -----------------------------------------------------------------------------------
# -------------------------- 运行测试接口（完全删除后端模板，直接使用前端传递的test_code） --------------------------
@test_bp.route('/api/run_test', methods=['POST'])
@login_required
def run_test():
    """运行测试（原image/text逻辑100%保留，新增head_mode支持，无需后端代码模板）"""
    import ast
    user_id = session['user_id']
    data = request.json
    framework = data.get('framework')
    model_name = data.get('model_name')
    # 直接使用前端传递的test_code，无需后端存储任何模板
    test_code = data.get('test_code')
    test_data_path = data.get('test_data_path')
    # 参数校验（原逻辑保留，新增head_mode适配）
    if framework == 'head_mode':
        if not test_data_path:
            return jsonify({'status': 'error', 'message': '请先上传手部照片'})
        if not test_code:
            return jsonify({'status': 'error', 'message': '测试代码不能为空'})
    else:
        # 原有image/text参数校验完全保留
        if not all([framework, model_name, test_code]):
            return jsonify({'status': 'error', 'message': '参数不完整，请检查模型和测试代码是否已选择'})
        if framework in ('image', 'image_vit', 'text_cls') and not test_data_path:
            return jsonify({'status': 'error',
                            'message': '分类测试请先上传测试数据集'})
    # 模型校验（head_mode下跳过，使用内置模型）
    if framework != 'head_mode':
        model_path = f'models/{user_id}/{model_name}'
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': '模型文件不存在，请检查模型是否已被删除'})
    else:
        model_path = "内置手部检测模型"
    # 原有语法校验逻辑完全保留
    try:
        ast.parse(test_code)
    except SyntaxError as e:
        user_error_line = e.lineno
        error_content = e.text.strip() if e.text else ""
        error_msg = f"测试代码第{user_error_line}行语法错误：{e.msg}"
        if error_content:
            error_msg += f"\n错误内容：{error_content}"
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error': error_msg
        })
    # 构造注入变量（原通用依赖保留，新增MediaPipe相关注入，解决导入冲突）
    inject_vars = {
        'model_path': model_path,
        'test_data_path': test_data_path if test_data_path else "",
        'framework': framework,
        'torch': __import__('torch'),
        'cv2': cv2,
        'np': np,
        'plt': __import__('matplotlib.pyplot'),
        'os': os,
        'sys': __import__('sys'),
        'json': __import__('json'),
        'pandas': __import__('pandas'),
        # 新增MediaPipe相关注入
        'mp': mp,
        # 直接注入初始化函数，彻底解决导入test包冲突问题
        '_init_hand_models': _init_hand_models,
    }
    # 原有线程执行逻辑完全保留
    try:
        future = _executor.submit(_execute_code_in_thread, test_code, inject_vars, timeout=300)
        exec_result = future.result(timeout=300)
    except Exception as e:
        return jsonify({
            'status': 'failed',
            'output': '',
            'error': f'执行超时或失败：{str(e)}',
            'metrics': {}
        })
    # 原有结果处理逻辑完全保留，新增head_mode结果适配
    if not exec_result['success']:
        return jsonify({
            'status': 'failed',
            'output': exec_result['stdout'],
            'error': exec_result['stderr'] + exec_result.get('error', ''),
            'metrics': {}
        })
    output = exec_result['stdout']
    error = exec_result['stderr']
    metrics = {}
    # 原有指标解析逻辑完全保留
    acc_match = re.search(r'准确率[：:]\s*([\d.]+)%', output)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))
    loss_match = re.search(r'Loss[：:]\s*([\d.]+)', output)
    if loss_match:
        metrics['loss'] = float(loss_match.group(1))
    ppl_match = re.search(r'困惑度（PPL）[：:]\s*([\d.]+|inf)', output)
    if ppl_match:
        ppl_val = ppl_match.group(1)
        metrics['ppl'] = float(ppl_val) if ppl_val != 'inf' else 'inf'
    # 新增head_mode结果返回
    custom_result = exec_result.get('custom_result', {})
    
    # 记录行为日志
    if exec_result.get('success'):
        model_label = model_name or '内置手部模型'
        add_activity_log(user_id, 'test', f'运行 {framework} 测试（模型: {model_label}）',
                         f'指标: {json.dumps(metrics, ensure_ascii=False)}')
    
    return jsonify({
        'status': 'success',
        'output': output,
        'error': error,
        'metrics': metrics,
        'head_mode_result': custom_result
    })
# -----------------------------------------------------------------------------------
# -------------------------- 原有模型列表接口，完全保留逻辑，仅新增head_mode适配 --------------------------
@test_bp.route('/api/list_user_models')
@login_required
def list_user_models():
    """获取当前用户模型（按框架过滤；覆盖全部六类训练产物）"""
    user_id = session['user_id']
    framework = request.args.get('framework', 'all')
    models_dir = f'models/{user_id}'
    if not os.path.exists(models_dir):
        return jsonify({'status': 'success', 'models': []})
    models = []
    # 前端分区 -> 模型类型
    framework_to_model_type = {
        'image': 'cnn',
        'image_vit': 'vit',
        'text': 'text',
        'text_cls': 'text_cls',
        'diffusion': 'diffusion',
        'diffusion_edit': 'diffusion_edit',
        'multimodal': 'multimodal',
        'head_mode': 'hand',
    }
    target_type = framework_to_model_type.get(framework, framework)
    for f in os.listdir(models_dir):
        if f.endswith('.safetensors'):
            model_type = 'other'
            f_lower = f.lower()
            # 按训练产物文件名前缀归类（与 trainers 各实现保持一致）
            if f_lower.startswith('cnn_'):
                model_type = 'cnn'
            elif f_lower.startswith('vit_'):
                model_type = 'vit'
            elif f_lower.startswith('dif_gen_'):
                model_type = 'diffusion'
            elif f_lower.startswith('dif_edit_'):
                model_type = 'diffusion_edit'
            elif f_lower.startswith('text_cls_'):
                model_type = 'text_cls'
            elif f_lower.startswith('mm_stream_'):
                model_type = 'multimodal'
            elif f_lower.startswith('text_gen_'):
                model_type = 'text'
            elif 'transformer' in f_lower and 'text' in f_lower:
                model_type = 'text'
            if framework != 'all' and model_type != target_type:
                continue
            models.append({
                'name': f,
                'type': model_type,
                'path': os.path.join(models_dir, f),
                'size': f"{os.path.getsize(os.path.join(models_dir, f))/1024/1024:.1f} MB"
            })
    models.sort(key=lambda x: x['name'], reverse=True)
    # 新增head_mode返回内置模型
    if framework == 'head_mode':
        models = [{'name': '内置MediaPipe手部关键点模型', 'type': 'hand', 'path': 'built-in', 'size': '7.5 MB'}]
    return jsonify({'status': 'success', 'models': models})
# -----------------------------------------------------------------------------------