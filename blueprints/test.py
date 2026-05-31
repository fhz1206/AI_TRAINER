from flask import Blueprint, request, jsonify, session
import os
import shutil
import subprocess
import tempfile
import re
import zipfile
from time import time
from random import randint
from werkzeug.utils import secure_filename
from config import Config
from state import training_tasks
from blueprints.utils import login_required

test_bp = Blueprint('test', __name__)

@test_bp.route('/api/list_user_models')
@login_required
def list_user_models():
    """获取当前用户模型（按框架过滤，修复CNN模型检索bug）"""
    user_id = session['user_id']
    framework = request.args.get('framework', 'all')
    
    models_dir = f'models/{user_id}'
    if not os.path.exists(models_dir):
        return jsonify({'status': 'success', 'models': []})
    
    models = []
    # 前端框架标识 → 模型类型标识 映射（解决image和cnn无法匹配的bug）
    framework_to_model_type = {
        'image': 'cnn',   # 前端选CNN图片模型对应后端cnn类型
        'text': 'text'    # 前端选Transformer文本模型对应后端text类型
    }
    target_type = framework_to_model_type.get(framework, framework)

    for f in os.listdir(models_dir):
        if f.endswith('.pth'):
            # 优化模型类型判断，兼容非标准命名的模型
            model_type = 'other'
            f_lower = f.lower()
            if f_lower.startswith('cnn_') or 'cnn' in f_lower:
                model_type = 'cnn'
            elif f_lower.startswith('text_gen_') or ('transformer' in f_lower and 'text' in f_lower):
                model_type = 'text'
            
            # 按框架过滤（修复后的正确逻辑）
            if framework != 'all' and model_type != target_type:
                continue
            
            models.append({
                'name': f,
                'type': model_type,
                'path': os.path.join(models_dir, f),
                'size': f"{os.path.getsize(os.path.join(models_dir, f))/1024/1024:.1f} MB"
            })
    # 按时间倒序
    models.sort(key=lambda x: x['name'], reverse=True)
    return jsonify({'status': 'success', 'models': models})

@test_bp.route('/api/upload_test_data', methods=['POST'])
@login_required
def upload_test_data():
    """上传测试数据集（修复ZIP解压所有问题）"""
    user_id = session['user_id']
    framework = request.form.get('framework')
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': '请选择文件'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '请选择文件'})
    
    # 校验文件格式（不区分大小写，支持.ZIP/.Zip等）
    if framework == 'image' and not file.filename.lower().endswith('.zip'):
        return jsonify({'status': 'error', 'message': '图片测试仅支持.zip格式'})
    if framework == 'text' and not (file.filename.lower().endswith('.txt') or file.filename.lower().endswith('.zip')):
        return jsonify({'status': 'error', 'message': '文本测试仅支持.txt或.zip格式'})
    
    # 保存到用户测试数据目录
    test_dir = Config.TEST_DATA_FOLDER
    os.makedirs(test_dir, exist_ok=True)
    user_test_dir = os.path.join(test_dir, str(user_id))
    os.makedirs(user_test_dir, exist_ok=True)
    
    # 生成保存路径，强制确保zip文件有.zip后缀
    secure_name = secure_filename(file.filename)
    file_ext = os.path.splitext(secure_name)[1].lower()
    if framework in ['image', 'text'] and file.filename.lower().endswith('.zip') and not secure_name.lower().endswith('.zip'):
        secure_name = secure_name + '.zip'
        file_ext = '.zip'
    
    save_path = os.path.join(user_test_dir, f"test_{int(time())}_{secure_name}")
    file.save(save_path)
    
    extract_path = None
    file_count = 1
    if file_ext == '.zip':
        extract_base = os.path.splitext(save_path)[0]
        extract_path = extract_base
        
        # 解压前校验是否为有效ZIP
        if not zipfile.is_zipfile(save_path):
            os.remove(save_path)
            return jsonify({'status': 'error', 'message': '上传的文件不是有效的ZIP压缩包，请检查文件是否损坏'}), 400
        
        # 解压目录冲突处理
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

@test_bp.route('/api/run_test', methods=['POST'])
@login_required
def run_test():
    """运行测试（适配双框架：图片需上传数据，文本无需上传，修复引号冲突/编码问题）"""
    import ast
    user_id = session['user_id']
    data = request.json
    framework = data.get('framework')
    model_name = data.get('model_name')
    test_code = data.get('test_code')
    test_data_path = data.get('test_data_path')
    
    # -------------------------- 修改点1：参数校验适配双框架 --------------------------
    # 基础参数校验（模型、测试代码必填）
    if not all([framework, model_name, test_code]):
        return jsonify({'status': 'error', 'message': '参数不完整，请检查模型和测试代码是否已选择'})
    # 仅图片框架强制要求上传测试数据，文本框架不需要
    if framework == 'image' and not test_data_path:
        return jsonify({'status': 'error', 'message': '图片测试请先上传测试数据集'})
    # ---------------------------------------------------------------------

    # 校验模型存在
    model_path = f'models/{user_id}/{model_name}'
    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': '模型文件不存在，请检查模型是否已被删除'})
    
    # 构造完整测试脚本（注入公共变量+用户代码）
    injected_lines = [
        "import sys",
        "sys.path.insert(0, '.')",
        "sys.stdout.reconfigure(encoding='utf-8')",
        "sys.stderr.reconfigure(encoding='utf-8')",
        f'model_path = {repr(model_path)}',
        # 文本框架下test_data_path为None/空字符串，注入后不影响用户代码逻辑
        f'test_data_path = {repr(test_data_path if test_data_path else "")}',
        f'framework = {repr(framework)}',
        "",
        test_code,
        ""
    ]
    full_script = "\n".join(injected_lines)
    
    # -------------------------- 语法校验（精准定位用户代码错误行） --------------------------
    try:
        ast.parse(full_script)
    except SyntaxError as e:
        # 前端注入的固定代码有7行，错误行减去7就是用户代码的实际行数
        user_error_line = e.lineno - 7 if e.lineno > 7 else e.lineno
        error_content = e.text.strip() if e.text else ""
        error_msg = f"测试代码第{user_error_line}行语法错误：{e.msg}"
        if error_content:
            error_msg += f"\n错误内容：{error_content}"
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error': error_msg
        })
    # -----------------------------------------------------------------------------------

    test_script_path = None
    try:
        # 写入临时脚本
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(full_script)
            test_script_path = f.name
        
        # 设置UTF-8环境
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        
        # 执行测试
        result = subprocess.run(
            ['python', test_script_path],
            capture_output=True,
            timeout=300,
            cwd=os.getcwd(),
            env=env
        )
        
        # 解码输出
        output = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
        error = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
        success = result.returncode == 0
        
        # 解析指标（兼容准确率/Loss/PPL）
        metrics = {}
        if success and isinstance(output, str):
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
        
        return jsonify({
            'status': 'success' if success else 'failed',
            'output': output,
            'error': error,
            'metrics': metrics
        })
    
    except subprocess.TimeoutExpired:
        return jsonify({
            'status': 'failed',
            'output': '',
            'error': '测试超时（最多5分钟），请简化代码或减小数据集',
            'metrics': {}
        })
    except Exception as e:
        return jsonify({
            'status': 'failed',
            'output': '',
            'error': f'后端运行错误: {str(e)}',
            'metrics': {}
        })
    finally:
        if test_script_path and os.path.exists(test_script_path):
            try:
                os.remove(test_script_path)
            except:
                pass