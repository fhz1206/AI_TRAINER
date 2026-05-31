from flask import Blueprint, request, jsonify, session
from werkzeug.utils import secure_filename
import os
import zipfile
import tempfile
import base64
from time import time
from config import Config, IMAGE_EXTENSIONS
from database import save_file_record, get_user_files, get_db
from state import training_tasks
from trainer import train_image_model, train_text_model
from blueprints.utils import login_required

training_bp = Blueprint('training', __name__, url_prefix='/api')

# ==================== 文件上传相关 ====================
@training_bp.route('/preview_zip', methods=['POST'])
@login_required
def api_preview_zip():
    """解析ZIP文件结构，返回类别和图片预览（和原有逻辑完全一致）"""
    if 'zip_file' not in request.files:
        return jsonify({'status': 'error', 'message': '未选择ZIP文件'}), 400
    
    zip_file = request.files['zip_file']
    if not zip_file.filename.endswith('.zip'):
        return jsonify({'status': 'error', 'message': '请上传.zip格式的文件'}), 400
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            zip_path = os.path.join(temp_dir, 'upload.zip')
            zip_file.save(zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'ZIP解压失败: {str(e)}'}), 400
        
        top_items = os.listdir(temp_dir)
        top_folders = [f for f in top_items if os.path.isdir(os.path.join(temp_dir, f))]
        
        if len(top_folders) < 1:
            return jsonify({'status': 'error', 'message': 'ZIP内未找到任何文件夹，请确保每个类别对应一个子文件夹'}), 400
        
        classes = []
        total_images = 0
        
        for folder in top_folders:
            folder_path = os.path.join(temp_dir, folder)
            image_paths = []
            
            for root, dirs, files in os.walk(folder_path):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        image_paths.append(os.path.join(root, f))
            
            if not image_paths:
                continue
            
            sample_images = []
            for img_path in image_paths[:4]:
                try:
                    with open(img_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                    ext = os.path.splitext(img_path)[1].lower().replace('.', '')
                    sample_images.append(f'data:image/{ext};base64,{img_data}')
                except:
                    pass
            
            classes.append({
                'name': folder,
                'count': len(image_paths),
                'samples': sample_images
            })
            total_images += len(image_paths)
        
        if not classes:
            return jsonify({'status': 'error', 'message': 'ZIP内所有文件夹均未找到图片文件，请检查文件夹内容'}), 400
        
        return jsonify({
            'status': 'success',
            'classes': classes,
            'total_images': total_images,
            'total_classes': len(classes)
        })

@training_bp.route('/upload', methods=['POST'])
@login_required
def api_upload():
    """上传训练数据（和原有逻辑完全一致）"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': '未选择文件'}), 400
    
    file = request.files['file']
    train_type = request.form.get('train_type', 'image')
    user_id = session['user_id']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '文件名为空'}), 400
    
    user_upload_dir = os.path.join(Config.UPLOAD_FOLDER, str(user_id), train_type)
    os.makedirs(user_upload_dir, exist_ok=True)
    
    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({'status': 'error', 'message': '文件名不合法'}), 400
    
    filepath = os.path.join(user_upload_dir, filename)
    file.save(filepath)
    
    if filename.endswith('.zip'):
        extract_dir = os.path.join(user_upload_dir, filename.replace('.zip', ''))
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(filepath, 'r') as zf:
                zf.extractall(extract_dir)
            os.remove(filepath)
            filepath = extract_dir
            filename = filename.replace('.zip', '')
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'ZIP解压失败: {str(e)}'}), 400
    
    file_count = 0
    if os.path.isdir(filepath):
        for root, dirs, files in os.walk(filepath):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if train_type == 'image' and ext in IMAGE_EXTENSIONS:
                    file_count += 1
                elif train_type == 'text' and ext == '.txt':
                    file_count += 1
    else:
        ext = os.path.splitext(filepath)[1].lower()
        if (train_type == 'image' and ext in IMAGE_EXTENSIONS) or (train_type == 'text' and ext == '.txt'):
            file_count = 1
    
    file_size = os.path.getsize(filepath) if os.path.isfile(filepath) else 0
    save_file_record(
        user_id=user_id,
        filename=os.path.basename(filepath),
        original_name=filename,
        file_size=file_size,
        file_path=filepath,
        train_type=train_type
    )
    
    conn = get_db()
    ds = conn.execute(
        'SELECT id FROM files WHERE user_id = ? AND file_path = ? ORDER BY created_at DESC LIMIT 1',
        (user_id, filepath)
    ).fetchone()
    file_id = ds['id'] if ds else None
    conn.close()
    
    return jsonify({
        'status': 'success',
        'message': f'上传成功！共 {file_count} 个文件',
        'file_count': file_count,
        'file_id': file_id
    })

@training_bp.route('/preview_data')
@login_required
def api_preview_data():
    """预览已上传的数据（和原有逻辑完全一致）"""
    train_type = request.args.get('train_type', 'image')
    user_id = session['user_id']
    
    files = get_user_files(user_id, train_type)
    
    items = []
    MAX_PREVIEW = 12
    
    for f in files:
        if len(items) >= MAX_PREVIEW:
            break
            
        file_path = f['file_path']
        if not file_path or not os.path.exists(file_path):
            continue
        
        if train_type == 'image':
            try:
                image_paths = []
                if os.path.isdir(file_path):
                    for root, dirs, fs in os.walk(file_path):
                        for img_name in fs:
                            ext = os.path.splitext(img_name)[1].lower()
                            if ext in IMAGE_EXTENSIONS:
                                image_paths.append(os.path.join(root, img_name))
                                if len(image_paths) >= 12:
                                    break
                        if len(image_paths) >= 12:
                            break
                else:
                    if os.path.splitext(file_path)[1].lower() in IMAGE_EXTENSIONS:
                        image_paths.append(file_path)
                
                for img_path in image_paths:
                    if len(items) >= MAX_PREVIEW:
                        break
                    if os.path.exists(img_path):
                        with open(img_path, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode()
                        ext = os.path.splitext(img_path)[1].lower().replace('.', '')
                        items.append({
                            'type': 'image',
                            'name': os.path.basename(img_path),
                            'data': f'data:image/{ext};base64,{img_data}'
                        })
            except Exception as e:
                print(f"[Preview] 图片加载失败: {e}")
                continue
                
        else:
            try:
                txt_paths = []
                if os.path.isdir(file_path):
                    for root, dirs, fs in os.walk(file_path):
                        for txt_name in fs:
                            if txt_name.endswith('.txt'):
                                txt_paths.append(os.path.join(root, txt_name))
                                if len(txt_paths) >= 6:
                                    break
                        if len(txt_paths) >= 6:
                            break
                else:
                    if file_path.endswith('.txt'):
                        txt_paths.append(file_path)
                
                for txt_path in txt_paths:
                    if len(items) >= MAX_PREVIEW:
                        break
                    try:
                        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as tf:
                            content = tf.read()[:200]
                        items.append({
                            'type': 'text',
                            'name': os.path.basename(txt_path),
                            'preview': content
                        })
                    except:
                        pass
            except Exception as e:
                print(f"[Preview] 文本加载失败: {e}")
                continue
    
    return jsonify({'items': items})

# ==================== 训练任务相关 ====================
@training_bp.route('/list_datasets')
@login_required
def api_list_datasets():
    """获取用户所有训练数据集（和原有逻辑完全一致）"""
    user_id = session['user_id']
    datasets = get_user_files(user_id, train_type=None)
    
    result = []
    for ds in datasets:
        file_path = ds['file_path']
        if not file_path or not os.path.exists(file_path):
            continue
        sample_count = 0
        ds_type = ds['train_type']
        if ds_type == 'image':
            if os.path.isdir(file_path):
                for root, dirs, files in os.walk(file_path):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                            sample_count += 1
            else:
                if os.path.splitext(file_path)[1].lower() in IMAGE_EXTENSIONS:
                    sample_count = 1
        elif ds_type == 'text':
            if os.path.isdir(file_path):
                for root, dirs, files in os.walk(file_path):
                    for f in files:
                        if os.path.splitext(f)[1].lower() == '.txt':
                            sample_count += 1
            else:
                if os.path.splitext(file_path)[1].lower() == '.txt':
                    sample_count = 1
        if sample_count == 0:
            continue
        size_mb = ds['file_size'] / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb > 1 else f"{ds['file_size']/1024:.1f} KB"
        result.append({
            'id': ds['id'],
            'name': ds['original_name'],
            'path': file_path,
            'sample_count': sample_count,
            'file_size': size_str,
            'created_at': ds['created_at'],
            'type': ds_type
        })
    return jsonify({'datasets': result})

@training_bp.route('/start_training', methods=['POST'])
@login_required
def api_start_training():
    """启动训练任务（和原有逻辑完全一致）"""
    from time import time as time_now
    from uuid import uuid4
    data = request.json
    user_id = session['user_id']
    train_type = data.get('train_type', 'image')
    task_id = str(uuid4())[:8]
    
    dataset_id = data.get('dataset_id')
    data_path = None
    if dataset_id:
        conn = get_db()
        ds = conn.execute(
            'SELECT file_path FROM files WHERE id = ? AND user_id = ?',
            (dataset_id, user_id)
        ).fetchone()
        conn.close()
        if ds:
            data_path = ds['file_path']
            print(f"[训练] 使用选中数据集 ID={dataset_id}, 路径={data_path}")
    
    if not data_path:
        user_upload_dir = os.path.join(Config.UPLOAD_FOLDER, str(user_id), train_type)
        files = get_user_files(user_id, train_type)
        for f in files:
            if f['file_path'] and os.path.exists(f['file_path']):
                data_path = f['file_path']
                print(f"[训练] 使用最新数据集 ID={f['id']}, 路径={data_path}")
                break
        if not data_path and os.path.exists(user_upload_dir):
            items = os.listdir(user_upload_dir)
            if items:
                data_path = os.path.join(user_upload_dir, items[-1])
    
    if not data_path:
        return jsonify({'status': 'error', 'message': '请先上传训练数据'}), 400
    
    model_params = {
        'image_size': data.get('image_size', 224),
        'num_classes': data.get('num_classes', 10),
        'base_channels': data.get('base_channels', 64),
        'vocab_size': data.get('vocab_size', 1000),
        'max_seq_len': data.get('max_seq_len', 128),
        'd_model': data.get('d_model', 512),
        'n_layers': data.get('n_layers', 6),
        'n_heads': data.get('n_heads', 8),
        'd_ff': data.get('d_ff', 2048),
        'dropout': data.get('dropout', 0.1),
        'use_moe': data.get('use_moe', False),
        'use_mla': data.get('use_mla', False),
    }
    
    train_params = {
        'data_path': data_path,
        'learning_rate': data.get('learning_rate', 0.0001),
        'epochs': data.get('epochs', 10),
        'batch_size': data.get('batch_size', 32),
    }
    
    if train_type == 'image':
        thread = __import__('threading').Thread(target=train_image_model, args=(
            str(user_id), task_id, model_params, train_params, training_tasks
        ))
    else:
        thread = __import__('threading').Thread(target=train_text_model, args=(
            str(user_id), task_id, model_params, train_params, training_tasks
        ))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'success',
        'task_id': task_id,
        'message': f'{train_type} 训练已启动 (task_id: {task_id})'
    })

@training_bp.route('/task_status/<task_id>')
@login_required
def api_task_status(task_id):
    """查询训练任务状态（和原有逻辑完全一致）"""
    task = training_tasks.get(task_id)
    if not task:
        return jsonify({'status': 'not_found', 'message': '任务不存在'}), 404
    
    return jsonify({
        'status': task.get('status', 'unknown'),
        'progress': task.get('progress', 0),
        'loss': task.get('loss', None),
        'accuracy': task.get('accuracy', None),
        'message': task.get('message', '')
    })