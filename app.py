"""
app.py — 统一训练平台主入口
"""
import os
import sys
# 核心依赖在启动阶段逐步导入

# ==================== 启动进度条 + 模块加载 ====================
# 总进度100%，分7个阶段对应不同的加载步骤
try:
    print("⏳ 平台启动中...")

    # --------------------- 阶段1：加载核心依赖（0→15%） ---------------------
    print("  [1/7] 加载核心依赖...")
    from flask import Flask, render_template, request, jsonify, session, redirect, send_file
    from flask_cors import CORS
    from threading import Thread
    from uuid import uuid4
    from werkzeug.utils import secure_filename
    from time import time as time_now
    from traceback import format_exc
    from urllib.parse import unquote
    import base64
    import shutil
    import zipfile
    import tempfile
    from PIL import Image as PILImage

    # --------------------- 阶段2：加载全局配置（15→30%） ---------------------
    print("  [2/7] 加载全局配置...")
    from config import Config

    # --------------------- 阶段3：加载数据库模块（30→50%） ---------------------
    print("  [3/7] 加载数据库模块...")
    from database import init_db, register_user, verify_user, get_user_by_id, get_db
    from database import save_file_record, get_user_files, get_file_count
    from database import save_model_record, get_user_models, delete_model_record, get_model_count

    # --------------------- 阶段4：加载训练核心模块（50→65%） ---------------------
    print("  [4/7] 加载训练核心模块...")
    from trainer import train_image_model, train_text_model

    # --------------------- 阶段5：加载业务蓝图模块（65→85%） ---------------------
    print("  [5/7] 加载业务模块...")
    from state import training_tasks
    from blueprints.main import main_bp
    from blueprints.auth import auth_bp
    from blueprints.training import training_bp
    from blueprints.model import model_bp
    from blueprints.test import test_bp
    from blueprints.study import study_bp
    from blueprints.profile import profile_bp
    from blueprints.admin import admin_bp

    # --------------------- 阶段6：初始化应用（85→95%） ---------------------
    print("  [6/7] 初始化应用服务...")
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app, supports_credentials=True)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TEST_DATA_FOLDER'], exist_ok=True)

    init_db()

    # --------------------- 阶段7：注册路由（95→100%） ---------------------
    print("  [7/7] 注册路由...")
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(test_bp)
    app.register_blueprint(study_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(admin_bp)

    # --------------------- 启动完成 ---------------------
    print("\n" + "="*50)
    print("✅ 统一训练平台启动")
    print(f"📁 上传目录: {app.config['UPLOAD_FOLDER']}")
    print(f"💾 模型目录: {app.config['MODEL_FOLDER']}")
    print(f"🗄️  数据库: training_platform.db")
    print(f"🔗 访问地址: http://127.0.0.1:5000")
    print("="*50)

except Exception as e:
    import traceback as _tb
    print(f"\n❌ 启动失败: {e}")
    _tb.print_exc()
    print("\n按 Enter 键退出...")
    input()
    sys.exit(1)

# ==================== 全局请求日志系统（写入单个 .log 文件） ====================
import logging as _logging
from datetime import datetime as _datetime
from flask import request as _flask_request
import json as _json
import time as _time

_logger = _logging.getLogger('app_access')
_logger.setLevel(_logging.INFO)
_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log')
_log_handler = _logging.FileHandler(_log_file, encoding='utf-8')
_log_handler.setFormatter(_logging.Formatter('%(message)s'))
_logger.handlers.clear()
_logger.addHandler(_log_handler)
_logger.propagate = False

@app.before_request
def _log_before_request():
    _flask_request._request_start_time = _time.time()
    # 记录请求体大小（用于上行带宽统计）
    try:
        req_data = _flask_request.get_data()
        _flask_request._request_bytes = len(req_data)
    except:
        _flask_request._request_bytes = 0

@app.after_request
def _log_after_request(response):
    start_time = getattr(_flask_request, '_request_start_time', _time.time())
    duration_ms = (_time.time() - start_time) * 1000
    now_str = _datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    # 带宽统计（上行+下行）
    try:
        from flask import session as _session
        uid = _session.get('user_id')
        if uid:
            req_bytes = getattr(_flask_request, '_request_bytes', 0)
            resp_bytes = len(response.get_data() or b'')
            total_bytes = req_bytes + resp_bytes
            from state import track_bandwidth
            exceeded, cur_mbps, limit = track_bandwidth(uid, total_bytes)
            if exceeded:
                # 超出带宽限制，添加响应头标记
                response.headers['X-RateLimit-Limit'] = str(limit)
                response.headers['X-RateLimit-Remaining'] = '0'
    except:
        pass

    # 请求体
    body = ''
    content_type = _flask_request.content_type or ''
    if _flask_request.method in ('POST', 'PUT', 'PATCH'):
        if 'application/json' in content_type:
            body = _flask_request.get_data(as_text=True)[:2000]
        elif 'multipart/form-data' in content_type:
            body = f'<multipart form, {len(_flask_request.get_data() or b"")} bytes>'
        else:
            body = _flask_request.get_data(as_text=True)[:1000]

    # 响应体
    resp_body = ''
    try:
        resp_body = response.get_data(as_text=True)[:2000]
    except:
        resp_body = f'<binary, {response.content_length or 0} bytes>'

    # 用户信息
    user_id = ''
    try:
        from flask import session as _session
        user_id = str(_session.get('user_id', ''))
    except:
        pass

    log_lines = [
        f'[REQUEST]  {now_str}',
        f'  IP:       {_flask_request.remote_addr}',
        f'  User:     {user_id or "(anonymous)"}',
        f'  Method:   {_flask_request.method}',
        f'  Path:     {_flask_request.full_path}',
        f'  Headers:  {dict(_flask_request.headers)}',
        f'  Body:     {body}',
        f'[RESPONSE] {response.status_code} ({duration_ms:.1f}ms)',
        f'  Content:  {resp_body}',
        f'{"="*80}',
    ]
    _logger.info('\n'.join(log_lines))
    return response

# ==================== 服务启动 ====================
if __name__ == '__main__':
    try:
        app.run(debug=app.config['DEBUG'], port=5000, host='0.0.0.0')
    except Exception as e:
        import traceback as _tb
        print(f"\n❌ 启动失败: {e}")
        _tb.print_exc()
    finally:
        print("\n服务已停止，按 Enter 键退出...")
        input()