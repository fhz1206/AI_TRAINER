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

# ==================== 全局请求日志系统（RotatingFileHandler 轮转 + 内存友好） ====================
# 实现已抽取到独立模块：磁盘有界（单文件5MB×5备份）、不缓冲请求/响应体、不记录敏感头
from app_logger import init_app_logger

init_app_logger(app)

# 后台自动清理调度（按管理员配置的保留天数清理过期文件；守护线程，随主进程退出）
from cleanup import start_auto_clean_scheduler
start_auto_clean_scheduler()

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