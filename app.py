"""
app.py — 统一训练平台主入口
带启动进度条，纯后端改动，前端零修改
"""
import os
import sys
from tqdm import tqdm
# 核心依赖导入放到进度条步骤中，方便展示加载进度

# ==================== 启动进度条 + 模块加载 ====================
# 总进度100%，分7个阶段对应不同的加载步骤
with tqdm(
    total=100,
    desc="平台启动中",
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ncols=100,
    leave=True,
    dynamic_ncols=True  # 自动适配终端宽度
) as pbar:
    # --------------------- 阶段1：加载核心依赖（0→15%） ---------------------
    pbar.set_description("⏳ 加载核心依赖...")
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
    pbar.update(15)

    # --------------------- 阶段2：加载全局配置（15→30%） ---------------------
    pbar.set_description("⏳ 加载全局配置...")
    from config import Config
    pbar.update(15)

    # --------------------- 阶段3：加载数据库模块（30→50%） ---------------------
    pbar.set_description("⏳ 加载数据库模块...")
    from database import init_db, register_user, verify_user, get_user_by_id, get_db
    from database import save_file_record, get_user_files, get_file_count
    from database import save_model_record, get_user_models, delete_model_record, get_model_count
    pbar.update(20)

    # --------------------- 阶段4：加载训练核心模块（50→65%） ---------------------
    pbar.set_description("⏳ 加载训练核心模块...")
    from trainer import train_image_model, train_text_model
    pbar.update(15)

    # --------------------- 阶段5：加载业务蓝图模块（65→85%） ---------------------
    pbar.set_description("⏳ 加载业务模块...")
    from state import training_tasks  # 导入全局训练状态
    from blueprints.main import main_bp
    from blueprints.auth import auth_bp
    from blueprints.training import training_bp
    from blueprints.model import model_bp
    from blueprints.test import test_bp
    pbar.update(20)

    # --------------------- 阶段6：初始化应用（85→95%） ---------------------
    pbar.set_description("⏳ 初始化应用服务...")
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app, supports_credentials=True)

    # 创建必要目录
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TEST_DATA_FOLDER'], exist_ok=True)

    # 初始化数据库
    init_db()
    pbar.update(10)

    # --------------------- 阶段7：注册路由（95→100%） ---------------------
    pbar.set_description("⏳ 注册路由...")
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(test_bp)
    pbar.update(5)

    # --------------------- 启动完成 ---------------------
    pbar.set_description("✅ 启动完成")
    # 进度条走完后打印启动信息
    print("\n" + "="*50)
    print("🧠 统一训练平台启动")
    print(f"📁 上传目录: {app.config['UPLOAD_FOLDER']}")
    print(f"💾 模型目录: {app.config['MODEL_FOLDER']}")
    print(f"🗄️  数据库: training_platform.db")
    print(f"🔗 访问地址: http://127.0.0.1:5000")
    print("="*50)

# ==================== 服务启动 ====================
if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], port=5000, host='0.0.0.0')