import os

basedir = os.path.abspath(os.path.dirname(__file__))

# ==================== 全局常量（支持直接导入） ====================
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
TEXT_EXTENSIONS = {'.txt'}

class Config:
    # 基础配置
    SECRET_KEY = 'your-secret-key-here-change-in-production'  # 生产环境请用环境变量替换
    DEBUG = True
    
    # 目录配置
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    MODEL_FOLDER = os.path.join(basedir, 'models')
    TEST_DATA_FOLDER = os.path.join(basedir, 'test_data')
    
    # 上传限制
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 最大上传500MB

# 生产环境可切换为ProductionConfig
class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-production-secret-key'