"""database.py — SQLite3 数据库管理（用户、文件、模型）"""
import sqlite3
import hashlib
import os
import time
from datetime import datetime

DB_PATH = 'training_platform.db'


def get_db():
    """获取数据库连接（每次调用返回新连接，线程安全）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 支持按列名访问
    conn.execute("PRAGMA journal_mode=WAL")  # 提高并发性能
    conn.execute("PRAGMA foreign_keys=ON")    # 启用外键约束
    return conn


def init_db():
    """初始化数据库表结构"""
    conn = get_db()
    cursor = conn.cursor()
    
    # ---- 用户表 ----
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # ---- 上传文件记录表 ----
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            train_type TEXT NOT NULL DEFAULT 'image',
            file_size INTEGER DEFAULT 0,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    # ---- 训练模型记录表 ----
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            file_size INTEGER DEFAULT 0,
            file_path TEXT,
            accuracy REAL,
            loss REAL,
            epochs INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[DB] 数据库初始化完成: {DB_PATH}")


# ==================== 用户操作 ====================

def hash_password(password):
    """使用 SHA-256 哈希密码（生产环境建议用 bcrypt）"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def register_user(username, password):
    """注册新用户，返回 (success, message)"""
    if not username or not password:
        return False, '用户名和密码不能为空'
    if len(username) < 2 or len(username) > 20:
        return False, '用户名长度应为 2-20 个字符'
    if len(password) < 4:
        return False, '密码长度至少 4 个字符'
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (username, password_hash) VALUES (?, ?)',
            (username, hash_password(password))
        )
        conn.commit()
        return True, '注册成功'
    except sqlite3.IntegrityError:
        return False, '用户名已存在'
    finally:
        conn.close()


def verify_user(username, password):
    """验证用户登录，返回 (success, user_id 或 None)"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, password_hash FROM users WHERE username = ?',
        (username,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return False, None
    
    stored_hash = row['password_hash']
    if stored_hash == hash_password(password):
        # 更新最后登录时间
        conn = get_db()
        conn.execute(
            'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?',
            (row['id'],)
        )
        conn.commit()
        conn.close()
        return True, row['id']
    return False, None


def get_user_by_id(user_id):
    """根据 ID 获取用户信息"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, created_at, last_login FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


# ==================== 文件操作 ====================

def save_file_record(user_id, filename, original_name, file_size, file_path, train_type='image'):
    """保存文件上传记录"""
    conn = get_db()
    conn.execute(
        '''INSERT INTO files (user_id, filename, original_name, file_size, file_path, train_type)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (user_id, filename, original_name, file_size, file_path, train_type)
    )
    conn.commit()
    conn.close()


def get_user_files(user_id, train_type=None):
    """获取用户的上传文件列表"""
    conn = get_db()
    if train_type:
        rows = conn.execute(
            'SELECT * FROM files WHERE user_id = ? AND train_type = ? ORDER BY created_at DESC',
            (user_id, train_type)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT * FROM files WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_file_count(user_id, train_type=None):
    """获取用户上传文件数量"""
    conn = get_db()
    if train_type:
        count = conn.execute(
            'SELECT COUNT(*) as cnt FROM files WHERE user_id = ? AND train_type = ?',
            (user_id, train_type)
        ).fetchone()['cnt']
    else:
        count = conn.execute(
            'SELECT COUNT(*) as cnt FROM files WHERE user_id = ?',
            (user_id,)
        ).fetchone()['cnt']
    conn.close()
    return count


# ==================== 模型操作 ====================

def save_model_record(user_id, model_name, model_type, file_size, file_path, accuracy=None, loss=None, epochs=None):
    """保存训练模型记录"""
    conn = get_db()
    conn.execute(
        '''INSERT INTO models (user_id, model_name, model_type, file_size, file_path, accuracy, loss, epochs)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (user_id, model_name, model_type, file_size, file_path, accuracy, loss, epochs)
    )
    conn.commit()
    conn.close()


def get_user_models(user_id):
    """获取用户的所有训练模型"""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM models WHERE user_id = ? ORDER BY created_at DESC',
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_model_record(model_name, user_id):
    """删除模型记录"""
    conn = get_db()
    conn.execute(
        'DELETE FROM models WHERE model_name = ? AND user_id = ?',
        (model_name, user_id)
    )
    conn.commit()
    conn.close()


def get_model_count(user_id):
    """获取用户的模型数量"""
    conn = get_db()
    count = conn.execute(
        'SELECT COUNT(*) as cnt FROM models WHERE user_id = ?',
        (user_id,)
    ).fetchone()['cnt']
    conn.close()
    return count