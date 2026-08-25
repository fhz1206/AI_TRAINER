"""database.py — SQLite3 数据库管理（用户、文件、模型）"""
import json
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
            role TEXT NOT NULL DEFAULT 'user',
            group_name TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    # 确保 admin 用户存在且为管理员
    admin = cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',)).fetchone()
    if admin:
        cursor.execute('UPDATE users SET role = ? WHERE username = ?', ('admin', 'admin'))
        print(f"[DB] 管理员账户状态已确认")
    else:
        cursor.execute(
            'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
            ('admin', hash_password('123456'), 'admin')
        )
        print(f"[DB] 管理员账户已创建（admin / 123456）")
    
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
    
    # ---- 行为日志表 ----
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            activity_type TEXT NOT NULL,
            description TEXT NOT NULL,
            detail TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_activity_user ON activity_logs(user_id, created_at)')
    # 支持按时间的范围过滤（如清理/统计场景）；主列表查询走 rowid(id) 倒序，无需此索引
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_logs(created_at)')
    
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


# ==================== 管理员操作 ====================

def is_admin(user_id):
    """检查用户是否为管理员（基于 role 判断）"""
    conn = get_db()
    row = conn.execute('SELECT role FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if row and row['role'] == 'admin':
        return True
    return False


def get_all_users():
    """获取所有用户列表（管理员用）"""
    conn = get_db()
    rows = conn.execute(
        'SELECT id, username, role, group_name, created_at, last_login FROM users ORDER BY id'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def admin_update_user_role(user_id, new_role):
    """管理员修改用户角色"""
    conn = get_db()
    conn.execute('UPDATE users SET role = ? WHERE id = ?', (new_role, user_id))
    conn.commit()
    conn.close()
    return True


def admin_update_user_group(user_id, group_name):
    """管理员修改用户分组"""
    conn = get_db()
    conn.execute('UPDATE users SET group_name = ? WHERE id = ?', (group_name, user_id))
    conn.commit()
    conn.close()
    return True


def admin_delete_user(user_id):
    """管理员删除用户（级联删除关联数据）"""
    conn = get_db()
    conn.execute('DELETE FROM users WHERE id = ? AND username != ?', (user_id, 'admin'))
    conn.commit()
    conn.close()


def admin_reset_password(user_id, new_password):
    """管理员重置用户密码"""
    if len(new_password) < 4:
        return False, '密码长度至少 4 个字符'
    conn = get_db()
    conn.execute(
        'UPDATE users SET password_hash = ? WHERE id = ? AND username != ?',
        (hash_password(new_password), user_id, 'admin')
    )
    conn.commit()
    conn.close()
    return True, '密码已重置'


# 日志查询的显式列清单（避免 SELECT * 拷贝不可控的大字段）
_LOG_COLUMNS = (
    "a.id, a.user_id, a.activity_type, a.description, "
    "a.detail, a.created_at"
)


def get_all_activity_logs_paged(page=1, page_size=100):
    """分页获取行为日志（管理员用）：LIMIT/OFFSET 每次只取一页，
    防止一次性把全部日志拉进内存。返回 dict 含 logs/total/page/page_size。
    """
    page = max(1, int(page))
    page_size = min(max(1, int(page_size)), 500)   # 单页上限保护
    offset = (page - 1) * page_size
    conn = get_db()
    try:
        total = conn.execute(
            'SELECT COUNT(*) FROM activity_logs').fetchone()[0]
        rows = conn.execute(
            "SELECT " + _LOG_COLUMNS + ", u.username FROM activity_logs a "
            "JOIN users u ON a.user_id = u.id "
            "ORDER BY a.id DESC LIMIT ? OFFSET ?",
            (page_size, offset)
        ).fetchall()
    finally:
        conn.close()
    return {
        'logs': [dict(r) for r in rows],
        'total': total,
        'page': page,
        'page_size': page_size,
    }


# ==================== 行为日志操作 ====================

# ==================== 日志存储上限（管理员可调；-1 = 无上限） ====================
_LOG_LIMIT_DEFAULT = 1000
_LOG_LIMIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'instance', 'log_limits.json')


def get_log_limit():
    """读取日志存储上限；-1 表示无上限。未配置时默认 1000 条。"""
    try:
        with open(_LOG_LIMIT_PATH, encoding='utf-8') as f:
            cfg = json.load(f)
        return int(cfg.get('max_logs', _LOG_LIMIT_DEFAULT))
    except (OSError, ValueError):
        return _LOG_LIMIT_DEFAULT


def set_log_limit(max_logs):
    """设置日志存储上限（-1=无上限，>=0 为保留的最新条数），并立即裁剪存量。

    返回生效的上限值；参数非法时抛 ValueError。
    """
    max_logs = int(max_logs)
    if max_logs != -1 and max_logs < 0:
        raise ValueError('日志上限应为 -1 或非负整数')
    os.makedirs(os.path.dirname(_LOG_LIMIT_PATH), exist_ok=True)
    with open(_LOG_LIMIT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'max_logs': max_logs}, f, ensure_ascii=False)
    if max_logs >= 0:
        _trim_activity_logs(max_logs)
    return get_log_limit()


def _trim_activity_logs(keep):
    """把行为日志裁剪到最新的 keep 条（按 id 主键倒序保留），返回删除行数。

    利用主键有序性：找到第 keep 新的边界 id，一次范围删除其之前的所有行，
    避免对全表做 NOT IN 集合运算。
    """
    if keep is None or keep < 0:
        return 0
    conn = get_db()
    try:
        if keep == 0:
            cur = conn.execute('DELETE FROM activity_logs')
        else:
            cur = conn.execute(
                'DELETE FROM activity_logs '
                'WHERE id < (SELECT id FROM activity_logs '
                '            ORDER BY id DESC LIMIT 1 OFFSET ?)',
                (keep - 1,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def add_activity_log(user_id, activity_type, description, detail=''):
    """添加一条行为日志；若配置了上限，同事务内删除最旧的日志（新增一条删一条）"""
    conn = get_db()
    try:
        conn.execute(
            'INSERT INTO activity_logs (user_id, activity_type, description, detail) VALUES (?, ?, ?, ?)',
            (user_id, activity_type, description, detail)
        )
        limit = get_log_limit()
        if limit >= 0:
            if limit == 0:
                conn.execute('DELETE FROM activity_logs')
            else:
                conn.execute(
                    'DELETE FROM activity_logs '
                    'WHERE id < (SELECT id FROM activity_logs '
                    '            ORDER BY id DESC LIMIT 1 OFFSET ?)',
                    (limit - 1,))
        conn.commit()
    finally:
        conn.close()


def get_user_activity_logs(user_id, limit=50):
    """获取用户的行为日志，最新在前（走 idx_activity_user 的 user_id 前缀过滤）"""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT " + _LOG_COLUMNS + " FROM activity_logs a "
            "WHERE a.user_id = ? ORDER BY a.id DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ==================== 用户信息修改 ====================

def update_username(user_id, new_username):
    """修改用户名，返回 (success, message)"""
    if not new_username or len(new_username) < 2 or len(new_username) > 20:
        return False, '用户名长度应为 2-20 个字符'
    conn = get_db()
    try:
        conn.execute('UPDATE users SET username = ? WHERE id = ?', (new_username, user_id))
        conn.commit()
        return True, '用户名修改成功'
    except sqlite3.IntegrityError:
        return False, '用户名已存在'
    finally:
        conn.close()


def update_password(user_id, old_password, new_password):
    """修改密码，返回 (success, message)"""
    if len(new_password) < 4:
        return False, '新密码长度至少 4 个字符'
    conn = get_db()
    row = conn.execute(
        'SELECT password_hash FROM users WHERE id = ?', (user_id,)
    ).fetchone()
    if not row or row['password_hash'] != hash_password(old_password):
        conn.close()
        return False, '原密码错误'
    conn.execute(
        'UPDATE users SET password_hash = ? WHERE id = ?',
        (hash_password(new_password), user_id)
    )
    conn.commit()
    conn.close()
    return True, '密码修改成功'