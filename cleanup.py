"""
cleanup.py — 存储清理机制（管理员可配）

功能：
1. 自动清理：后台线程按可配置的保留天数，定期清理过期的训练产物
   （模型文件、上传数据）与轮转日志；天数=0 表示不自动清理。
2. 手动清理：管理员面板一键清除模型/上传数据/日志。
3. 状态查看：各类文件的占用体积与数量。

配置持久化在 <项目根>/instance/cleanup_config.json：
{
  "retention_days": 30,      # 0 = 关闭自动清理
  "auto_clean_models": true,
  "auto_clean_uploads": true
}
"""
import json
import os
import shutil
import threading
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
CONFIG_PATH = os.path.join(INSTANCE_DIR, 'cleanup_config.json')

_lock = threading.Lock()
_scheduler_started = False


# ==================== 配置 ====================
def get_cleanup_config():
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    return {
        'retention_days': int(cfg.get('retention_days', 0)),
        'auto_clean_models': bool(cfg.get('auto_clean_models', True)),
        'auto_clean_uploads': bool(cfg.get('auto_clean_uploads', True)),
    }


def set_cleanup_config(retention_days=None, auto_clean_models=None, auto_clean_uploads=None):
    cfg = get_cleanup_config()
    if retention_days is not None:
        cfg['retention_days'] = max(0, int(retention_days))
    if auto_clean_models is not None:
        cfg['auto_clean_models'] = bool(auto_clean_models)
    if auto_clean_uploads is not None:
        cfg['auto_clean_uploads'] = bool(auto_clean_uploads)
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


# ==================== 目录定位 ====================
def _models_root():
    return os.path.join(BASE_DIR, 'models')


def _uploads_root():
    return os.path.join(BASE_DIR, 'uploads')


def _logs_dir():
    return os.path.join(BASE_DIR, 'logs')


# ==================== 统计 / 清理 ====================
def _dir_stats(root):
    total_size, file_count = 0, 0
    if not os.path.isdir(root):
        return 0, 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            try:
                total_size += os.path.getsize(os.path.join(dirpath, name))
                file_count += 1
            except OSError:
                pass
    return total_size, file_count


def cleanup_status():
    """各存储区域的占用情况（供管理页展示）"""
    result = {}
    for key, root in (('models', _models_root()), ('uploads', _uploads_root())):
        size, count = _dir_stats(root)
        result[key] = {'size': size, 'count': count}
    log_dir = _logs_dir()
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')] \
        if os.path.isdir(log_dir) else []
    result['logs'] = {
        'size': sum(os.path.getsize(os.path.join(log_dir, f)) for f in log_files),
        'count': len(log_files),
    }
    cfg = get_cleanup_config()
    result['config'] = cfg
    return result


def _remove_expired(root, retention_days):
    """删除 root 下修改时间早于保留期的文件，返回 (删除数, 释放字节)"""
    if not os.path.isdir(root) or retention_days <= 0:
        return 0, 0
    cutoff = time.time() - retention_days * 86400
    removed, freed = 0, 0
    for dirpath, dirs, files in os.walk(root, topdown=False):
        for name in files:
            p = os.path.join(dirpath, name)
            try:
                if os.path.getmtime(p) < cutoff:
                    freed += os.path.getsize(p)
                    os.remove(p)
                    removed += 1
            except OSError:
                pass
        # 自底向上移除空目录（保留根目录）
        for d in dirs:
            full = os.path.join(dirpath, d)
            try:
                if not os.listdir(full):
                    os.rmdir(full)
            except OSError:
                pass
    return removed, freed


def run_cleanup(targets=None, retention_days=None):
    """
    执行清理。targets 为 None 时按配置执行自动清理；
    手动调用时传列表，如 ['models', 'uploads', 'logs']。
    返回 {target: {removed, freed}} 明细。
    """
    cfg = get_cleanup_config()
    days = cfg['retention_days'] if retention_days is None else max(0, int(retention_days))
    detail = {}

    if targets is None:
        targets = []
        if days > 0 and cfg['auto_clean_models']:
            targets.append('models')
        if days > 0 and cfg['auto_clean_uploads']:
            targets.append('uploads')

    for t in targets:
        if t == 'logs':
            removed, freed = _clear_logs()
        else:
            root = _models_root() if t == 'models' else _uploads_root()
            removed, freed = (_remove_expired(root, days) if days > 0 else (0, 0))
            # 手动清理（days==0 且显式指定 target）视为全量清空该区域
            if days == 0 and targets is not None and t in targets:
                r2, f2 = _purge_dir(root)
                removed += r2
                freed += f2
        detail[t] = {'removed': removed, 'freed': freed}
    return detail


def _purge_dir(root):
    """清空整个目录内容（保留目录本身），返回 (删除数, 释放字节)"""
    if not os.path.isdir(root):
        return 0, 0
    removed, freed = 0, 0
    for name in os.listdir(root):
        p = os.path.join(root, name)
        try:
            if os.path.isdir(p):
                size, count = _dir_stats(p)
                shutil.rmtree(p)
                removed += count
                freed += size
            else:
                freed += os.path.getsize(p)
                os.remove(p)
                removed += 1
        except OSError:
            pass
    return removed, freed


def _clear_logs():
    """清空 logs/ 下所有日志内容（truncate 而非删除，保持文件句柄有效）"""
    log_dir = _logs_dir()
    cleared, freed = 0, 0
    if os.path.isdir(log_dir):
        for name in os.listdir(log_dir):
            p = os.path.join(log_dir, name)
            try:
                if os.path.isfile(p):
                    freed += os.path.getsize(p)
                    with open(p, 'r+b') as fh:
                        fh.truncate(0)
                    cleared += 1
            except OSError:
                pass
    return cleared, freed


# ==================== 后台自动清理 ====================
def start_auto_clean_scheduler(interval_seconds=3600):
    """启动后台守护线程定期执行自动清理（幂等）"""
    global _scheduler_started
    with _lock:
        if _scheduler_started:
            return
        _scheduler_started = True

    def _loop():
        while True:
            try:
                detail = run_cleanup()
                cleaned = {k: v for k, v in detail.items() if v['removed']}
                if cleaned:
                    print(f"[Cleanup] 自动清理完成 {datetime.now():%Y-%m-%d %H:%M} -> {cleaned}")
            except Exception as e:
                print(f"[Cleanup] 自动清理异常: {e}")
            time.sleep(interval_seconds)

    threading.Thread(target=_loop, daemon=True, name='cleanup-scheduler').start()
