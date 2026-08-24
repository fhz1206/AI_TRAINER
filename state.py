# 全局训练任务状态（内存存储，重启丢失，后续可扩展为Redis等持久化）
from threading import Lock
from time import time as time_now

training_tasks = {}

# ==================== 训练队列系统 ====================
# 最大并发训练数（默认5，可在管理员面板修改）
_max_concurrent = 5
_concurrent_lock = Lock()
_queue = []  # 队列中的任务 [(task_id, user_id, params, enqueue_time), ...]
_active_tasks = set()  # 当前正在训练的任务ID集合
_pending_starters = {}  # 排队任务的线程启动器：被调度到时由状态层拉起


def get_max_concurrent():
    """获取最大并发训练数"""
    global _max_concurrent
    return _max_concurrent


def set_max_concurrent(n):
    """设置最大并发训练数（管理员用）"""
    global _max_concurrent
    if n < 1:
        n = 1
    _max_concurrent = n


def enqueue_task(task_id, user_id, params):
    """将任务加入队列，返回 (success, message, position)"""
    global _queue, _active_tasks, _max_concurrent
    with _concurrent_lock:
        if task_id in _active_tasks:
            return False, '任务已在运行中', -1
        if any(t[0] == task_id for t in _queue):
            return False, '任务已在队列中', -1
        _queue.append((task_id, user_id, params, time_now()))
        # 尝试启动队列中的任务
        _try_start_next()
        position = len(_queue)
        return True, '已加入队列', position


def _try_start_next():
    """尝试启动队列中的下一个任务"""
    global _queue, _active_tasks, _max_concurrent
    while len(_active_tasks) < _max_concurrent and _queue:
        task_id, user_id, params, enqueue_time = _queue.pop(0)
        if task_id in _active_tasks:
            continue
        # 标记为活跃，实际启动由调用方处理
        _active_tasks.add(task_id)
        from state import training_tasks
        training_tasks[task_id] = {
            'status': 'queued',
            'progress': 0,
            'loss': None,
            'accuracy': None,
            'message': '⏳ 等待队列调度...',
            'queue_position': 0,
            'user_id': user_id,
            'params': params,
            'enqueue_time': enqueue_time,
        }
        # 拉起该任务在蓝图层注册的等待线程（若已注册）
        starter = _pending_starters.pop(task_id, None)
        if starter:
            try:
                starter()
            except Exception:
                pass
        return True  # 通知调用方可以启动
    return False


def can_start_task(task_id):
    """检查任务是否可以启动（被调度到）"""
    global _active_tasks
    with _concurrent_lock:
        return task_id in _active_tasks


def mark_task_done(task_id):
    """标记任务完成（从活跃集合中移除，启动下一个）"""
    global _active_tasks, _queue, _max_concurrent
    with _concurrent_lock:
        if task_id in _active_tasks:
            _active_tasks.remove(task_id)
        # 尝试启动下一个
        _try_start_next()


def get_queue_status():
    """获取队列状态（管理员用）"""
    global _queue, _active_tasks, _max_concurrent
    with _concurrent_lock:
        active = []
        for tid in _active_tasks:
            task = training_tasks.get(tid, {})
            active.append({
                'task_id': tid,
                'user_id': task.get('user_id'),
                'params': task.get('params', {}),
                'progress': task.get('progress', 0),
                'status': task.get('status', 'running'),
                'enqueue_time': task.get('enqueue_time'),
            })
        waiting = []
        for item in _queue:
            waiting.append({
                'task_id': item[0],
                'user_id': item[1],
                'params': item[2],
                'enqueue_time': item[3],
            })
        return {
            'max_concurrent': _max_concurrent,
            'active_count': len(_active_tasks),
            'queue_length': len(_queue),
            'active': active,
            'waiting': waiting,
        }


def register_pending_starter(task_id, starter):
    """
    登记"已入队但尚未启动"的任务线程启动器。
    返回 False 表示任务已被直接调度（调用方应立即自行启动线程）；
    返回 True 表示已登记——任务被调度到时会由 _try_start_next 自动拉起。
    """
    with _concurrent_lock:
        if task_id in _active_tasks:
            return False
        _pending_starters[task_id] = starter
        return True


def discard_pending_starter(task_id):
    """注销启动器（取消任务等场景），避免残留引用"""
    with _concurrent_lock:
        _pending_starters.pop(task_id, None)


def cancel_task(task_id):
    """取消队列中的任务（管理员用）"""
    global _queue, _active_tasks
    with _concurrent_lock:
        # 尝试从队列中移除
        for i, item in enumerate(_queue):
            if item[0] == task_id:
                _queue.pop(i)
                _pending_starters.pop(task_id, None)  # 取消排队任务时同步清理启动器
                if task_id in training_tasks:
                    training_tasks[task_id] = {
                        'status': 'cancelled',
                        'progress': 0,
                        'loss': None,
                        'accuracy': None,
                        'message': '❌ 已被管理员取消',
                    }
                return True, '任务已取消'
        # 检查是否在活跃中
        if task_id in _active_tasks:
            # 标记为取消，实际终止由调用方处理
            if task_id in training_tasks:
                training_tasks[task_id] = {
                    'status': 'cancelling',
                    'progress': 0,
                    'loss': None,
                    'accuracy': None,
                    'message': '🛑 管理员正在强制停止...',
                }
            return True, '正在强制停止...'
        return False, '任务不存在'


# ==================== 带宽限制系统 ====================
_default_bandwidth_mbps = 10
_bandwidth_lock = Lock()
_user_bandwidth = {}
_user_bandwidth_limit = {}
_WINDOW_SECONDS = 1


def get_default_bandwidth():
    return _default_bandwidth_mbps


def set_default_bandwidth(mbps):
    global _default_bandwidth_mbps
    if mbps < 0.1:
        mbps = 0.1
    _default_bandwidth_mbps = mbps


def get_user_bandwidth_limit(user_id):
    return _user_bandwidth_limit.get(str(user_id), _default_bandwidth_mbps)


def set_user_bandwidth_limit(user_id, mbps):
    if mbps < 0.1:
        mbps = 0.1
    _user_bandwidth_limit[str(user_id)] = mbps


def track_bandwidth(user_id, bytes_count):
    """
    记录用户带宽使用，返回 (是否超出限制, 当前速率Mbps, 限制Mbps)
    滑动窗口算法，窗口1秒
    """
    now = time_now()
    uid = str(user_id)
    with _bandwidth_lock:
        if uid not in _user_bandwidth:
            _user_bandwidth[uid] = []
        records = _user_bandwidth[uid]
        while records and now - records[0]['time'] > _WINDOW_SECONDS:
            records.pop(0)
        records.append({'time': now, 'bytes': bytes_count})
        total_bytes = sum(r['bytes'] for r in records)
        limit = get_user_bandwidth_limit(uid)
        current_mbps = (total_bytes * 8) / 1e6
        return current_mbps > limit, round(current_mbps, 2), limit


def get_bandwidth_stats():
    """获取所有用户的带宽统计（管理员用）"""
    with _bandwidth_lock:
        now = time_now()
        stats = []
        for uid, records in list(_user_bandwidth.items()):
            while records and now - records[0]['time'] > _WINDOW_SECONDS:
                records.pop(0)
            total_bytes = sum(r['bytes'] for r in records) if records else 0
            current_mbps = (total_bytes * 8) / 1e6 if records else 0
            limit = _user_bandwidth_limit.get(uid, _default_bandwidth_mbps)
            # 获取用户名
            username = uid
            try:
                from database import get_db
                conn = get_db()
                row = conn.execute('SELECT username FROM users WHERE id = ?', (int(uid),)).fetchone()
                conn.close()
                if row:
                    username = row['username']
            except:
                pass
            stats.append({
                'user_id': uid,
                'username': username,
                'current_mbps': round(current_mbps, 2),
                'limit_mbps': limit,
                'exceeded': current_mbps > limit,
            })
        return {
            'default_limit_mbps': _default_bandwidth_mbps,
            'users': stats,
        }