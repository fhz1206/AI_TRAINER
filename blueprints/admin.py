"""admin.py — 管理员面板蓝图"""
import os
import platform
import psutil
from flask import Blueprint, render_template, request, jsonify, session, redirect
from blueprints.utils import login_required
from database import is_admin, get_all_users, get_user_by_id
from database import admin_update_user_role, admin_update_user_group, admin_delete_user
from database import admin_reset_password, get_all_activity_logs_paged, add_activity_log
from database import hash_password, get_db
from state import get_queue_status, set_max_concurrent, cancel_task, training_tasks
from state import get_bandwidth_stats, get_default_bandwidth, set_default_bandwidth, set_user_bandwidth_limit

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


def admin_required(f):
    """管理员权限装饰器"""
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        if not is_admin(session['user_id']):
            return jsonify({'status': 'error', 'message': '权限不足'}), 403
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


@admin_bp.route('')
@login_required
@admin_required
def admin_panel():
    """管理员面板首页"""
    user = get_user_by_id(session['user_id'])
    return render_template('admin.html', username=user['username'])


@admin_bp.route('/api/users')
@login_required
@admin_required
def api_get_users():
    """获取所有用户列表"""
    users = get_all_users()
    return jsonify({'status': 'success', 'users': users})


@admin_bp.route('/api/user/<int:user_id>/role', methods=['POST'])
@login_required
@admin_required
def api_set_role(user_id):
    """设置用户角色"""
    data = request.json
    new_role = data.get('role', 'user')
    if new_role not in ('admin', 'user'):
        return jsonify({'status': 'error', 'message': '无效的角色'}), 400
    admin_update_user_role(user_id, new_role)
    uname = get_user_by_id(user_id)
    add_activity_log(session['user_id'], 'admin', f'修改用户 {uname["username"] if uname else user_id} 角色为 {new_role}')
    return jsonify({'status': 'success'})


@admin_bp.route('/api/user/<int:user_id>/group', methods=['POST'])
@login_required
@admin_required
def api_set_group(user_id):
    """设置用户分组"""
    data = request.json
    group = data.get('group', '')
    admin_update_user_group(user_id, group)
    uname = get_user_by_id(user_id)
    add_activity_log(session['user_id'], 'admin', f'将用户 {uname["username"] if uname else user_id} 分到组 {group or "默认"}')
    return jsonify({'status': 'success'})


@admin_bp.route('/api/user/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def api_delete_user(user_id):
    """删除用户"""
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({'status': 'error', 'message': '用户不存在'}), 404
    if user['username'] == 'admin':
        return jsonify({'status': 'error', 'message': '不能删除管理员账户'}), 400
    admin_delete_user(user_id)
    add_activity_log(session['user_id'], 'admin', f'删除用户 {user["username"]}')
    return jsonify({'status': 'success'})


@admin_bp.route('/api/user/<int:user_id>/reset_password', methods=['POST'])
@login_required
@admin_required
def api_reset_password(user_id):
    """重置用户密码"""
    data = request.json
    new_pw = data.get('password', '')
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({'status': 'error', 'message': '用户不存在'}), 404
    if user['username'] == 'admin':
        return jsonify({'status': 'error', 'message': '不能重置管理员密码'}), 400
    success, msg = admin_reset_password(user_id, new_pw)
    if success:
        add_activity_log(session['user_id'], 'admin', f'重置用户 {user["username"]} 的密码')
    return jsonify({'status': 'success' if success else 'error', 'message': msg})


@admin_bp.route('/api/logs')
@login_required
@admin_required
def api_get_logs():
    """分页获取行为日志（LIMIT/OFFSET 每次只取一页，防止大结果撑爆内存）"""
    page = request.args.get('page', default=1, type=int)
    page_size = request.args.get('page_size', default=100, type=int)
    data = get_all_activity_logs_paged(page=page, page_size=page_size)
    data['status'] = 'success'
    return jsonify(data)


@admin_bp.route('/api/log_limits', methods=['GET', 'POST'])
@login_required
@admin_required
def api_log_limits():
    """
    日志存储上限：GET 查看当前值；POST 设置（-1=无上限，>=0 保留最新 N 条）。
    每新增一条日志即删除最旧一条，使存量不超过上限。
    """
    from database import get_log_limit, set_log_limit
    if request.method == 'GET':
        return jsonify({'status': 'success', 'max_logs': get_log_limit()})
    data = request.json or {}
    try:
        limit = set_log_limit(data.get('max_logs'))
    except (TypeError, ValueError):
        return jsonify({'status': 'error',
                        'message': '日志上限应为 -1 或非负整数'}), 400
    add_activity_log(session['user_id'], 'admin',
                     f'设置日志存储上限为 {limit} 条（-1=无上限）')
    return jsonify({'status': 'success', 'max_logs': limit})


@admin_bp.route('/api/search')
@login_required
@admin_required
def api_search():
    """搜索用户"""
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'status': 'success', 'users': []})
    conn = get_db()
    rows = conn.execute(
        'SELECT id, username, role, group_name, created_at, last_login FROM users WHERE username LIKE ? OR group_name LIKE ?',
        (f'%{q}%', f'%{q}%')
    ).fetchall()
    conn.close()
    return jsonify({'status': 'success', 'users': [dict(r) for r in rows]})


@admin_bp.route('/api/device_status')
@login_required
@admin_required
def api_device_status():
    """获取设备实时状态信息"""
    import torch as _torch

    # CPU
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_percent = psutil.cpu_percent(interval=0.5)
    cpu_freq = psutil.cpu_freq()
    cpu_info = f'{cpu_physical}核/{cpu_count}线程'

    # RAM
    mem = psutil.virtual_memory()
    ram_total = round(mem.total / (1024**3), 2)
    ram_used = round(mem.used / (1024**3), 2)
    ram_avail = round(mem.available / (1024**3), 2)
    ram_percent = mem.percent

    # Disk（代码启动目录）
    disk = psutil.disk_usage(os.path.abspath(os.path.dirname(__file__)))
    disk_total = round(disk.total / (1024**3), 2)
    disk_used = round(disk.used / (1024**3), 2)
    disk_free = round(disk.free / (1024**3), 2)
    disk_percent = disk.percent

    # GPU
    gpu_available = _torch.cuda.is_available()
    gpu_info = {}
    if gpu_available:
        gpu_info = {
            'name': _torch.cuda.get_device_name(0),
            'count': _torch.cuda.device_count(),
            'memory_allocated': round(_torch.cuda.memory_allocated(0) / (1024**3), 2),
            'memory_reserved': round(_torch.cuda.memory_reserved(0) / (1024**3), 2),
        }

    # NPU / XPU
    npu_available = hasattr(_torch, 'npu') and _torch.npu.is_available()
    xpu_available = hasattr(_torch, 'xpu') and _torch.xpu.is_available()

    # 系统信息
    os_name = f'{platform.system()} {platform.release()}'
    python_version = platform.python_version()

    return jsonify({
        'status': 'success',
        'os': os_name,
        'python': python_version,
        'cpu': {
            'model': cpu_info,
            'usage': cpu_percent,
            'freq': round(cpu_freq.current, 0) if cpu_freq else 0,
            'supported': True,
            'note': ''
        },
        'xpu': {
            'supported': xpu_available,
            'note': '本软件不支持 XPU' if not xpu_available else ''
        },
        'gpu': {
            'supported': gpu_available,
            'info': gpu_info,
            'note': '本软件不支持 GPU（PyTorch 未检测到 CUDA）' if not gpu_available else ''
        },
        'npu': {
            'supported': npu_available,
            'note': '本软件不支持 NPU' if not npu_available else ''
        },
        'ram': {
            'total': ram_total,
            'used': ram_used,
            'available': ram_avail,
            'usage': ram_percent
        },
        'disk': {
            'total': disk_total,
            'used': disk_used,
            'free': disk_free,
            'usage': disk_percent
        }
    })


@admin_bp.route('/api/monitor')
@login_required
@admin_required
def api_monitor():
    """实时监控 CPU 和 RAM 占用（前端轮询用）"""
    cpu_percent = psutil.cpu_percent(interval=0.3)
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()
    ram_percent = mem.percent
    ram_used = round(mem.used / (1024**3), 2)
    ram_total = round(mem.total / (1024**3), 2)
    return jsonify({
        'status': 'success',
        'cpu': {
            'usage': cpu_percent,
            'physical': cpu_physical,
            'logical': cpu_logical,
        },
        'ram': {
            'usage': ram_percent,
            'used': ram_used,
            'total': ram_total,
        }
    })


@admin_bp.route('/api/queue')
@login_required
@admin_required
def api_get_queue():
    """获取训练队列状态"""
    q = get_queue_status()
    return jsonify({'status': 'success', 'queue': q})


@admin_bp.route('/api/queue/max_concurrent', methods=['POST'])
@login_required
@admin_required
def api_set_max_concurrent():
    """设置最大并发训练数"""
    data = request.json
    n = int(data.get('max_concurrent', 5))
    set_max_concurrent(n)
    add_activity_log(session['user_id'], 'admin', f'设置最大并发训练数为 {n}')
    return jsonify({'status': 'success'})


@admin_bp.route('/api/queue/cancel/<task_id>', methods=['POST'])
@login_required
@admin_required
def api_cancel_task(task_id):
    """取消/停止训练任务"""
    success, msg = cancel_task(task_id)
    if success:
        add_activity_log(session['user_id'], 'admin', f'取消训练任务 {task_id}')
    return jsonify({'status': 'success' if success else 'error', 'message': msg})


@admin_bp.route('/api/bandwidth')
@login_required
@admin_required
def api_get_bandwidth():
    """获取带宽统计"""
    stats = get_bandwidth_stats()
    return jsonify({'status': 'success', 'bandwidth': stats})


@admin_bp.route('/api/bandwidth/default', methods=['POST'])
@login_required
@admin_required
def api_set_default_bandwidth():
    """设置默认带宽限制"""
    data = request.json
    mbps = float(data.get('mbps', 10))
    set_default_bandwidth(mbps)
    add_activity_log(session['user_id'], 'admin', f'设置默认带宽限制为 {mbps} Mbps')
    return jsonify({'status': 'success'})


@admin_bp.route('/api/bandwidth/user/<user_id>', methods=['POST'])
@login_required
@admin_required
def api_set_user_bandwidth(user_id):
    """设置用户单独带宽限制"""
    data = request.json
    mbps = float(data.get('mbps', 10))
    set_user_bandwidth_limit(user_id, mbps)
    add_activity_log(session['user_id'], 'admin', f'设置用户 {user_id} 带宽限制为 {mbps} Mbps')
    return jsonify({'status': 'success'})


# ==================== 存储清理 ====================
# ==================== 资源占用上限 ====================
@admin_bp.route('/api/resource_limits', methods=['GET', 'POST'])
@login_required
@admin_required
def api_resource_limits():
    """
    GET  ：查看资源上限（含系统总量、默认值来源与当前占用）
    POST ：设置上限；传 0 表示不限制。未配置过时默认为系统资源的一半
    """
    from resource_limits import get_limits, set_limits, current_usage
    if request.method == 'GET':
        data = get_limits()
        data['usage'] = current_usage()
        return jsonify({'status': 'success', 'limits': data})

    data = request.json or {}
    try:
        cfg = set_limits(max_cpu_threads=data.get('max_cpu_threads'),
                         max_memory_mb=data.get('max_memory_mb'))
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'message': '参数格式错误'}), 400
    add_activity_log(session['user_id'], 'admin',
                     f"设置资源上限：CPU {cfg['max_cpu_threads']} 线程 / "
                     f"内存 {cfg['max_memory_mb']} MB（0=不限）")
    return jsonify({'status': 'success', 'limits': cfg})


@admin_bp.route('/api/cleanup/status')
@login_required
@admin_required
def api_cleanup_status():
    """存储占用统计与清理配置"""
    from cleanup import cleanup_status
    return jsonify({'status': 'success', 'cleanup': cleanup_status()})


@admin_bp.route('/api/cleanup/config', methods=['POST'])
@login_required
@admin_required
def api_cleanup_config():
    """设置自动清理时效（retention_days=0 表示关闭自动清理）"""
    from cleanup import set_cleanup_config, get_cleanup_config
    data = request.json or {}
    cfg = set_cleanup_config(
        retention_days=data.get('retention_days'),
        auto_clean_models=data.get('auto_clean_models'),
        auto_clean_uploads=data.get('auto_clean_uploads'),
    )
    add_activity_log(session['user_id'], 'admin',
                     f"设置自动清理：保留 {cfg['retention_days']} 天（0=关闭）")
    return jsonify({'status': 'success', 'config': get_cleanup_config()})


@admin_bp.route('/api/cleanup/run', methods=['POST'])
@login_required
@admin_required
def api_cleanup_run():
    """手动清理：targets 为空列表时清全部（模型/上传数据/日志）"""
    from cleanup import run_cleanup
    data = request.json or {}
    targets = [t for t in data.get('targets', ['models', 'uploads', 'logs'])
               if t in ('models', 'uploads', 'logs')]
    if not targets:
        return jsonify({'status': 'error', 'message': '未选择清理目标'}), 400
    detail = run_cleanup(targets=targets)
    total_freed = sum(d['freed'] for d in detail.values())
    add_activity_log(session['user_id'], 'admin',
                     f'手动清理 {",".join(targets)}，释放 {total_freed / 1024 / 1024:.1f} MB')
    return jsonify({'status': 'success', 'detail': detail,
                    'freed_mb': round(total_freed / 1024 / 1024, 2)})