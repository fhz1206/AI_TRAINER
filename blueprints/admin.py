"""admin.py — 管理员面板蓝图"""
import os
import platform
import psutil
from flask import Blueprint, render_template, request, jsonify, session, redirect
from blueprints.utils import login_required
from database import is_admin, get_all_users, get_user_by_id
from database import admin_update_user_role, admin_update_user_group, admin_delete_user
from database import admin_reset_password, get_all_activity_logs, add_activity_log
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
    """获取所有行为日志"""
    logs = get_all_activity_logs(limit=500)
    return jsonify({'status': 'success', 'logs': logs})


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