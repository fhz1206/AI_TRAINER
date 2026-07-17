"""profile.py — 账号主页蓝图（信息展示、修改密码、修改名称、行为日志）"""
from flask import Blueprint, render_template, request, jsonify, session, redirect
from blueprints.utils import login_required, get_current_username
from database import get_user_by_id, update_username, update_password, get_user_activity_logs

profile_bp = Blueprint('profile', __name__)


@profile_bp.route('/profile')
@login_required
def profile_page():
    """账号主页"""
    user = get_user_by_id(session['user_id'])
    if user is None:
        session.clear()
        return redirect('/login')
    return render_template('profile.html', username=user['username'])


@profile_bp.route('/api/profile/info')
@login_required
def api_profile_info():
    """获取用户信息 + 统计"""
    user = get_user_by_id(session['user_id'])
    if not user:
        return jsonify({'status': 'error', 'message': '用户不存在'}), 404

    from database import get_file_count, get_model_count
    file_count = get_file_count(session['user_id'])
    model_count = get_model_count(session['user_id'])

    return jsonify({
        'status': 'success',
        'user': {
            'id': user['id'],
            'username': user['username'],
            'created_at': user['created_at'],
            'last_login': user['last_login'],
            'file_count': file_count,
            'model_count': model_count
        }
    })


@profile_bp.route('/api/profile/update_username', methods=['POST'])
@login_required
def api_update_username():
    """修改用户名"""
    data = request.json
    new_username = data.get('username', '').strip()
    success, message = update_username(session['user_id'], new_username)
    if success:
        session['username'] = new_username
        from database import add_activity_log
        add_activity_log(session['user_id'], 'profile', f'修改用户名为 {new_username}')
    return jsonify({'status': 'success' if success else 'error', 'message': message})


@profile_bp.route('/api/profile/update_password', methods=['POST'])
@login_required
def api_update_password():
    """修改密码"""
    data = request.json
    old_pw = data.get('old_password', '')
    new_pw = data.get('new_password', '')
    success, message = update_password(session['user_id'], old_pw, new_pw)
    if success:
        from database import add_activity_log
        add_activity_log(session['user_id'], 'profile', '修改了登录密码')
    return jsonify({'status': 'success' if success else 'error', 'message': message})


@profile_bp.route('/api/profile/activity_logs')
@login_required
def api_activity_logs():
    """获取行为日志"""
    logs = get_user_activity_logs(session['user_id'], limit=100)
    return jsonify({'status': 'success', 'logs': logs})