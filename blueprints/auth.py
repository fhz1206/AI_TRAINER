from flask import Blueprint, request, jsonify, session
# 新增导入公共装饰器
from .utils import login_required
from database import register_user, verify_user, get_user_by_id, get_file_count, get_model_count

auth_bp = Blueprint('auth', __name__, url_prefix='/api')

@auth_bp.route('/register', methods=['POST'])
def api_register():
    """用户注册（和原有逻辑完全一致）"""
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    success, message = register_user(username, password)
    
    if success:
        return jsonify({'status': 'success', 'message': message})
    else:
        return jsonify({'status': 'error', 'message': message}), 400

@auth_bp.route('/login', methods=['POST'])
def api_login():
    """用户登录（和原有逻辑完全一致）"""
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    success, user_id = verify_user(username, password)
    
    if success:
        session['user_id'] = user_id
        session['username'] = username
        session.permanent = True
        return jsonify({
            'status': 'success',
            'message': '登录成功',
            'user': {
                'id': user_id,
                'username': username
            }
        })
    else:
        return jsonify({'status': 'error', 'message': '用户名或密码错误'}), 401

@auth_bp.route('/user_info')
@login_required
def api_user_info():
    """获取当前用户信息（和原有逻辑完全一致）"""
    user = get_user_by_id(session['user_id'])
    if user:
        file_count = get_file_count(session['user_id'])
        model_count = get_model_count(session['user_id'])
        return jsonify({
            'status': 'success',
            'user': {
                'username': user['username'],
                'created_at': user['created_at'],
                'last_login': user['last_login'],
                'file_count': file_count,
                'model_count': model_count
            }
        })
    return jsonify({'status': 'error', 'message': '用户不存在'}), 404